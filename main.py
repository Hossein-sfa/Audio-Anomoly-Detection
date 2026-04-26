from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from net import STgramMFN, Mobilefacenet_bottleneck_setting 
from streamlit_autorefresh import st_autorefresh
import torch.nn.functional as F
from typing import Tuple
import streamlit as st
import numpy as np
import threading
import tempfile
import datetime
import librosa
import torch
import queue
import yaml
import os
import av


# SETTINGS
CHECKPOINT_PATH = "best_checkpoint.pth.tar"
CONFIG_PATH = "config.yaml"
CLASS_NAMES = ["no_event", "babycry", "glassbreak", "gunshot", "speech"]
DEFAULT_WINDOW_SEC = 10 
MAX_LOG_LINES = 300


# UI
st.set_page_config(page_title="Audio Event Detection", page_icon="🎧", layout="wide")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.25rem; }
.small { color: rgba(0,0,0,0.55); font-size: 0.92rem; }
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background: rgba(0,0,0,0.06); font-size: 0.85rem; margin-right: 6px;
}
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("🎧 Audio Event Detection")
st.caption("Upload audio or run live microphone detection.")

# LOAD MODEL + CONFIG
@st.cache_resource
def load_model_and_args() -> Tuple[torch.nn.Module, object, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    class Args:
        pass

    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)

    model = STgramMFN(
        num_classes=args.num_classes,
        c_dim=args.n_mels,
        win_len=args.win_length,
        hop_len=args.hop_length,
        bottleneck_setting=Mobilefacenet_bottleneck_setting,
        use_arcface=args.use_arcface,
        m=args.m, s=args.s, sub=args.sub,
    ).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, args, device


model, args, device = load_model_and_args()
TARGET_SR = int(args.sr)

# UPLOAD AUDIO LOADER (MP3/M4A via ffmpeg backend for librosa)
def load_uploaded_audio(uploaded_file) -> Tuple[np.ndarray, int]:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, sr=None, mono=False)
    os.remove(tmp_path)

    if y.ndim > 1:
        y = np.mean(y, axis=0)
    return y.astype(np.float32), int(sr)


# PREPROCESS
def pad_trim_np(y: np.ndarray, ns: int) -> np.ndarray:
    if len(y) < ns:
        return np.pad(y, (0, ns - len(y)))
    return y[:ns]


def preprocess_like_train(wav_1d: np.ndarray, sr_in: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if wav_1d.ndim > 1:
        wav_1d = wav_1d.mean(axis=1)

    if int(sr_in) != TARGET_SR:
        wav_1d = librosa.resample(
            wav_1d.astype(np.float32),
            orig_sr=int(sr_in),
            target_sr=TARGET_SR
        ).astype(np.float32)

    wav_1d = np.clip(wav_1d.astype(np.float32), -1.0, 1.0)

    ns = int(TARGET_SR * float(args.secs))
    wav_1d = pad_trim_np(wav_1d, ns).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=wav_1d,
        sr=TARGET_SR,
        n_fft=int(args.n_fft),
        hop_length=int(args.hop_length),
        win_length=int(args.win_length),
        n_mels=int(args.n_mels),
        power=float(args.power),
        center=True,
    )
    mel = np.log(mel + 1e-8).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    x_wav = torch.from_numpy(wav_1d).float().unsqueeze(0).to(device)
    x_mel = torch.from_numpy(mel).float().unsqueeze(0).to(device)
    return x_wav, x_mel


def infer_audio(wav_1d: np.ndarray, sr_in: int):
    x_wav, x_mel = preprocess_like_train(wav_1d, sr_in)
    with torch.no_grad():
        logits, _ = model(x_wav, x_mel, None)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    label = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else str(pred)
    return label, conf, probs


# FAST RESAMPLE FOR LIVE BUFFERING (avoid librosa per frame)
def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if x.size == 0 or sr_in == sr_out:
        return x.astype(np.float32)
    t_in = np.linspace(0, len(x) / sr_in, num=len(x), endpoint=False)
    n_out = int(len(x) * sr_out / sr_in)
    t_out = np.linspace(0, len(x) / sr_in, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


def frame_to_mono_float(frame: av.AudioFrame) -> Tuple[np.ndarray, int]:
    arr = frame.to_ndarray()
    sr_in = int(frame.sample_rate)

    if arr.ndim == 2:
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            x = arr.mean(axis=0)
        else:
            x = arr.mean(axis=1)
    else:
        x = arr

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)

    return x, sr_in


# LIVE MIC PROCESSOR WITH BACKGROUND INFERENCE WORKER
# - recv() only buffers audio and pushes ready chunks to a queue
# - a worker thread does the heavy inference
class LiveMicProcessor(AudioProcessorBase):
    def __init__(self, event_q: queue.Queue):
        self.event_q = event_q
        self.buf = np.zeros((0,), dtype=np.float32)
        self.window_sec = float(DEFAULT_WINDOW_SEC)
        self.sr_in = None

        self.chunk_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=6)

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while True:
            try:
                chunk_rs = self.chunk_q.get(timeout=0.2)
            except queue.Empty:
                continue

            label, conf, _ = infer_audio(chunk_rs, TARGET_SR)
            ts = datetime.datetime.now().strftime("%H:%M:%S")

            if label == "no_event":
                self.event_q.put((ts, "🔴", f"speech (conf={conf:.2f})"))
    

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        x, sr_in = frame_to_mono_float(frame)
        self.sr_in = sr_in

        # ALWAYS buffer frames (do NOT gate per-frame, it prevents filling the 10s window)
        self.buf = np.concatenate([self.buf, x], axis=0)

        win_in = int(self.sr_in * self.window_sec)  # 10s

        if self.buf.shape[0] >= win_in:
            chunk = self.buf[:win_in]
            self.buf = self.buf[win_in:]

            # gate on the FULL 10s chunk
            rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
            if rms < 0.003:
                # too quiet; skip inference, but keep streaming
                return frame

            chunk_rs = resample_linear(chunk, self.sr_in, TARGET_SR)

            try:
                self.chunk_q.put_nowait(chunk_rs)
            except queue.Full:
                pass

        return frame

# SESSION STATE INIT (this is the only persistence that survives reruns)
if "live_event_q" not in st.session_state:
    st.session_state.live_event_q = queue.Queue()

if "live_processor" not in st.session_state:
    st.session_state.live_processor = LiveMicProcessor(st.session_state.live_event_q)

if "log_list" not in st.session_state:
    st.session_state.log_list = []


def drain_event_queue_to_ui(max_items: int = 500):
    q = st.session_state.live_event_q
    moved = 0
    while moved < max_items:
        try:
            ts, kind, msg = q.get_nowait()
            st.session_state.log_list.append(f"[{ts}] {kind} {msg}")
            st.session_state.log_list = st.session_state.log_list[-MAX_LOG_LINES:]
            moved += 1
        except queue.Empty:
            break


# TABS
tab1, tab2 = st.tabs(["📁 Upload", "🎙️ Live Microphone"])

# TAB 1: UPLOAD
with tab1:
    st.subheader("Upload audio (WAV/MP3/FLAC/OGG/M4A)")
    
    uploaded = st.file_uploader("Choose a file", type=["wav", "mp3", "flac", "ogg", "m4a"])
    if uploaded:
        y, sr = load_uploaded_audio(uploaded)
        st.audio(uploaded)

        label, conf, _ = infer_audio(y, sr)

        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {conf:.2f}")


# TAB 2: LIVE
with tab2:
    st.subheader("Live microphone detection")

    # Auto-refresh to update UI logs
    st_autorefresh(interval=1000, key="live_refresh")

    # Drain events into UI list
    drain_event_queue_to_ui()

    processor_ref = st.session_state.live_processor
    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=lambda: processor_ref,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=False,
    )

    st.divider()

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🧹 Clear log"):
            st.session_state.log_list = []
            # clear queues
            while not st.session_state.live_event_q.empty():
                try:
                    st.session_state.live_event_q.get_nowait()
                except Exception:
                    break
            while not st.session_state.live_processor.chunk_q.empty():
                try:
                    st.session_state.live_processor.chunk_q.get_nowait()
                except Exception:
                    break

    st.markdown("### Event Log")
    if st.session_state.log_list:
        for line in st.session_state.log_list[::-1][:50]:
            st.write(line)
