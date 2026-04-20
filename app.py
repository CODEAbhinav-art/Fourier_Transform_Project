import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import soundfile as sf
from scipy.io import wavfile
from scipy.fft import fft, fftfreq, ifft

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SS Project – Audio Denoiser",
    page_icon="🎛️",
    layout="wide",
)

st.title("🎛️ Signals & Systems — FFT-Based Audio Denoiser")
st.markdown(
    "Upload an audio file, tune the **notch filter** from the sidebar, "
    "and see the frequency-domain denoising in real time — all using "
    "pure **numpy / scipy** DSP (no ML)."
)

# ─────────────────────────────────────────────────────────────
# Upload Guidelines (collapsible)
# ─────────────────────────────────────────────────────────────
with st.expander("📋 Upload Guidelines & Supported Formats", expanded=False):
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### ✅ Supported Formats")
        st.markdown("""
| Format | Extension | Notes |
|--------|-----------|-------|
| WAV    | `.wav`    | ⭐ **Ideal** — uncompressed, lossless |
| FLAC   | `.flac`   | ✅ Lossless compressed — great quality |
| OGG    | `.ogg`    | ✅ Open-source compressed format |
| AIFF   | `.aiff`/`.aif` | ✅ Apple lossless format |
| MP3    | `.mp3`    | ❌ Not supported (needs external codec) |
| M4A    | `.m4a`    | ❌ Not supported (needs external codec) |
""")

    with col_b:
        st.markdown("### 📐 Recommended Specs for Best Results")
        st.markdown("""
| Parameter | Ideal Value | Why |
|-----------|------------|-----|
| **Format** | `.wav` (PCM) | No quality loss from compression |
| **Sample Rate** | 44100 Hz or 48000 Hz | Covers full human hearing range |
| **Channels** | Mono (or Stereo) | Stereo auto-converted to mono |
| **Duration** | 5 – 60 seconds | Shorter = faster FFT processing |
| **File Size** | < 50 MB | Streamlit's upload limit is **200 MB** |
| **Amplitude** | Not clipped/distorted | Clipping causes artefacts in FFT |
""")

    st.info(
        "💡 **Why WAV is ideal:** WAV files store raw audio samples with zero compression. "
        "MP3/M4A discard some frequency data to save space, which can introduce artefacts "
        "that interfere with FFT analysis. For a Signals & Systems project, WAV gives the "
        "cleanest, most accurate frequency spectrum."
    )
    st.warning(
        "⚠️ **File size limit:** Streamlit allows uploads up to **200 MB**. "
        "A standard 44100 Hz mono WAV file at ~5 MB/min means you can upload up to ~40 minutes of audio. "
        "Practically, keep files under **30 seconds** for fast real-time response."
    )

# ─────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filter Controls")

    interference_freq = st.slider(
        "Interference Frequency (Hz)",
        min_value=1000,
        max_value=8000,
        value=5000,
        step=100,
        help="The frequency of the sine-wave noise that will be injected into the audio.",
    )

    filter_bandwidth = st.slider(
        "Filter Bandwidth (Hz)",
        min_value=5,
        max_value=500,
        value=50,
        step=5,
        help="Width of the notch on each side of the interference frequency to zero out.",
    )

    noise_amplitude = st.slider(
        "Noise Amplitude",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        help="Amplitude of injected interference relative to the clean signal.",
    )

    st.divider()
    st.markdown("### 📂 Supported Formats")
    st.markdown("""
- ⭐ `.wav` — Best quality
- ✅ `.flac` — Lossless
- ✅ `.ogg` — Compressed
- ✅ `.aiff` / `.aif` — Apple
- ❌ `.mp3` — Not supported
""")
    st.caption("Signals & Systems Project | FFT + IFFT Notch Filter")

# ─────────────────────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload your audio file",
    type=["wav", "flac", "ogg", "aiff", "aif"],
    help="Supported: WAV (ideal), FLAC, OGG, AIFF. Max 200 MB. MP3 is not supported.",
)

if uploaded_file is None:
    st.info("👆 Upload an audio file to get started. Click **'Upload Guidelines'** above for tips.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# File size check
# ─────────────────────────────────────────────────────────────
file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
if file_size_mb > 50:
    st.warning(
        f"⚠️ Your file is **{file_size_mb:.1f} MB**. Files larger than 50 MB may process slowly. "
        "For best results, use files under 30 seconds."
    )

# ─────────────────────────────────────────────────────────────
# Audio loading (multi-format via soundfile)
# ─────────────────────────────────────────────────────────────
def load_audio(file_bytes: bytes, filename: str):
    """Load audio from bytes using soundfile (WAV, FLAC, OGG, AIFF)."""
    buf = io.BytesIO(file_bytes)
    data, fs = sf.read(buf, dtype="float64", always_2d=True)
    # Mono: take first channel
    data = data[:, 0]
    return fs, data

# ─────────────────────────────────────────────────────────────
# DSP Pipeline
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Processing audio…")
def run_pipeline(file_bytes: bytes, filename: str, interference_freq: int,
                 filter_bandwidth: int, noise_amplitude: float):
    """Full DSP pipeline: load → normalize → add noise → FFT → notch → IFFT."""

    # 1. Load (multi-format)
    fs, data = load_audio(file_bytes, filename)

    # 2. Normalize
    clean_signal = data / np.max(np.abs(data))

    duration = len(clean_signal) / fs
    t = np.linspace(0, duration, len(clean_signal))

    # 3. Add interference noise
    noise_signal = noise_amplitude * np.sin(2 * np.pi * interference_freq * t)
    noisy_signal = clean_signal + noise_signal

    # 4. FFT
    n = len(noisy_signal)
    yf = fft(noisy_signal)
    xf = fftfreq(n, 1 / fs)

    # 5. Notch filter (zero out band around interference frequency)
    yf_clean = yf.copy()
    mask = (np.abs(xf) >= interference_freq - filter_bandwidth) & \
           (np.abs(xf) <= interference_freq + filter_bandwidth)
    yf_clean[mask] = 0

    # 6. IFFT → cleaned signal
    cleaned_signal = np.real(ifft(yf_clean))
    cleaned_signal = cleaned_signal / np.max(np.abs(cleaned_signal))

    return fs, n, xf, yf, yf_clean, noisy_signal, cleaned_signal, duration


try:
    file_bytes = uploaded_file.getvalue()
    fs, n, xf, yf, yf_clean, noisy_signal, cleaned_signal, duration = run_pipeline(
        file_bytes, uploaded_file.name, interference_freq, filter_bandwidth, noise_amplitude
    )
except Exception as e:
    st.error(
        f"❌ Could not read this file: `{e}`\n\n"
        "Please upload a valid **WAV, FLAC, OGG, or AIFF** file. "
        "MP3 and M4A are not supported."
    )
    st.stop()

# Format badge
fmt = uploaded_file.name.split(".")[-1].upper()
fmt_color = {"WAV": "🟢", "FLAC": "🟢", "OGG": "🟡", "AIFF": "🟡", "AIF": "🟡"}.get(fmt, "🔵")
st.success(
    f"✅ {fmt_color} **{fmt}** file loaded | "
    f"**{duration:.2f} s** duration | "
    f"Sample rate: **{fs:,} Hz** | "
    f"Size: **{file_size_mb:.2f} MB**"
)

# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

def make_spectrum_fig(xf, yf_half, n, title, color, interference_freq):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xf[:n // 2], np.abs(yf_half), color=color, linewidth=0.8)
    ax.axvline(x=interference_freq, color="red", linestyle="--",
               linewidth=1, alpha=0.6, label=f"{interference_freq} Hz")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, min(10000, fs // 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

with col1:
    st.subheader("📊 Noisy Magnitude Spectrum")
    fig_noisy = make_spectrum_fig(
        xf, yf[:n // 2], n,
        f"Noisy Spectrum (spike @ {interference_freq} Hz)",
        "#e05c5c", interference_freq,
    )
    st.pyplot(fig_noisy)
    plt.close(fig_noisy)

with col2:
    st.subheader("✅ Filtered Magnitude Spectrum")
    fig_clean = make_spectrum_fig(
        xf, yf_clean[:n // 2], n,
        "Filtered Spectrum (notch applied)",
        "#4caf76", interference_freq,
    )
    st.pyplot(fig_clean)
    plt.close(fig_clean)

# ─────────────────────────────────────────────────────────────
# Audio players
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔊 Listen & Compare")

def signal_to_wav_bytes(signal: np.ndarray, fs: int) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, fs, signal.astype(np.float32))
    return buf.getvalue()

audio_col1, audio_col2 = st.columns(2)

with audio_col1:
    st.markdown("**🔴 Noisy Audio** (with interference tone)")
    st.audio(signal_to_wav_bytes(noisy_signal, fs), format="audio/wav")

with audio_col2:
    st.markdown("**🟢 Cleaned Audio** (after notch filter)")
    st.audio(signal_to_wav_bytes(cleaned_signal, fs), format="audio/wav")

# Download buttons
dl_col1, dl_col2 = st.columns(2)
with dl_col1:
    st.download_button(
        "⬇️ Download Noisy WAV",
        data=signal_to_wav_bytes(noisy_signal, fs),
        file_name="noisy_output.wav",
        mime="audio/wav",
    )
with dl_col2:
    st.download_button(
        "⬇️ Download Cleaned WAV",
        data=signal_to_wav_bytes(cleaned_signal, fs),
        file_name="final_cleaned_audio.wav",
        mime="audio/wav",
    )

# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Filter Stats")

m1, m2, m3, m4 = st.columns(4)
bins_muted = int(np.sum(
    (np.abs(xf) >= interference_freq - filter_bandwidth) &
    (np.abs(xf) <= interference_freq + filter_bandwidth)
))
m1.metric("Interference Freq", f"{interference_freq} Hz")
m2.metric("Filter Bandwidth", f"±{filter_bandwidth} Hz")
m3.metric("FFT Bins Zeroed", bins_muted)
m4.metric("Audio Duration", f"{duration:.2f} s")
