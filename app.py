import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
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
    "Upload a `.wav` file, tune the **notch filter** from the sidebar, "
    "and see the frequency-domain denoising in real time — all using "
    "pure **numpy / scipy** DSP (no ML)."
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
    st.caption("Signals & Systems Project | FFT + IFFT Notch Filter")

# ─────────────────────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload your WAV file",
    type=["wav"],
    help="Supports mono or stereo WAV files.",
)

if uploaded_file is None:
    st.info("👆 Upload a `.wav` file to get started.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# DSP Pipeline
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Processing audio…")
def run_pipeline(file_bytes: bytes, interference_freq: int,
                 filter_bandwidth: int, noise_amplitude: float):
    """Full DSP pipeline: load → normalize → add noise → FFT → notch → IFFT."""

    # 1. Load
    fs, data = wavfile.read(io.BytesIO(file_bytes))

    # 2. Mono conversion
    if len(data.shape) > 1:
        data = data[:, 0]

    # 3. Normalize
    clean_signal = data.astype(np.float64)
    clean_signal = clean_signal / np.max(np.abs(clean_signal))

    duration = len(clean_signal) / fs
    t = np.linspace(0, duration, len(clean_signal))

    # 4. Add interference noise
    noise_signal = noise_amplitude * np.sin(2 * np.pi * interference_freq * t)
    noisy_signal = clean_signal + noise_signal

    # 5. FFT
    n = len(noisy_signal)
    yf = fft(noisy_signal)
    xf = fftfreq(n, 1 / fs)

    # 6. Notch filter (zero out band around interference frequency)
    yf_clean = yf.copy()
    mask = (np.abs(xf) >= interference_freq - filter_bandwidth) & \
           (np.abs(xf) <= interference_freq + filter_bandwidth)
    yf_clean[mask] = 0

    # 7. IFFT → cleaned signal
    cleaned_signal = np.real(ifft(yf_clean))
    cleaned_signal = cleaned_signal / np.max(np.abs(cleaned_signal))

    return fs, n, xf, yf, yf_clean, noisy_signal, cleaned_signal, duration


file_bytes = uploaded_file.read()
fs, n, xf, yf, yf_clean, noisy_signal, cleaned_signal, duration = run_pipeline(
    file_bytes, interference_freq, filter_bandwidth, noise_amplitude
)

st.success(f"✅ Loaded **{duration:.2f} s** of audio  |  Sample rate: **{fs:,} Hz**")

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
    ax.set_xlim(0, 10000)
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
