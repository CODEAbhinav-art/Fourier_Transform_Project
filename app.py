import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# 1. Load your stock audio
# Replace 'stock_audio.wav' with your actual filename
fs, data = wavfile.read(r'SS_PROJECT\harvard-[AudioTrimmer.com].wav')

# 2. Pre-processing: If stereo (2 channels), convert to mono
if len(data.shape) > 1:
    data = data[:, 0]

# 3. Normalization (Scaling amplitude to -1.0 to 1.0)
# This is a standard DSP practice
clean_signal = data / np.max(np.abs(data))

# 4. Create Time Axis
duration = len(clean_signal) / fs
t = np.linspace(0, duration, len(clean_signal))

# 5. Add "Interference" Noise (e.g., a 5000Hz tone)
noise_freq = 5000 
noise_signal = 0.1 * np.sin(2 * np.pi * noise_freq * t)
noisy_signal = clean_signal + noise_signal

# 6. Save the Noisy File for the project
wavfile.write("noisy_output.wav", fs, noisy_signal.astype(np.float32))

print(f"Loaded {duration:.2f} seconds of audio.")
print("Generated 'noisy_output.wav' for analysis in Step 2.")

#step 2

from scipy.fft import fft, fftfreq

# 1. Compute the Fourier Transform
n = len(noisy_signal)
# We calculate the FFT and take the absolute value (Magnitude)
yf = fft(noisy_signal)
xf = fftfreq(n, 1/fs) # This creates the frequency axis (Hz)

# 2. Plotting the results
plt.figure(figsize=(12, 6))

# We only plot the positive frequencies (first half of the array)
plt.plot(xf[:n//2], np.abs(yf[:n//2]))

plt.title("Magnitude Spectrum (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

# Zoom in on the area of interest
plt.xlim(0, 10000)
plt.savefig(r'SS_PROJECT\magnitude_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 'magnitude_spectrum.png' in SS_PROJECT folder.")

# 1. Find the target frequency index
# We look for where our frequency axis (xf) is closest to 5000Hz
target_freq = 5000
# We create a 'mask' - a range of frequencies to kill
bandwidth = 20 # How many Hz around the center to remove

# 2. Apply the mask (The "Notch Filter")
# We copy the original FFT data first
yf_clean = yf.copy()

# Find indices where frequency is between 4990 and 5010 (and the negative mirror)
indices_to_mute = np.where((np.abs(xf) >= target_freq - bandwidth) & 
                           (np.abs(xf) <= target_freq + bandwidth))

yf_clean[indices_to_mute] = 0

# 3. Visualization
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(xf[:n//2], np.abs(yf[:n//2]))
plt.title("Original Noisy Spectrum (With 5000Hz Spike)")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(xf[:n//2], np.abs(yf_clean[:n//2]), color='green')
plt.title("Filtered Spectrum (Noise Spike Removed)")
plt.xlabel("Frequency (Hz)")
plt.grid()

plt.tight_layout()
plt.show()

# step 4
from scipy.fft import ifft

# 1. Perform the Inverse FFT
# yf_clean is the spectrum where we muted the 5000Hz spike
cleaned_signal_complex = ifft(yf_clean)

# 2. Get the Real part
# Since physical audio cannot be complex/imaginary, we take the real part
cleaned_signal = np.real(cleaned_signal_complex)

# 3. Final Normalization 
# After math operations, the volume might have shifted slightly
cleaned_signal = cleaned_signal / np.max(np.abs(cleaned_signal))

# 4. Save the Final Product
wavfile.write("final_cleaned_audio.wav", fs, cleaned_signal.astype(np.float32))

print("Success! 'final_cleaned_audio.wav' has been created.")
