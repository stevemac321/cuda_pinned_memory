import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
fs = 1000  # Sampling rate in Hz
T = 1      # Duration in seconds
N = fs * T # Number of samples
t = np.linspace(0, T, N, endpoint=False)

# Create a composite signal: 50 Hz + 120 Hz + noise
signal = (
    1.0 * np.sin(2 * np.pi * 50 * t) +     # 50 Hz component
    0.5 * np.sin(2 * np.pi * 120 * t) +    # 120 Hz component
    0.2 * np.random.randn(N)              # Random noise
)

# Compute FFT
fft_vals = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(N, 1/fs)

# Only keep the positive frequencies
pos_mask = fft_freqs >= 0
fft_vals = np.abs(fft_vals[pos_mask])
fft_freqs = fft_freqs[pos_mask]

# Plot time domain
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, signal)
plt.title("Time Domain (Voltage vs Time)")
plt.xlabel("Time [s]")
plt.ylabel("Voltage")

# Plot frequency domain
plt.subplot(1, 2, 2)
plt.plot(fft_freqs, fft_vals)
plt.title("Frequency Domain (FFT Magnitude)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()