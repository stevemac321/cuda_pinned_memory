import numpy as np
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python analyse_spectrum_report.py [cuda|mkl|...]")
    sys.exit(1)

prefix = sys.argv[1]
REPORT_FILE = prefix + "_report.txt"
OUTPUT_PNG = prefix + "_report_summary.png"
SAMPLE_RATE = 48000  # Hz
FFT_SIZE = 1024      # Adjust if needed

def parse_spectrum_matrix(filename):
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("Chunk "):
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[1].strip():
                    values = [float(v) for v in parts[1].strip().split()]
                    matrix.append(values)
    return np.array(matrix)

def compute_metrics(matrix, sample_rate, fft_size):
    freqs = np.arange(fft_size) * (sample_rate / fft_size)
    dominant = []
    centroid = []
    spread = []
    power = []

    for row in matrix:
        mags = np.array(row)
        if mags.size == 0 or np.sum(mags) == 0:
            dominant.append(0)
            centroid.append(0)
            spread.append(0)
            power.append(0)
            continue

        dom_bin = np.argmax(mags)
        dominant.append(freqs[dom_bin])

        c = np.sum(freqs * mags) / np.sum(mags)
        centroid.append(c)

        s = np.sqrt(np.sum(((freqs - c) ** 2) * mags) / np.sum(mags))
        spread.append(s)

        p = np.sum(mags ** 2)
        power.append(p)

    return np.array(dominant), np.array(centroid), np.array(spread), np.array(power)

def plot_summary(matrix, dominant, centroid, spread, power, output_file):
    # Use raw magnitudes, no normalization
    print("Raw magnitude stats:")
    print("  min:", np.min(matrix))
    print("  max:", np.max(matrix))
    print("  mean:", np.mean(matrix))
    print("  median:", np.median(matrix))
    print("  std dev:", np.std(matrix))
    #matrix_db = 20 * np.log10(matrix / np.max(matrix))
    matrix_db = 20 * np.log10(np.maximum(matrix, 1e-12) / np.max(matrix))
    print("dB-scaled stats:")
    print("  min:", np.min(matrix_db))
    print("  max:", np.max(matrix_db))
    print("  mean:", np.mean(matrix_db))
    print("  median:", np.median(matrix_db))
    # Fixed dB range for perceptual clarity
    vmin = -100
    vmax = 0

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # Spectrogram
    im = axs[0, 0].imshow(matrix_db, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("Spectrogram")
    axs[0, 0].set_xlabel("Frequency Bin")
    axs[0, 0].set_ylabel("Chunk Index")
    fig.colorbar(im, ax=axs[0, 0], label="Magnitude (dB)")

    # Dominant Frequency
    axs[0, 1].plot(dominant, label="Dominant Freq", color='red')
    axs[0, 1].set_title("Dominant Frequency Over Time")
    axs[0, 1].set_xlabel("Chunk Index")
    axs[0, 1].set_ylabel("Frequency (Hz)")

    # Spectral Centroid & Spread
    axs[1, 0].plot(centroid, label="Centroid", color='blue')
    axs[1, 0].plot(spread, label="Spread", color='green')
    axs[1, 0].set_title("Spectral Centroid & Spread")
    axs[1, 0].set_xlabel("Chunk Index")
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].legend()

    # Power
    axs[1, 1].plot(power, label="Power", color='purple')
    axs[1, 1].set_title("Power per Batch")
    axs[1, 1].set_xlabel("Chunk Index")
    axs[1, 1].set_ylabel("Power")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved summary plot to: {output_file}")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(REPORT_FILE):
        print(f"Report file not found: {REPORT_FILE}")
        exit(1)

    matrix = parse_spectrum_matrix(REPORT_FILE)
    dominant, centroid, spread, power = compute_metrics(matrix, SAMPLE_RATE, FFT_SIZE)
    plot_summary(matrix, dominant, centroid, spread, power, OUTPUT_PNG)