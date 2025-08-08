import matplotlib.pyplot as plt
import csv

# Constants
TOTAL_SIZE_BYTES = 268435456
BYTES_PER_MB = 1_000_000

# Parse perf.txt
methods = {}
with open("perf.txt", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        method = row[0]
        if "MKL" in method:
            continue  # Skip MKL entries

        chunk = int(row[1])
        mbps = float(row[4])
        gflops = float(row[5])
        time_sec = TOTAL_SIZE_BYTES / (mbps * BYTES_PER_MB)
        time_ms = time_sec * 1000

        if method not in methods:
            methods[method] = {
                "chunk": [], "MB/s": [], "Time (ms)": [], "GFLOP/s": []
            }
        methods[method]["chunk"].append(chunk)
        methods[method]["MB/s"].append(mbps)
        methods[method]["Time (ms)"].append(time_ms)
        methods[method]["GFLOP/s"].append(gflops)

# Try to parse utilization_log.txt
cpu_vals, ram_vals, gpu_vals = [], [], []
try:
    with open("utilization_log.txt", "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                cpu_vals.append(float(parts[0]))
                ram_vals.append(float(parts[1]))
                gpu_vals.append(float(parts[2]))
except FileNotFoundError:
    print("utilization_log.txt not found — skipping system stats plot.")

# Plot all metrics
num_plots = 4 if cpu_vals else 3
plt.figure(figsize=(5 * num_plots, 5))

# MB/s
plt.subplot(1, num_plots, 1)
for method, data in methods.items():
    plt.plot(data["chunk"], data["MB/s"], marker='o', label=method)
plt.xlabel("Chunk Size")
plt.ylabel("Throughput (MB/s)")
plt.title("Throughput vs Chunk Size")
plt.legend()
plt.grid(True)

# Time (ms)
plt.subplot(1, num_plots, 2)
for method, data in methods.items():
    plt.plot(data["chunk"], data["Time (ms)"], marker='o', label=method)
plt.xlabel("Chunk Size")
plt.ylabel("Time per Run (ms)")
plt.title("Latency vs Chunk Size")
plt.legend()
plt.grid(True)

# GFLOP/s
plt.subplot(1, num_plots, 3)
for method, data in methods.items():
    plt.plot(data["chunk"], data["GFLOP/s"], marker='o', label=method)
plt.xlabel("Chunk Size")
plt.ylabel("GFLOP/s")
plt.title("Compute Performance vs Chunk Size")
plt.legend()
plt.grid(True)

# Utilization
if cpu_vals:
    plt.subplot(1, num_plots, 4)
    x = list(range(len(cpu_vals)))
    plt.plot(x, cpu_vals, label="CPU %", color="red")
    plt.plot(x, ram_vals, label="RAM %", color="blue")
    plt.plot(x, gpu_vals, label="GPU %", color="green")
    plt.xlabel("Sample Index")
    plt.ylabel("Utilization (%)")
    plt.title("System Utilization Over Time")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("perf.png")
plt.show()