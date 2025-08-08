import psutil, GPUtil, time

PROCESS_NAME = "CudaBigData.exe"
SAMPLE_INTERVAL = 0.5

def find_target():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == PROCESS_NAME:
            return proc
    return None

# Wait for process to start
print(f"Waiting for {PROCESS_NAME} to start...")
while True:
    target_proc = find_target()
    if target_proc:
        print(f"{PROCESS_NAME} detected (PID {target_proc.pid}) — starting monitoring.")
        break
    time.sleep(SAMPLE_INTERVAL)

# Monitor until process exits
with open("utilization_log.txt", "w") as log:
    while target_proc.is_running():
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        try:
            gpu = GPUtil.getGPUs()[0].load * 100
        except:
            gpu = 0.0
        log.write(f"{cpu:.1f},{ram:.1f},{gpu:.1f}\n")
        time.sleep(SAMPLE_INTERVAL)

print(f"{PROCESS_NAME} exited — monitoring stopped.")