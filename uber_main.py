import os
import subprocess
import time

target_pid = 48436
print(f"Waiting for process {target_pid} to finish...", flush=True)
# Loop and wait as long as the process directory exists in /proc
while os.path.exists(f"/proc/{target_pid}"):
    time.sleep(60*5)
print(f"Process {target_pid} has finished. Initiating the über script runs...", flush=True)

files = ["phase2.py", "phase3.py"]

for i, file in enumerate(files):
    print(f"Running {file}...", flush=True)
    with open("nohup.log", "w") as f:
        result = subprocess.run(["python", "-u", file], stdout=f, stderr=f)
    print(f"Finished {file} with return code {result.returncode}", flush=True)

    if i < len(files) - 1:
        print("Waiting 60 seconds before next run...", flush=True)
        time.sleep(60*5)

print("All done!", flush=True)
