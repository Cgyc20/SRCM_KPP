import subprocess
import datetime

def log(msg):
    with open("logs.txt", "a") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")

log("Starting first simulation...")


subprocess.run(["python", "running_diff_thresh.py"])
log("All simulations done!")

subprocess.run(["python", "running_diff_thresh_higher_thresh.py"])

subprocess.run(["python","main.py"])


