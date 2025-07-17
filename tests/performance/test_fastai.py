import time
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Initial RAM usage: {process.memory_info().rss / 1e6:.2f} MB")


print(f"RAM usage after import: {process.memory_info().rss / 1e6:.2f} MB")

print("Framework loaded. Waiting for 30 seconds...")
time.sleep(30)
