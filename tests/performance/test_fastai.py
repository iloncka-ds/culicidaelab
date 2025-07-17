import time
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Initial RAM usage: {process.memory_info().rss / 1e6:.2f} MB")

# Now, just import the framework
# Use the one relevant to your model (e.g., fastai or just torch)
# from fastai.vision.all import *

print(f"RAM usage after import: {process.memory_info().rss / 1e6:.2f} MB")

# Keep the script running for a moment to check the memory in your system monitor
print("Framework loaded. Waiting for 30 seconds...")
time.sleep(30)
