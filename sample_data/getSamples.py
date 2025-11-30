"""
Author: Rana Elladki
Data: 10/15/2025
Description: Get 100 random files from the entire dataset to add as a sample to github for testing. 
             For the entire dataset, install the tar ball from the QM9 database.
"""
import os
import random
import shutil

# Directories
src_dir = "/mnt/home/elladki1/CMSE802/Final_Project/cmse802_project/data/raw_data"       # full dataset folder
dst_dir = "/mnt/home/elladki1/CMSE802/Final_Project/cmse802_project/sample_data"    # new folder for samples
num_samples = 1000          # how many files to copy

# Make sure the destination exists
os.makedirs(dst_dir, exist_ok=True)

# List all files in source directory
all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# Randomly pick files
sample_files = random.sample(all_files, num_samples)

# Copy the files
for f in sample_files:
    shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

print(f"Copied {len(sample_files)} random files to '{dst_dir}/'")
