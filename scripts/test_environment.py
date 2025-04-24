#!/usr/bin/env python3
"""
Test script to verify the environment setup and GPU availability.
"""

import sys
import platform
import time
import numpy as np
import torch
import matplotlib
import gym

def print_header(title):
    print(f"\n{'=' * 40}")
    print(f"{title:^40}")
    print(f"{'=' * 40}")

def main():
    print_header("System Information")
    print(f"Python version: {sys.version}")
    print(f"OS: {platform.system()} {platform.release()}")
    
    print_header("Package Versions")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"Gym: {gym.__version__}")
    
    print_header("GPU Information")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("No CUDA-capable GPU found. Training will be on CPU only.")
    
    print_header("Test Tensor Operations")
    # CPU tensor operation
    a = torch.randn(1000, 1000)
    b = torch.randn(1000, 1000)
    print("CPU matrix multiplication... ", end="")
    start_time = time.time()
    c = torch.matmul(a, b)
    cpu_time = time.time() - start_time
    print(f"done in {cpu_time:.4f} seconds")
    
    # GPU tensor operation if available
    if torch.cuda.is_available():
        a_gpu = a.cuda()
        b_gpu = b.cuda()
        torch.cuda.synchronize()  # Ensure GPU is ready
        print("GPU matrix multiplication... ", end="")
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        print(f"done in {gpu_time:.4f} seconds")
        
        # Verify results match
        max_diff = torch.max(torch.abs(c - c_gpu.cpu())).item()
        print(f"Maximum difference between CPU and GPU results: {max_diff}")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    
    print_header("Environment Setup Complete")
    print("All tests completed successfully!")

if __name__ == "__main__":
    main() 