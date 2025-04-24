#!/usr/bin/env python3
"""
Script to monitor GPU utilization during training.
This script runs in the background while training is in progress.
"""

import subprocess
import time
import argparse
import sys
import os
import signal
import datetime
import threading

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor GPU usage while running a command")
    parser.add_argument('command', nargs='+', help='Command to run')
    parser.add_argument('--interval', type=float, default=5.0, 
                        help='Monitoring interval in seconds (default: 5)')
    parser.add_argument('--log-file', type=str, default='gpu_usage.log',
                        help='Log file to save GPU usage (default: gpu_usage.log)')
    return parser.parse_args()

def get_gpu_info():
    """Get GPU utilization and memory usage."""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                         '--format=csv,noheader,nounits']).decode('utf-8')
        gpu_info = []
        for line in output.strip().split('\n'):
            values = [float(x) for x in line.split(',')]
            gpu_info.append({
                'index': int(values[0]),
                'gpu_util': values[1],
                'mem_used': values[2],
                'mem_total': values[3],
                'mem_util': values[2] / values[3] * 100 if values[3] > 0 else 0
            })
        return gpu_info
    except (subprocess.SubprocessError, FileNotFoundError):
        # No GPU available or nvidia-smi not found
        return []

def print_gpu_info(gpu_info, log_file=None):
    """Print GPU information to console and optionally to a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] GPU Utilization:")
    
    if not gpu_info:
        print("  No GPU information available")
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{timestamp} | No GPU information available\n")
        return
    
    for gpu in gpu_info:
        print(f"  GPU {gpu['index']}: {gpu['gpu_util']:.1f}% | Memory: {gpu['mem_used']/1024:.1f}/{gpu['mem_total']/1024:.1f} GB ({gpu['mem_util']:.1f}%)")
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} |")
            for gpu in gpu_info:
                f.write(f" GPU{gpu['index']}:{gpu['gpu_util']:.1f}%,{gpu['mem_used']/1024:.1f}/{gpu['mem_total']/1024:.1f}GB |")
            f.write('\n')

def monitor_gpu(interval, log_file):
    """Monitor GPU usage at specified intervals."""
    try:
        while True:
            gpu_info = get_gpu_info()
            print_gpu_info(gpu_info, log_file)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nGPU monitoring stopped")

def run_command(command):
    """Run the specified command and return the exit code."""
    try:
        process = subprocess.Popen(' '.join(command), shell=True)
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        # Forward the signal to the subprocess
        process.send_signal(signal.SIGINT)
        process.wait()
        return process.returncode

def main():
    args = parse_args()
    
    # Initialize log file
    with open(args.log_file, 'w') as f:
        f.write(f"GPU Monitoring started at {datetime.datetime.now()}\n")
        f.write(f"Command: {' '.join(args.command)}\n")
        f.write("=" * 80 + "\n")
    
    # Start GPU monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_gpu, args=(args.interval, args.log_file), daemon=True)
    monitor_thread.start()
    
    # Run the command
    print(f"Running command: {' '.join(args.command)}")
    exit_code = run_command(args.command)
    
    # Add a final GPU reading
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info, args.log_file)
    
    with open(args.log_file, 'a') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Command exited with code {exit_code} at {datetime.datetime.now()}\n")
    
    print(f"\nCommand exited with code {exit_code}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 