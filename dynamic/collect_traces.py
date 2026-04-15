import os
import sys
import subprocess
import time
from pathlib import Path

def collect_strace(input_dir, output_dir, timeout_sec=10):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    binaries = [f for f in input_path.iterdir() if f.is_file()]
    total = len(binaries)
    success = 0

    print(f"Starting collection from: {input_dir}")
    print(f"Output directory: {output_dir}")

    for i, bin_file in enumerate(binaries, 1):
        #we use .txt for raw logs to distinguish from .json/graphml later
        log_file = output_path / f"{bin_file.name}.txt"
        
        print(f"[{i}/{total}] Tracing: {bin_file.name}...", end=" ", flush=True)

        try:
            # STRACE lags where
            # -f: follow forks (child processes)
            # -o: output to file
            # -e trace=all ie to capture every syscall
            cmd = ["timeout", str(timeout_sec), "strace", "-f", "-o", str(log_file), str(bin_file.absolute())]
            
            # Run the binary. We suppress stdout/stderr so the terminal stays clean.
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Check if a log was actually created
            if log_file.exists() and log_file.stat().st_size > 0:
                print("Captured.")
                success += 1
            else:
                print("Empty/Failed.")
        
        except Exception as e:
            print(f"Error: {str(e)[:20]}")

    print(f" Collection Finished. Successfully traced {success}/{total} binaries.")
    return success

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 collect_traces.py <bin_dir> <log_out_dir>") # for ref
        sys.exit(1)

    bin_dir = sys.argv[1]
    log_dir = sys.argv[2]
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    collect_strace(bin_dir, log_dir, timeout)

if __name__ == "__main__":
    main()
