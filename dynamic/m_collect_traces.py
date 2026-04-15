import os
import subprocess
import signal
import sys
from pathlib import Path

def is_elf(file_path):
    #Checks if the file is a Linux ELF executable.
    try:
        with open(file_path, 'rb') as f:
            return f.read(4) == b'\x7fELF'
    except:
        return False

def process_samples(input_dir, output_dir):
    #creates output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output folder: {output_dir}")

    #finds all files in the input directory
    samples = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            full_path = os.path.join(root, file)
            # Only add to list if it's an ELF file and not an existing trace
            if is_elf(full_path):
                samples.append(full_path)
    
    total = len(samples)
    success_count = 0

    print(f"Starting trace collection for {total} ELF samples...")

    for index, sample_path in enumerate(samples, 1):
        sample_name = os.path.basename(sample_path)
        output_file = os.path.join(output_dir, f"{sample_name}.txt")
        
        #standard print (no \r) so it doesn't disappear
        print(f"[{index}/{total}] Processing: {sample_name}")

        try:
            #fix permissions
            os.chmod(sample_path, 0o755)

            # Run strace
            # -f: follow forks, -o: output file
            cmd = ["strace", "-f", "-o", output_file, sample_path]
            
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Kill the main process
                proc.kill()
                proc.wait()
                #secondary cleanup for any leftovers
                subprocess.run(["pkill", "-9", "-f", sample_name], stderr=subprocess.DEVNULL)
            
            # Verify the file was actually written
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                success_count += 1
            else:
                print(f"  --> Warning: No trace data produced for {sample_name}")

        except Exception as e:
            print(f"  --> Error processing {sample_name}: {e}")
            continue

    print(f"\nCompleted: {success_count}/{total} samples successfully traced.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 m_collect_traces.py.py [input_folder] [output_folder]") #for ref
    else:
        process_samples(sys.argv[1], sys.argv[2])
