import os
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 pipeline_v2.py <benign_raw> <malware_raw> <output_workspace>")
        sys.exit(1)

    raw_benign = sys.argv[1]
    raw_malware = sys.argv[2]
    workspace = sys.argv[3]

    dirs = {
        "cfg_b": os.path.join(workspace, "cfg/c_b"),
        "cfg_m": os.path.join(workspace, "cfg/c_m"),
        "nod_b": os.path.join(workspace, "nodes/n_b"),
        "nod_m": os.path.join(workspace, "nodes/n_m"),
        "edg_b": os.path.join(workspace, "edges/e_b"),
        "edg_m": os.path.join(workspace, "edges/e_m")
    }

    print("Creating directory structure...")
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    # for cfg
    print("\nSTEP 1: Generating CFGs...")
    
    #benign
    files_b = [f for f in os.listdir(raw_benign) if os.path.isfile(os.path.join(raw_benign, f))]
    print(f"Benign ({len(files_b)} files):")
    for i, f in enumerate(files_b, 1):
        in_p = os.path.join(raw_benign, f)
        print(f"[{i}/{len(files_b)}] {f}...", end=" ", flush=True)
        exit_code = os.system(f"python3 cfg3.py {in_p} {dirs['cfg_b']} > /dev/null 2>&1")
        print("Successfully done" if exit_code == 0 else "Error")

    #malware
    files_m = [f for f in os.listdir(raw_malware) if os.path.isfile(os.path.join(raw_malware, f))]
    print(f"\nMalware ({len(files_m)} files):")
    for i, f in enumerate(files_m, 1):
        in_p = os.path.join(raw_malware, f)
        print(f"[{i}/{len(files_m)}] {f}...", end=" ", flush=True)
        #exit_code = os.system(f"python3 cfg3.py {in_p} {dirs['cfg_m']} ")
        exit_code = os.system(f"python3 cfg3.py {in_p} {dirs['cfg_m']} > /dev/null 2>&1")
        print("Successfully done" if exit_code == 0 else "Error")

    # for node embedding
    print("\nSTEP 2: Generating Node Embeddings...")
    for cat in ['b', 'm']:
        cfg_dir = dirs[f'cfg_{cat}']
        nod_dir = dirs[f'nod_{cat}']
        dot_files = [f for f in os.listdir(cfg_dir) if f.endswith(".dot")]
        print(f"{'Benign' if cat == 'b' else 'Malware'} ({len(dot_files)} files):")
        for i, f in enumerate(dot_files, 1):
            in_p = os.path.join(cfg_dir, f)
  #          out_p = os.path.join(nod_dir, f.replace(".dot", ".npy"))
            print(f"[{i}/{len(dot_files)}] {f}...", end=" ", flush=True)
            exit_code = os.system(f"python3 node2vec_cfg.py {in_p} {nod_dir} > /dev/null 2>&1")
            print("Successfully done" if exit_code == 0 else "Error")

    # -for edge lists
    print("\n STEP 3: Generating Edge Lists...")
    for cat in ['b', 'm']:
        cfg_dir = dirs[f'cfg_{cat}']
        edg_dir = dirs[f'edg_{cat}']
        dot_files = [f for f in os.listdir(cfg_dir) if f.endswith(".dot")]
        print(f"{'Benign' if cat == 'b' else 'Malware'} ({len(dot_files)} files):")
        for i, f in enumerate(dot_files, 1):
            in_p = os.path.join(cfg_dir, f)
            out_p = os.path.join(edg_dir, f.replace(".dot", ".npy"))
            print(f"[{i}/{len(dot_files)}] {f}...", end=" ", flush=True)
            exit_code = os.system(f"python3 edge_final.py {in_p} {out_p} > /dev/null 2>&1")
            print("Successfully done" if exit_code == 0 else "Error")

    print("\n" + "="*30)
    print("Pipeline Complete Summary:")
    print(f"Benign Edges:  {len(os.listdir(dirs['edg_b']))}")
    print(f"Malware Edges: {len(os.listdir(dirs['edg_m']))}")
    print("="*30)

if __name__ == "__main__":
    main()
