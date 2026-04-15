import os
import numpy as np
import glob
import shutil
import sys

def fuse_and_save(static_root, dynamic_root, output_root):
    subsets = [
        ('benign', 'n_b', 'e_b', 'benign'), 
        ('malware', 'n_m', 'e_m', 'malware')
    ]
    
    for label_name, s_node_sub, s_edge_sub, d_sub in subsets:
        s_node_dir = os.path.join(static_root, "nodes", s_node_sub)
        s_edge_dir = os.path.join(static_root, "edges", s_edge_sub) 
        d_node_dir = os.path.join(dynamic_root, "embeddings", d_sub, "nodes")
        
        out_node_dir = os.path.join(output_root, label_name, "nodes")
        out_edge_dir = os.path.join(output_root, label_name, "edges")
        os.makedirs(out_node_dir, exist_ok=True)
        os.makedirs(out_edge_dir, exist_ok=True)

        s_files = sorted(glob.glob(os.path.join(s_node_dir, "*.npy")))

        count = 0
        for sf in s_files:
            s_fname = os.path.basename(sf)
            d_fname = s_fname.replace("_cfg.npy", ".npy")
            
            df = os.path.join(d_node_dir, d_fname)
            se_path = os.path.join(s_edge_dir, s_fname)
            
            if os.path.exists(df) and os.path.exists(se_path):
                try:
                    s_emb = np.load(sf)
                    d_emb = np.load(df)
                    edge_index = np.load(se_path)
                    
                    #to determine new node limit
                    min_nodes = min(s_emb.shape[0], d_emb.shape[0])
                    
                    #trim Nodes
                    fused = np.concatenate([s_emb[:min_nodes, :], d_emb[:min_nodes, :]], axis=1)
                    
                    # Keep only edges where BOTH source and target are < min_nodes
                    mask = (edge_index[0] < min_nodes) & (edge_index[1] < min_nodes)
                    clean_edges = edge_index[:, mask]

                    np.save(os.path.join(out_node_dir, d_fname), fused)
                    np.save(os.path.join(out_edge_dir, d_fname), clean_edges)
                    count += 1
                except Exception as e:
                    print(f"Error on {s_fname}: {e}")
            
        print(f"Successfully processed {count} {label_name} samples.")

if __name__ == "__main__": # input, static, dynamic and output
    fuse_and_save(sys.argv[1], sys.argv[2], sys.argv[3])
