import os
import sys
import numpy as np
import networkx as nx
from node2vec import Node2Vec

def run_embedding_pipeline(graphml_input_path, output_base_folder):
    # 1. Validation & Loading
    if not os.path.exists(graphml_input_path):
        print(f"[ERROR] File {graphml_input_path} not found.")
        return

    # Check for empty files (char 0 errors)
    if os.path.getsize(graphml_input_path) == 0:
        print(f"[SKIP] {graphml_input_path} is empty. Skipping...")
        return

    try:
        # CRITICAL FIX: Use read_graphml instead of json.load
        G = nx.read_graphml(graphml_input_path)
    except Exception as e:
        print(f"[ERROR] {graphml_input_path} is not a valid GraphML file: {e}")
        return

    if G.number_of_nodes() < 2:
        print(f"[SKIP] {graphml_input_path} has too few nodes for embedding.")
        return

    print(f"--- Processing: {os.path.basename(graphml_input_path)} ---")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    # 2. Node2Vec Feature Extraction
    print("Generating Random Walks...")
    # dimensions=64 for GNN input; workers=1 to prevent zsh suspension/crashes
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=100, workers=1, quiet=True)
    
    print("Fitting Node2Vec model...")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # 3. Create Feature Matrix [Nodes, 64]
    node_list = list(G.nodes())
    embeddings = np.array([model.wv[node] for node in node_list])

    # 4. Create Edge List [2, Edges]
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edge_index = np.array([[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()]).T

    # 5. Clean Output Structure
    base_name = os.path.splitext(os.path.basename(graphml_input_path))[0]
    
    # Store all .npy files in these two central folders
    node_dir = os.path.join(output_base_folder, "nodes")
    edge_dir = os.path.join(output_base_folder, "edges")

    os.makedirs(node_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)

    node_out_path = os.path.join(node_dir, f"{base_name}.npy")
    edge_out_path = os.path.join(edge_dir, f"{base_name}.npy")

    np.save(node_out_path, embeddings)
    np.save(edge_out_path, edge_index)
    
    print(f"✓ SUCCESS: {base_name}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage: python3 ge_em.py <input_graphml> <base_output_directory>")
    else:
        run_embedding_pipeline(sys.argv[1], sys.argv[2])


