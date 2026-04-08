import os
import networkx as nx
import numpy as np
import sys

def main(dot_file, out_file):
  
    out_dir = os.path.dirname(out_file) 
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        G = nx.drawing.nx_pydot.read_dot(dot_file) #loading
        if not G.is_directed():
            G = G.to_directed()

        mapping = {node: i for i, node in enumerate(G.nodes())} #mapping where every node matches its integer index
        edges = []
        for u, v in G.edges():
            edges.append([mapping[u], mapping[v]]) #list of source with destination index lists

        if not edges:
            edge_index = np.empty((2, 0), dtype=np.int64)
        else:
            edge_index = np.array(edges, dtype=np.int64).T

        np.save(out_file, edge_index)
        print(f"Edge index saved to: {out_file}")
        print(f"Shape: {edge_index.shape}")

    except Exception as e:
        print(f"Failed to process {dot_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 edge_final.py <input.dot> <output.npy>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
