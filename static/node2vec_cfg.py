import sys
import os
import networkx as nx
from node2vec import Node2Vec
import numpy as np

def run_node2vec(dot_file,OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.basename(dot_file)
    base = os.path.splitext(base)[0]    
    out_file = os.path.join(OUT_DIR, base + ".npy")
    G = nx.drawing.nx_pydot.read_dot(dot_file) #loading the file
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
      
    G = nx.convert_node_labels_to_integers(G) #convert hex memory to integers
    node2vec = Node2Vec( #random walk generation
        G,
        dimensions=64,
        walk_length=20,
        num_walks=10,
        workers=4
    )
    model = node2vec.fit( #training from generated walks above
        window=10,
        min_count=1,
        batch_words=128
    )
    nodes = sorted(G.nodes()) #map each node back to its vector 
    emb = np.array([model.wv[str(n)] for n in nodes])
    np.save(out_file, emb)
    print("Saved:", out_file)
    print("Embedding shape:", emb.shape)
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python node2vec_cfg.py <cfg.dot> <output path>")
        sys.exit(1)
    run_node2vec(sys.argv[1],sys.argv[2])
