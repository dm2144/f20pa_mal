import os
import sys
import json
import networkx as nx

def build_scdg(syscall_sequence, window=2):
    G = nx.DiGraph()
    if not syscall_sequence or len(syscall_sequence) < window:
        return None

    for syscall in set(syscall_sequence):
        G.add_node(syscall)

    for i in range(len(syscall_sequence) - window + 1):
        src = syscall_sequence[i]
        dst = syscall_sequence[i + 1]
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += 1
        else:
            G.add_edge(src, dst, weight=1)

    return G

def run_single_file(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sample_id = os.path.splitext(os.path.basename(input_path))[0]

    try:
        with open(input_path, 'r') as f:
            data = json.load(f)

        seq = []

        if isinstance(data, dict) and 'nodes' in data:
            seq = [node.get('id') for node in data['nodes'] if 'id' in node]
        elif isinstance(data, dict) and 'sequence' in data:
            seq = data['sequence']

        G = build_scdg(seq)

        if G:
            out_path = os.path.join(output_dir, f"{sample_id}.graphml")
            nx.write_graphml(G, out_path)
            return True
        return False

    except Exception as e:
        print(f"Internal Error: {e}")
        return False

if __name__ == "__main__":
    # Check if we are being called by the Batch Pipeline
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_folder = sys.argv[2]
        success = run_single_file(input_file, output_folder)
        if not success:
            sys.exit(1)  # Tell the batch script we failed
    else:
        print("Usage: python3 graph.py <input_json> <output_dir>")
        sys.exit(1)
