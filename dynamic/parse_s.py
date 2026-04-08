import networkx as nx
import re
import json
import sys
import os
from networkx.readwrite import json_graph

def parse_strace_to_graph(log_file):
    G = nx.DiGraph() #creates a directed graph
    fd_owner = {} #created a file descriptor (fd)-dictionary to track syscalls created

    #regex pattern to match syscall format
    #ex if sys call is open("/etc/passwd", O_RDONLY)
    # it extracts name = open and args = "/etc/passwd", O_RDONLY
    syscall_pattern = re.compile(r'(\w+)\((.*?)\)')
    ret_pattern = re.compile(r'=\s+(-?\d+)') #extracts return value

    with open(log_file, 'r') as f: #opens the strace file
        for line in f: #to go line by line
            if '-' in line or '+++' in line: #skips errors like --- SIGINT --- or +++ exited with 0 +++
                continue

            match = syscall_pattern.search(line) #applies the regex
            if match: # for valid sys calls
                name = match.group(1)
                args = match.group(2)

                node_id = f"{name}_{G.number_of_nodes()}" #creates node
                G.add_node(node_id, label=name) #adds the node to graph

                # Find return value for FD tracking
                ret_match = ret_pattern.search(line)
                ret_val = ret_match.group(1) if ret_match else None

                #If it creates a file descriptor, remember it
                if ret_val and name in ['open', 'openat', 'socket', 'creat'] and int(ret_val) > 2:
                    fd_owner[ret_val] = node_id

                #If this call uses a known FD, draw an edge (The SCDG part)-stores the edge
                for fd, last_node in fd_owner.items():
                    if f"{fd}" in args:
                        G.add_edge(last_node, node_id) #adds an edge
                        fd_owner[fd] = node_id

    return G

if __name__ == "__main__":
   
    if len(sys.argv) < 3:
        print("Usage: python3 parse_s.py <input.txt> <output.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # 1. Build
    graph = parse_strace_to_graph(input_file)
    print(f"Parsed {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    # 2. Save (Crucial: Use node_link_data to fix the visualization error)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(json_graph.node_link_data(graph), f, indent=4)

    print(f"Successfully saved to {output_file}")
