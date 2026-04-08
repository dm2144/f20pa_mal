import angr
import os
import sys
import networkx as nx

def main():
    if len(sys.argv) != 3:
        print("Usage: python cfg3.py <binary_path> <output_dir>") #for ref
        return

    binary = os.path.abspath(os.path.expanduser(sys.argv[1])) #assigning to first argument
    out_dir = os.path.abspath(os.path.expanduser(sys.argv[2])) #2nd argument we give

    if not os.path.isfile(binary): #if the binary does exist->error
        print("Binary not found:", binary)
        return

    os.makedirs(out_dir, exist_ok=True) #if the 2nd argument (output directory) doesnt exist, will create

    print("Loading:", binary)
    proj = angr.Project(binary,auto_load_libs=False) #loads elf into angr and avoids loading shared library making the graph more clean and focusing more on the binary than library

    print("Building CFG...")
    cfg = proj.analyses.CFGFast( #builds the cfg using cfg fast
        normalize=True,
        resolve_indirect_jumps=False, #skips complex jumps
        force_complete_scan=False
    )

    print("Converting to NetworkX graph...")
    nx_graph = nx.DiGraph() #converts to networkx graph

    for node in cfg.graph.nodes(): #adds all cfg nodes
        nx_graph.add_node(node)
      
    for src, dst in cfg.graph.edges():  # Add all cfg edges
        nx_graph.add_edge(src, dst)
      
    base = os.path.basename(binary)
    dot_file = os.path.join(out_dir, base + "_cfg.dot") #creates output file name

    nx.drawing.nx_pydot.write_dot(nx_graph, dot_file) #saves in .dot file
    print("[+] CFG saved to:", dot_file)

if __name__ == "__main__":
    main()
