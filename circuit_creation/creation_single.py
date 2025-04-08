from functools import partial

import pennylane as qml
from pennylane import numpy as np
import sys, json
import networkx as nx

from typing import List, Optional, Tuple


def clustered_chain_graph(
    n: int, r: int, k: int, q1: float, q2: float, seed: Optional[int] = None
) -> Tuple[nx.Graph, List[List[int]], List[List[int]]]:
    """
    Function to build clustered chain graph

    Args:
        n (int): number of nodes in each cluster
        r (int): number of clusters
        k (int): number of vertex separators between each cluster pair
        q1 (float): probability of an edge connecting any two nodes in a cluster
        q2 (float): probability of an edge connecting a vertex separator to any node in a cluster
        seed (Optional[int]=None): seed for fixing edge generation

    Returns:
        nx.Graph: clustered chain graph
    """

    if r <= 0 or not isinstance(r, int):
        raise ValueError("Number of clusters must be an integer greater than 0")

    clusters = []
    for i in range(r):
        _seed = seed * i if seed is not None else None
        cluster = nx.erdos_renyi_graph(n, q1, seed=_seed)
        nx.set_node_attributes(cluster, f"cluster_{i}", "subgraph")
        clusters.append(cluster)

    separators = []
    for i in range(r - 1):
        separator = nx.empty_graph(k)
        nx.set_node_attributes(separator, f"separator_{i}", "subgraph")
        separators.append(separator)

    G = nx.disjoint_union_all(clusters + separators)

    cluster_nodes = [
        [n[0] for n in G.nodes(data="subgraph") if n[1] == f"cluster_{i}"] for i in range(r)
    ]
    separator_nodes = [
        [n[0] for n in G.nodes(data="subgraph") if n[1] == f"separator_{i}"] for i in range(r - 1)
    ]

    rng = np.random.default_rng(seed)

    for i, separator in enumerate(separator_nodes):
        for s in separator:
            for c in cluster_nodes[i] + cluster_nodes[i + 1]:
                if rng.random() < q2:
                    G.add_edge(s, c)

    return G, cluster_nodes, separator_nodes


def get_qaoa_circuit(
    G: nx.Graph,
    cluster_nodes: List[List[int]],
    separator_nodes: List[List[int]],
    params: Tuple[Tuple[float]],
    layers: int = 1,
) -> qml.tape.QuantumTape:
    """
    Function to build QAOA max-cut circuit tape from graph including `WireCut` 
    operations
    
    Args:
        G (nx.Graph): problem graph to be solved using QAOA
        cluster_nodes (List[List[int]]): nodes of the clusters within the graph
        separator_nodes (List[List[int]]): nodes of the separators in the graph
        params (Tuple[Tuple[float]]): parameters of the QAOA circuit to be optimized
        layers (int): number of layer in the QAOA circuit
        
    Returns:
        QuantumTape: the QAOA tape containing `WireCut` operations
    """
    wires = len(G)
    r = len(cluster_nodes)

    with qml.tape.QuantumTape() as tape:
        for w in range(wires):
            qml.Hadamard(wires=w)

        for l in range(layers):
            gamma, beta = params[l]

            for i, c in enumerate(cluster_nodes):
                if i == 0:
                    current_separator = []
                    next_separator = separator_nodes[0]
                elif i == r - 1:
                    current_separator = separator_nodes[-1]
                    next_separator = []
                else:
                    current_separator = separator_nodes[i - 1]
                    next_separator = separator_nodes[i]

                #for cs in current_separator:
                #    qml.WireCut(wires=cs)

                nodes = c + current_separator + next_separator
                subgraph = G.subgraph(nodes)

                for edge in subgraph.edges:
                    qml.IsingZZ(2*gamma, wires=edge) # multiply param by 2 for consistency with analytic cost

            # mixer layer
            for w in range(wires):
                qml.RX(2*beta, wires=w)


            # reset cuts
            #if l < layers - 1:
            #    for s in separator_nodes:
            #        qml.WireCut(wires=s)

        #[qml.expval(op) for op in cost.ops if not isinstance(op, qml.ops.identity.Identity)]
        observable = "Z"*wires
        [qml.expval(qml.pauli.string_to_pauli_word(observable))]

    return tape

def generate_qaoa_maxcut_circuit(n: int, r: int, k: int, layers: int = 1, q1: float = 0.7, q2: float = 0.3, seed: Optional[int] = None) -> qml.tape.QuantumTape:
    """
    Function to generate QAOA max-cut circuit tape from graph including `WireCut` 
    operations
    
    Args:
        n (int): number of nodes in each cluster
        r (int): number of clusters
        k (int): number of vertex separators between each cluster pair
        layers (int): number of layer in the QAOA circuit
        q1 (float): probability of an edge connecting any two nodes in a cluster
        q2 (float): probability of an edge connecting a vertex separator to any node in a cluster
        seed (Optional[int]=None): seed for fixing edge generation
        
    Returns:
        QuantumTape: the QAOA tape containing `WireCut` operations
    """
    G, cluster_nodes, separator_nodes = clustered_chain_graph(n, r, k, q1, q2, seed)
    params = ((0.1, 0.2), (0.3, 0.4))
    return get_qaoa_circuit(G, cluster_nodes, separator_nodes, params, layers)

def cut(cut_graph, nvariations_max, nqubits, max_var_qubits):
    # Substituting the nodes to cut in the graph
    qml.qcut.replace_wire_cut_nodes(cut_graph)

    # Cut the graph in fragments (and mantain information about the communication)
    fragments, communication_graph = qml.qcut.fragment_graph(cut_graph)

    # Convert the fragments to tapes
    fragment_tapes = [qml.qcut.graph_to_tape(f) for f in fragments]
    if len(fragment_tapes) > nvariations_max:
        #print ("Number of fragments exceeds the limit")
        return None

    # Creation of fragments varations
    expanded = [qml.qcut.expand_fragment_tape(t) for t in fragment_tapes]
    #configurations = []
    #prepare_nodes = []
    #measure_nodes = []
    
    len_tot = 0
    for tapes, p, m in expanded:
        len_tot += len(tapes)
        if len_tot > nvariations_max:
            #print ("Number of variations exceeds the limit")
            return None
        
    #for tapes, p, m in expanded:
    #    configurations.append(tapes)
    #    prepare_nodes.append(p)
    #    measure_nodes.append(m)

    #tapes = tuple(tape for c in configurations for tape in c)
   
    #print(tapes)
    #print(f"------------------ Circuit of {nqubits} qubits ------------------")
    #print("Fragments number: ", len(fragment_tapes))
    i = 1
    frag_q = []
    for f in fragment_tapes:
        #print(f.draw())
        if len(f.wires) > max_var_qubits:
            #print(f"Fragment {i} of size {len(f.wires)} exceeds the limit of {max_var_qubits} qubits by {len(f.wires) - max_var_qubits} qubits")
            return None
        #print(f"Fragment {i} of size {len(f.wires)}:")
        frag_q.append(len(f.wires))
        i += 1
        #print()

    #print("Variations number: ", len_tot)
    #print("------------------------------------")

    return (len(fragment_tapes), len_tot, frag_q)

def singleConfigurationTry(nqubits, n,r,k,seed, variations_max, max_var_qubits):
    cut_args = {
        # "num_fragments": 2,
        "max_free_wires": nqubits-1,
        "min_free_wires": 2, 
        "num_fragments_probed":(2, nqubits//2+1)
    }
    conf = {
    "n": n,
    "r": r,
    "k": k,
    "layers": 1,
    "seed": seed
    }

    try:
        circuit_qaoa = generate_qaoa_maxcut_circuit(**conf)
    except IndexError:
        #print(f"Error: index error: n={n}, r={r}, k={k}, layers=1, seed={seed}, nqubts={nqubits}")
        return None
    if circuit_qaoa.num_wires != nqubits:
        #print(f"Error: number of qubits does not match: n={n}, r={r}, k={k}, layers=1, seed={seed}, nqiubts={nqubits}")
        return None
    try:
        cut_graph = qml.qcut.find_and_place_cuts(
            graph = qml.qcut.tape_to_graph(circuit_qaoa),
            cut_strategy = qml.qcut.CutStrategy(**cut_args),
        )
    except ValueError:
        #print(f"Cut constraints too strict: n={n}, r={r}, k={k}, layers=1, seed={seed}, nqubts={nqubits}")
        return None
    cut_res = cut(cut_graph, variations_max, nqubits, max_var_qubits)
    if cut_res:
        nfrag, nvars, varsq = cut_res
        return [conf, {"nqubits":nqubits,"nfrags": nfrag, "nvars": nvars, "varsq": varsq}]

if __name__ == "__main__":
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    k = int(sys.argv[3])
    seed = int(sys.argv[4])
    nqubits = n*r + k*(r-1)

    variations_max = int(sys.argv[5])
    max_var_qubits = int(sys.argv[6])
    

    conf = singleConfigurationTry(nqubits, n,r,k,seed, variations_max, max_var_qubits)
    if conf:
        ris = str(conf).replace("'", "\"")
        print(ris+",")
        exit(0)
    else:
        exit(1)