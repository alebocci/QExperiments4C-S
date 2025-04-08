from functools import partial

import pennylane as qml
from pennylane import numpy as np
from mqt.bench import get_benchmark
import qiskit, json
import networkx as nx


def qasm_to_pennylane(qasm: str):
    qasm_circuit = qml.from_qasm(qasm)
    def fun():
        qasm_circuit()
    return fun

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


def cut(cut_graph):
    # Substituting the nodes to cut in the graph
    qml.qcut.replace_wire_cut_nodes(cut_graph)

    # Cut the graph in fragments (and mantain information about the communication)
    fragments, communication_graph = qml.qcut.fragment_graph(cut_graph)

    # Convert the fragments to tapes
    fragment_tapes = [qml.qcut.graph_to_tape(f) for f in fragments]

    print("------------------ Fragments ------------------", flush=True)
    print("Fragments number: ", len(fragment_tapes), flush=True)
    i = 1
    for f in fragment_tapes:
        #print(f.draw())
        print(f"Fragment {i} of size {len(f.wires)}:", flush=True)
        i += 1
        print()
    print("------------------------------------", flush=True)

    # Creation of fragments varations
    expanded = [qml.qcut.expand_fragment_tape(t) for t in fragment_tapes]

    configurations = []
    prepare_nodes = []
    measure_nodes = []
    for tapes, p, m in expanded:
        configurations.append(tapes)
        prepare_nodes.append(p)
        measure_nodes.append(m)

    tapes = tuple(tape for c in configurations for tape in c)
    #print(tapes)
    print("------------------ Variations ------------------", flush=True)
    print("Variations number: ", len(tapes), flush=True)

def cut2(cut_graph, nvariations_max, nqubits, max_var_qubits):
    # Substituting the nodes to cut in the graph
    qml.qcut.replace_wire_cut_nodes(cut_graph)

    # Cut the graph in fragments (and mantain information about the communication)
    fragments, communication_graph = qml.qcut.fragment_graph(cut_graph)

    # Convert the fragments to tapes
    fragment_tapes = [qml.qcut.graph_to_tape(f) for f in fragments]
    if len(fragment_tapes) > nvariations_max:
        print (f"Number of fragments exceeds the limit: {len(fragment_tapes)} > {nvariations_max}", flush=True)
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
            print (f"Number of variations exceeds the limit: {len_tot} > {nvariations_max}", flush=True)
            return None
        
    #for tapes, p, m in expanded:
    #    configurations.append(tapes)
    #    prepare_nodes.append(p)
    #    measure_nodes.append(m)

    #tapes = tuple(tape for c in configurations for tape in c)

    #print(tapes)
    i = 1
    frag_q = []
    for f in fragment_tapes:
        #print(f.draw())
        if len(f.wires) > max_var_qubits:
            print(f"Fragment {i} exceeds the limit of qubits: {len(f.wires)} > {max_var_qubits}", flush=True)
            return None
        # print(f"Fragment {i} of size {len(f.wires)}:")
        frag_q.append(len(f.wires))
        i += 1
        
    print(f"------------------ Circuit of {nqubits} qubits ------------------", flush=True)
    print("Fragments number: ", len(fragment_tapes), flush=True)
    
    for f in fragment_tapes:
        print(f"Fragment of size {len(f.wires)}:", flush=True)

    print("Variations number: ", len_tot, flush=True)
    print("------------------------------------", flush=True)

    return (len(fragment_tapes), len_tot, frag_q)



#qubits =  n*r + k*(r-1) -> r (n + k) - k 
# circuit_qubits = [(nquibt, ncircuits)]
def generate_configurations(min_qubits, max_qubits, step, ncircuits , variations_max, max_var_qubits, search_space):
    configurations = []
    try_conf = {}
    
    print(f"Generating configurations for {min_qubits} to {max_qubits} qubits, step {step}, {ncircuits} circuits, {variations_max} variations, {max_var_qubits} max qubits per variation, {search_space} search space", flush=True)

    for n in range(1, max_qubits+1):
        for r in range(1, n+1):
            for k in range(1, n+1):
                nqubits = n*r + k*(r-1)
                if nqubits < min_qubits or nqubits > max_qubits:
                    continue
                if nqubits not in try_conf:
                    try_conf[nqubits] = []
                try_conf[nqubits].append((n, r, k))
    
    #print(try_conf)

    for nqubits in range(min_qubits, max_qubits+1, step):
        print(f"Generating configurations for {nqubits} qubits", flush=True)
        cut_args = {
            # "num_fragments": 2,
            "max_free_wires": nqubits-1,
            "min_free_wires": 2, 
            "num_fragments_probed":(2, nqubits//2+1)
        }
        nqbuit_tries = 0
    
        for n, r, k in try_conf[nqubits]:
            conf = {
            "n": n,
            "r": r,
            "k": k
            }

            if nqbuit_tries >= ncircuits:
                break
            for layers in [1]:
                conf["layers"] = layers
                if nqbuit_tries >= ncircuits:
                    break
                for seed in range(0, search_space+1):
                    conf["seed"] = seed
                    if nqbuit_tries >= ncircuits:
                        break
                    try:
                        circuit_qaoa = generate_qaoa_maxcut_circuit(**conf)
                    except IndexError:
                        print(f"Error: index error: n={n}, r={r}, k={k}, layers={layers}, seed={seed}, nqubts={nqubits}", flush=True)
                        break
                    if circuit_qaoa.num_wires != nqubits:
                        print(f"Error: number of qubits does not match: n={n}, r={r}, k={k}, layers={layers}, seed={seed}, nqiubts={nqubits}", flush=True)
                        continue
                    
                    try:
                        cut_graph = qml.qcut.find_and_place_cuts(
                            graph = qml.qcut.tape_to_graph(circuit_qaoa),
                            cut_strategy = qml.qcut.CutStrategy(**cut_args),
                        )
                    except ValueError:
                        # print(f"Cut constraints too strict: n={n}, r={r}, k={k}, layers={layers}, seed={seed}, nqubts={nqubits}", flush=True)
                        continue
                    cut = cut2(cut_graph, variations_max, nqubits, max_var_qubits)
                    if cut:
                        nfrag, nvars, varsq = cut
                        configurations.append((conf.copy(), {"nqubits":nqubits,"nfrags": nfrag, "nvars": nvars, "varsq": varsq}))
                        nqbuit_tries += 1
                    
        if nqbuit_tries < ncircuits:
            print(f"Error: not enough configurations for {nqubits} qubits, found {nqbuit_tries}", flush=True)
            
    return configurations


min_qubits = 30
max_qubits = min_qubits
step = 5
ncircuits = 10
variations_max = 15
max_var_qubits = 20
search_space = 50

configurations = generate_configurations(min_qubits, max_qubits, step, ncircuits, variations_max, max_var_qubits, search_space)
print(configurations)
json.dump(configurations, open(f"minQ={min_qubits}_maxQ={max_qubits}_step={step}_ncirc={ncircuits}_varMax={variations_max}_maxQvar={max_var_qubits}_search={search_space}.json", "w"))
exit()

# n = 4
# r = 2
# k = 2
# p = 2 # number of layers
# # seed = 42, 24

configurations = [
    {
        "n": 4,
        "r": 3,
        "k": 1,
        "layers": 2,
        "seed": 42
    },
    {
        "n": 4,
        "r": 3,
        "k": 1,
        "layers": 2,
        "seed": 4224
    }
]

for conf in configurations:

    circuit_qaoa = generate_qaoa_maxcut_circuit(**conf)

    print("wires:", circuit_qaoa.num_wires)

    qubits = circuit_qaoa.num_wires
    # print("qubits:", n*r + k*(r-1))

    #cut_args = 
    cut_args = {
        # "num_fragments": 2,
        "max_free_wires": qubits,
        "min_free_wires": 2, 
        "num_fragments_probed":(2, qubits//2+1)
        }

    # graph = qml.qcut.tape_to_graph(circuit_qaoa)
    # cut_graph_1 = qml.qcut.find_and_place_cuts(
    #     graph = graph,
    #     num_fragments=cut_args['num_fragments'],
    # )
    # print("*** Find and place cuts ***")
    # cut(cut_graph_1)
    # del cut_args["num_fragments"]

    cut_graph_2 = qml.qcut.find_and_place_cuts(
        graph = qml.qcut.tape_to_graph(circuit_qaoa),
        cut_strategy = qml.qcut.CutStrategy(**cut_args),
    )
    print("*** CutStrategy ***")
    cut(cut_graph_2)
