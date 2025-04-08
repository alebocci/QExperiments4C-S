import utils as utils
import logging, datetime, inspect, os, multiprocessing, json
from os.path import join, dirname
from dotenv import load_dotenv
from qukit import Dispatcher, VirtualCircuit, QukitJSONEncoder
from time import process_time, perf_counter

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# create logger
my_logger = logging.getLogger("exp")
logging.basicConfig(level=logging.ERROR)
my_logger.setLevel(logging.INFO)

TIME_CUTTING = "time_cutting"
TIME_ALLOCATION = "time_allocation"
TIME_DISPATCH = "time_dispatch"
TIME_EXECUTION = "time_execution"
TIME_SYNCHRONIZATION = "time_synchronization"
TIME_COUNTS = "time_counts"
TIME_SIMULATION = "time_simulation"
TIME_MERGE = "time_merge"
TIME_EXPECTED_VALUES = "time_expected_values"
TIME_SEW = "time_sew"
TIME_TOTAL = "time_total"
TIME_EXECUTION_RETRIES = "time_execution_retries"


def single_execution(i,dispatch, string):
    dispatcher = Dispatcher()
    print(f"Executing a process as machine {string} "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    execution_results = dispatcher.run(dispatch)
    with open(f"./temp/{i}.json", "w") as f:
        json.dump(execution_results, f, cls=QukitJSONEncoder)

def parallel_execution(dispatch, times):
    processes = []
    start = process_time()
    i=0
    for provider in dispatch:
        for backend in dispatch[provider]:
            arg = {provider: {backend: dispatch[provider][backend]}}
            string = f"{provider}_{backend}"
            p = multiprocessing.Process(target=single_execution, args=(i,arg,string))
            processes.append(p)
            p.start()
            i+=1
    for p in processes:
        p.join()
    
    times = utils.record_time(times, TIME_EXECUTION, start)

    single_res = {}
    start = process_time()
    for j in range(0,i):
        with open(f"./temp/{j}.json", "r") as f:
            dict = json.load(f, object_hook=QukitJSONEncoder.decode)
        for backend in dict:
            if backend not in single_res:
                single_res[backend] = {}
            single_res[backend].update(dict[backend])

    times = utils.record_time(times, TIME_SYNCHRONIZATION, start)
    
    start = process_time()
    counts = utils.results_to_counts(single_res)
    times = utils.record_time(times, TIME_COUNTS, start)
    return counts, times
            

def execute_dispatch(dispatch,times):
    retry = True
    #retry when IBM fails
    while(retry):
        dispatcher = Dispatcher()
        
        #Execute the dispatch
        my_logger.info(f"Executing the dispatch "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        start = process_time()
        execution_results = dispatcher.run(dispatch)
        times = utils.record_time(times, TIME_EXECUTION, start)

        #Counts calculation
        my_logger.info(f"Calculating counts "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        start = process_time()
        counts = utils.results_to_counts(execution_results)
        times = utils.record_time(times, TIME_COUNTS, start)
        if counts:
            retry = False
        else:
            my_logger.info("IBM failed, retrying.")
    return counts,times


def prob_calc(counts):
    probs = {}
    for provider in counts:
        if provider not in probs:
            probs[provider] = {}
        for backend in counts[provider]:
            if backend not in probs[provider]:
                probs[provider][backend] = {}
            for circuit_id,obs, _counts in counts[provider][backend]:
                if (circuit_id,obs) not in probs[provider][backend]:
                    probs[provider][backend][(circuit_id,obs)] = {}
                probs[provider][backend][(circuit_id,obs)] = {k: v/sum(_counts.values()) for k,v in _counts.items()}
    return probs

def vanilla(
    circuit,
    observable_string,
    shots,
    provider_backend_couples,
    parallel_execution_flag,
    metadata = None
):
    
    times = {}
    my_logger.info("Starting Vanilla Run "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    my_logger.info(f"Qubits: {len(observable_string)}")
    my_logger.info(f"Observable: {observable_string}")
    my_logger.info(f"Shots: {shots}")
    my_logger.info(f"Len Provider Backend Couple: {len(provider_backend_couples)}")
    my_logger.info(f"Parallel Execution: {parallel_execution_flag}")
    my_logger.info(f"Metadata: {metadata}")
    
    initial_time = perf_counter()
    
    virtual_circuit = VirtualCircuit(circuit, metadata={"circuit_name": utils.hash_circuit(circuit), "observable": observable_string, "qubits": len(observable_string)})

    #pushing the observables in the circuit
    virtual_circuit = utils.push_obs(virtual_circuit)

    circuit_name = virtual_circuit.metadata["circuit_name"]

    #creating the dispatch
    my_logger.info(f"Creating the dispatch "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = process_time()
    dispatch = {}
    for provider, backend in provider_backend_couples:
        dispatch = utils.create_single_dispatch(dispatch, virtual_circuit, provider, backend, shots)
    times = utils.record_time(times, TIME_DISPATCH, start)

    #execution
    if not parallel_execution_flag:
        counts, times = execute_dispatch(dispatch, times)
    else:
        counts, times = parallel_execution(dispatch, times)
    
    #Probabilities calculation
    my_logger.info(f"Calculating probabilities "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    probs = prob_calc(counts) #probs = {provider: {backend: {circuit_id: {state: probability}}}}

    #Expected values calculation
    my_logger.info(f"Calculating expected values "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    final_results = {}
    time_exp_vals = {}
    for provider in probs:
        if provider not in final_results:
            final_results[provider] = {}
            time_exp_vals[provider] = {}
        for backend in probs[provider]:
            if backend not in final_results[provider]:
                final_results[provider][backend] = {}
                time_exp_vals[provider][backend] = {}
            for circuit_id_obs in probs[provider][backend]:
                if circuit_id_obs not in final_results[provider][backend]:
                    final_results[provider][backend][circuit_id_obs] = {}
                    time_exp_vals[provider][backend][circuit_id_obs] = {}
                start = process_time()
                final_results[provider][backend][circuit_id_obs] = utils.compute_expected_value(probs[provider][backend][(circuit_name,observable_string)], observable_string)
                stop = process_time()
                time_exp_vals[provider][backend][circuit_id_obs] = stop-start
    times[TIME_EXPECTED_VALUES] = time_exp_vals

    end_time = perf_counter()
    times[TIME_TOTAL] = end_time - initial_time

    params = {
        "circuits": circuit,
        "observable": observable_string,
        "shots": shots,
        "backends": provider_backend_couples,
        "operation": "vanilla",
        "metadata": metadata
    }

    results = {"params":params, "results": final_results, "times": times}
    results["stats"] = {
        "circuit_stats": virtual_circuit.describe(),
        "dispatch": dispatch,
        "counts": counts,
        "probs": probs,
    }

    return results

def cutting(
    circuit,
    observable_string,
    shots,
    cut_strategy_module,
    shots_allocation_module, 
    provider_backend_couples,
    parallel_execution_flag,
    metadata = None
):
    times = {}

    my_logger.info("Starting Cutting Only Run "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    my_logger.info(f"Qubits: {len(observable_string)}")
    my_logger.info(f"Observable: {observable_string}")
    my_logger.info(f"Shots: {shots}")
    my_logger.info(f"Cut Tool: {cut_strategy_module.__name__}")
    my_logger.info(f"Shots Allocation: {shots_allocation_module.__name__}")
    my_logger.info(f"Len Provider Backend Couple: {len(provider_backend_couples)}")
    my_logger.info(f"Parallel Execution: {parallel_execution_flag}")
    my_logger.info(f"Metadata: {metadata}")

    circuit_name = utils.hash_circuit(circuit)

    initial_time = perf_counter()
    #cut 
    my_logger.info(f"Cutting "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = process_time()
    cut_res = cut_strategy_module.cut(circuit, observable_string)
    if len(cut_res)==3: #cut_info is optional
        cut_output, sew_data, cut_info = cut_res
    else:
        cut_output, sew_data = cut_res
        cut_info = None
    times = utils.record_time(times, TIME_CUTTING, start)

    my_logger.info(f"Cut info: {cut_info}")
    vcs = utils.fragments_to_vc(cut_output)
    
    new_vcs = []
    old_vcs = {}
    for vc in vcs:
        new_vc = utils.push_obs(vc)
        new_vcs.append(new_vc)

        new_name = new_vc.metadata["circuit_name"]
        old_vc = vc.circuit
        old_vcs[new_name] = old_vc
    vcs = new_vcs

    #allocation of shots
    my_logger.info(f"Allocating shots "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = process_time()
    vcs_shots = shots_allocation_module.allocate_shots(vcs, shots)
    times = utils.record_time(times, TIME_ALLOCATION, start)

    #creating the dispatch
    my_logger.info(f"Creating the dispatch "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = process_time()
    dispatch = {}
    for provider, backend in provider_backend_couples:
        for fragment, fragment_shots in vcs_shots:
           dispatch = utils.create_single_dispatch(dispatch, fragment, provider, backend, fragment_shots)
    times = utils.record_time(times, TIME_DISPATCH, start)

    #execution
    if not parallel_execution_flag:
        counts, times = execute_dispatch(dispatch, times)
    else:
        counts, times = parallel_execution(dispatch, times)
    
    #Probabilities calculation
    my_logger.info(f"Calculating probabilities "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    probs = prob_calc(counts) #probs = {provider: {backend: {(circuit_id,obs): {state: probability}}}}

    #expected values
    my_logger.info(f"Expected values "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    exp_vals = {}
    times_exp_vals = {}
    for provider in probs:
        if provider not in exp_vals:
            exp_vals[provider] = {}
            times_exp_vals[provider] = {}
        for backend in probs[provider]:
            if backend not in exp_vals[provider]:
                exp_vals[provider][backend] = {}
                times_exp_vals[provider][backend] = {}
            single_prob = probs[provider][backend]
            start = process_time()
            #TODO USARE DOPO LA NUOVA VERSIONE DI QUKIT
            #utils.expected_values(probs, vcs)
            exp_vals[provider][backend] = utils.expected_values(single_prob, vcs, old_vcs)
            stop = process_time()
            times_exp_vals[provider][backend][(circuit_name,observable_string)] = stop-start
    times[TIME_EXPECTED_VALUES] = times_exp_vals


    #Sewing
    my_logger.info(f"Sewing "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    times_sew = {}
    final_results = {}
    for provider in exp_vals:
        if provider not in final_results:
            final_results[provider] = {}
            times_sew[provider] = {}
        for backend in exp_vals[provider]:
            if backend not in final_results[provider]:
                final_results[provider][backend] = {}
                times_sew[provider][backend] = {}
            single_prob = exp_vals[provider][backend]
            start = process_time()
            final_results[provider][backend][(circuit_name,observable_string)] = cut_strategy_module.sew(single_prob, sew_data)
            stop = process_time()
            times_sew[provider][backend] = stop-start
    times[TIME_SEW] = times_sew

    end_time = perf_counter()
    times[TIME_TOTAL] = end_time - initial_time

    params = {
            "circuits": circuit,
            "observable": observable_string,
            "shots": shots,
            "backends": provider_backend_couples,
            "cut_strategy": cut_strategy_module.__name__,
            "shots_allocation": shots_allocation_module.__name__,
            "operation": "cc",
            "metadata": metadata
        }

    results = {"params":params, "results": final_results, "times": times}
    #fragments stats in cut_output
    for i in range(len(cut_output)):
        circ, obs = cut_output[i]
        stats = vcs[i].describe()
        cut_output[i] = (circ, obs, stats)

    virtual_circuit = VirtualCircuit(circuit,{})
    results["stats"] = { 
        "circuit_stats": virtual_circuit.describe(),
        "cut_info": cut_info,
        "cut_output": cut_output,
        "dispatch": dispatch,
        "counts": counts,
        "probs": probs,
        "exp_values": exp_vals,
    }

    return results

def shot_wise(
    circuit,
    observable_string,
    shots,
    sw_policy_module,
    provider_backend_couples,
    parallel_execution_flag,
    metadata = None    
):
    
    times = {}

    my_logger.info("Starting Shot-Wise only Run "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    my_logger.info(f"Qubits: {len(observable_string)}")
    my_logger.info(f"Observable: {observable_string}")
    my_logger.info(f"Shots: {shots}")
    my_logger.info(f"Len Provider Backend Couple: {len(provider_backend_couples)}")
    my_logger.info(f"Shot-wise Policy: {sw_policy_module.__name__}")
    my_logger.info(f"Parallel Execution: {parallel_execution_flag}")
    my_logger.info(f"Metadata: {metadata}")

    
    initial_time = perf_counter()

    virtual_circuit = VirtualCircuit(circuit, metadata={"circuit_name": utils.hash_circuit(circuit), "observable": observable_string, "qubits": len(observable_string)})
    #pushing the observables in the circuit
    virtual_circuit= utils.push_obs(virtual_circuit)
    circuit_name = virtual_circuit.metadata["circuit_name"]

    #Splitting
    my_logger.info(f"Splitting "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = process_time()
    dispatch, split_coefficients = utils.create_dispatch([(virtual_circuit,shots)], provider_backend_couples, sw_policy_module.split)
    times = utils.record_time(times, TIME_DISPATCH, start)

    #Execution
    if not parallel_execution_flag:
        counts, times = execute_dispatch(dispatch, times)
    else:
        counts, times = parallel_execution(dispatch, times)

    #merging
    my_logger.info(f"Merging "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = process_time()
    probs, merge_coefficients =sw_policy_module.merge(counts) #probs = {(circuit_id,obs): {state: probability}}
    times = utils.record_time(times, TIME_MERGE, start)

    #Expected values calculation
    my_logger.info(f"Expected values "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = process_time()
    final_result = utils.compute_expected_value(probs[(circuit_name,observable_string)], observable_string)
    times = utils.record_time(times, TIME_EXPECTED_VALUES, start)

    end_time = perf_counter()
    times[TIME_TOTAL] = end_time - initial_time

    params = {
            "circuits": circuit,
            "observable": observable_string,
            "shots": shots,
            "backends": provider_backend_couples,
            "shot_wise_policy": sw_policy_module.__name__,
            "operation": "sw",
            "metadata": metadata
        }

    results = {"params":params, "results": final_result, "times": times}
    results["stats"] = {
        "circuit_stats": virtual_circuit.describe(),
        "dispatch": dispatch,
        "counts": counts,
        "probs": probs,
        "split_coefficients": split_coefficients,
        "merge_coefficients": merge_coefficients,
    }

    return results

def make_it_json_serializable(experiment_results):
    dispatch = experiment_results["stats"]["dispatch"]
    new_dispatch = {} #make it json serializable
    for provider in dispatch:
        for backend in dispatch[provider]:
            if provider not in new_dispatch:
                new_dispatch[provider] = {}
            if backend not in new_dispatch[provider]:
                new_dispatch[provider][backend] = []
            for frag, shots in dispatch[provider][backend]:
                new_dispatch[provider][backend].append((utils.hash_circuit(frag.circuit), shots))
    experiment_results["stats"]["dispatch"] = new_dispatch

    if type(experiment_results["stats"]["probs"]) == dict:
        probs = experiment_results["stats"]["probs"]
        new_probs = {}
        if type(list(probs.keys())[0]) == tuple:
            for k in probs:
                new_probs[str(k)] = probs[k]        
        else:
            for provider in probs:
                if provider not in new_probs:
                    new_probs[provider] = {}
                for backend in probs[provider]:
                    if backend not in new_probs[provider]:
                        new_probs[provider][backend] = {}
                    for circ,obs in probs[provider][backend]:
                        new_probs[provider][backend][str((circ,obs))] = probs[provider][backend][(circ,obs)]    
        experiment_results["stats"]["probs"] = new_probs


    if "exp_values" in experiment_results["stats"]:
        qasm_obs_exp_values = experiment_results["stats"]["exp_values"]
        exp_vals = {}
        if type (list(qasm_obs_exp_values.keys())[0]) == tuple:
            for qasm, obs in qasm_obs_exp_values:
                hash = utils.hash_circuit(qasm)
                exp_vals[str((hash,obs))] = qasm_obs_exp_values[(qasm, obs)]
        else:
            for provider in qasm_obs_exp_values:
                if provider not in exp_vals:
                    exp_vals[provider] = {}
                for backend in qasm_obs_exp_values[provider]:
                    if backend not in exp_vals[provider]:
                        exp_vals[provider][backend] = {}
                    for qasm, obs in qasm_obs_exp_values[provider][backend]:
                        hash = utils.hash_circuit(qasm)
                        exp_vals[provider][backend][str((hash,obs))] = qasm_obs_exp_values[provider][backend][(qasm, obs)]
        experiment_results["stats"]["exp_values"] = exp_vals
    
    if type(experiment_results["times"]["time_expected_values"]) == dict:
        tev = {}
        for provider in experiment_results["times"]["time_expected_values"]:
            if provider not in tev:
                tev[provider] = {}
            for backend in experiment_results["times"]["time_expected_values"][provider]:
                if backend not in tev[provider]:
                    tev[provider][backend] = {}
                for circ,obs in experiment_results["times"]["time_expected_values"][provider][backend]:
                    tev[provider][backend][str((circ,obs))] = experiment_results["times"]["time_expected_values"][provider][backend][(circ,obs)]
        experiment_results["times"]["time_expected_values"] = tev

    if type(experiment_results["results"]) == dict:
        res = {}
        for provider in experiment_results["results"]:
            if provider not in res:
                res[provider] = {}
            for backend in experiment_results["results"][provider]:
                if backend not in res[provider]:
                    res[provider][backend] = {}
                for circ,obs in experiment_results["results"][provider][backend]:
                    res[provider][backend][str((circ,obs))] = experiment_results["results"][provider][backend][(circ,obs)]
        experiment_results["results"] = res

    if type(experiment_results["error"]) == dict:
        error = {}
        for provider in experiment_results["error"]:
            if provider not in error:
                error[provider] = {}
            for backend in experiment_results["error"][provider]:
                if backend not in error[provider]:
                    error[provider][backend] = {}
                for circ,obs in experiment_results["error"][provider][backend]:
                    error[provider][backend][str((circ,obs))] = experiment_results["error"][provider][backend][(circ,obs)]
        experiment_results["error"] = error
    return experiment_results

def save_experiment(filename, exp):
    if filename is None:
        filename = "output.json"
    out = os.path.join(os.path.dirname(__file__), filename)
    with open(out, "w") as f:
        json.dump(exp, f)

if __name__ == "__main__":
    import json, configparser, importlib
    from argparse import ArgumentParser
    
    parser = ArgumentParser(prog='run_experiment.py',
                        description='Run an experiment among vanilla, only cutting or shot-wise.')
    parser.add_argument('--configfile', '-c', type=str, help='Configuration file .ini.',  default="temp/.temp.ini")
    parser.add_argument('--output', '-o', type=str, help='Specify the file name where print all the stats.',  default=None)
    parser.add_argument('--vanilla', '-va', help='Vanilla experiment.', action='store_true')
    parser.add_argument('--cutting', '-cc', help='Cutting experiment.', action='store_true')
    parser.add_argument('--shotwise', '-sw', help='Shot-wise experiment.', action='store_true')
        
    args = parser.parse_args()

    vanilla_flag = args.vanilla
    cutting_flag = args.cutting
    shotwise_flag = args.shotwise

    # Load configuration file
    config_file = args.configfile
    config = configparser.ConfigParser()
    config.read(config_file)

    settings = config["SETTINGS"]
    shots = int(json.loads(settings["shots"]))
    observable_string = json.loads(settings["observables"])

    circuit_file = json.loads(settings["circuit"])
    path = os.path.join(os.path.dirname(__file__), circuit_file)
    with open(path, "r") as f:
        circuit_qasm = f.read()

    provider_backend_couples = json.loads(settings["backends"])

    parallel_execution_flag = settings["parallel_execution"]
    if parallel_execution_flag == "True":
        parallel_execution_flag = True
    else:
        parallel_execution_flag = False

    params = {
        "circuits": circuit_qasm,
        "observable": observable_string,
        "shots": shots,
        "backends": provider_backend_couples,
        "parallel_execution": parallel_execution_flag
    }

    if cutting_flag:
        exp_type = "cc"
        cut_strategy = json.loads(settings["cut_strategy_module"])
        cut_strategy_module = importlib.import_module(cut_strategy)

        shots_allocation_name = json.loads(settings["shots_allocation_module"])
        shots_allocation_module = importlib.import_module(shots_allocation_name)
        
        params["cut_strategy"] = cut_strategy_module.__name__
        params["shots_allocation"] = shots_allocation_module.__name__

        experiment_results = cutting(
            circuit_qasm,
            observable_string,
            shots,
            cut_strategy_module,
            shots_allocation_module,
            provider_backend_couples,
            parallel_execution_flag
        )

    elif shotwise_flag:
        exp_type = "sw"

        shotwise_policy = json.loads(settings["sw_policy_module"])
        sw_policy_module = importlib.import_module(shotwise_policy)

        params["shotwise_policy"] = sw_policy_module.__name__

        experiment_results = shot_wise(
            circuit_qasm,
            observable_string,
            shots,
            sw_policy_module,
            provider_backend_couples,
            parallel_execution_flag
        )
        
    else:
        exp_type = "vanilla"
        experiment_results = vanilla(
            circuit_qasm,
            observable_string,
            shots,
            provider_backend_couples,
            parallel_execution_flag
        )
    
    FILE_OUT = args.output
   
    experiment_results["params"]["circuit_name"] = settings["circuit_name"].replace('\"', "")
    experiment_results["params"]["n_qubits"] = settings["n_qubits"]
    experiment_results["params"]["perf_exp_val"] = settings["perf_exp_val"]
    exp_val_res = experiment_results["results"]
    perf_exp_val = float(settings["perf_exp_val"])
    
    if (type(exp_val_res) == dict):
        experiment_results["error"] = {}
        for provider in exp_val_res:
            if provider not in experiment_results["error"]:
                experiment_results["error"][provider] = {}
            for backend in exp_val_res[provider]:
                if backend not in experiment_results["error"][provider]:
                    experiment_results["error"][provider][backend] = {}
                for circ_obs in exp_val_res[provider][backend]:
                    error = abs(exp_val_res[provider][backend][circ_obs] - perf_exp_val)
                    experiment_results["error"][provider][backend][circ_obs] = error
    else:
        error = abs (exp_val_res - perf_exp_val)
        experiment_results["error"] = error
    experiment_results = make_it_json_serializable(experiment_results)
    save_experiment(FILE_OUT, experiment_results)