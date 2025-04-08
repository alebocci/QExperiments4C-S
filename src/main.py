import sys, os, logging, configparser, datetime, json, importlib, subprocess
from qaoa import generate_qaoa_maxcut_circuit
from qiskit import QuantumCircuit
import exp as exp
from utils import hash_circuit
from datetime import datetime

ROOT_DIR = "../experiments_results"
if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

my_logger = logging.getLogger('main')
logging.basicConfig(level=logging.ERROR)
my_logger.setLevel(logging.INFO)

#ch = logging.StreamHandler()
#ch.setLevel(level)


if __name__ == "__main__":
    my_logger.info("Starting.")
    config_file = None
    start = datetime.now()
    
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        config_file = "./config.ini"
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    my_logger.debug("Config File: "+ str(config_file))

    settings = config["SETTINGS"]
    if "experiment_id" not in settings:
        now = datetime.datetime.now()
        exp_id = now.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        exp_id = config["SETTINGS"]["experiment_id"]
    
    if "git_push" not in config["SETTINGS"] or config["SETTINGS"]["git_push"] != "True":
        git_push = False
    else:
        git_push = True
    
    if "stop_on_error" not in config["SETTINGS"] or config["SETTINGS"]["stop_on_error"] != "True":
        stop_on_error = False
    else:
        stop_on_error = True

    if "retry_on_error" not in config["SETTINGS"] or config["SETTINGS"]["retry_on_error"] != "True":
        retry_on_error = False
    else:
        retry_on_error = True
    
    if "resume" not in config["SETTINGS"] or config["SETTINGS"]["resume"] != "True":
        resume = False
    else:
        resume = True

    if "parallel" not in config["SETTINGS"] or config["SETTINGS"]["Parallel"] != "True":
        parallel_execution = False
    else:
        parallel_execution = True

    if "combinations" not in config["SETTINGS"]:
        combinations = ["vanilla","cc","sw","cc_sw"]
    else:
        combinations = json.loads(config["SETTINGS"]["combinations"])

    if "stop_time" not in config["SETTINGS"]:
        stop_time_str = None
    else:
        stop_time_str = config["SETTINGS"]["stop_time"]    
        stop_time = datetime.strptime(stop_time_str, '%H:%M').time()
    if "start_time" not in config["SETTINGS"]:
        start_time_str = None
    else:
        start_time_str = config["SETTINGS"]["start_time"]    
        start_time = datetime.strptime(start_time_str, '%H:%M').time()

    timing = start_time_str is not None and stop_time_str is not None

    shots = int(json.loads(config["SETTINGS"]["shots"]))

    circuits_configuration_file = json.loads(config["SETTINGS"]["circuits"])
    all_backends = json.loads(config["BACKENDS"]["backends"])

    backend_distribution = json.loads(config["MODULES"]["backend_distribution"])
    backend_distribution_module = importlib.import_module(backend_distribution)

    cut_strategy = json.loads(config["MODULES"]["cut_strategy"])
    cut_strategy_module = importlib.import_module(cut_strategy)
    

    shotwise_policies = json.loads(config["MODULES"]["shotwise_policies"])
    sw_policies_modules = []
    for policy in shotwise_policies:
        sw_policies_modules.append(importlib.import_module(policy))
    
    shot_allocation_policies = json.loads(config["MODULES"]["shots_allocation"])
    shots_allocation_modules = []
    for policy in shot_allocation_policies:
        shots_allocation_modules.append(importlib.import_module(policy))

    circuits_configurations = json.load(open(circuits_configuration_file, "r"))

    EXPS_DIR = os.path.join(ROOT_DIR, exp_id)
    if not os.path.exists(EXPS_DIR):
        os.makedirs(EXPS_DIR)
    
    error_comb = []
    for comb in combinations:
        if comb != "vanilla" and comb != "cc" and comb != "sw" and comb != "cc_sw":
            my_logger.error(f"Invalid combination: {comb}. Skipping it.")
            error_comb.append(comb)
            continue
        folder = os.path.join(EXPS_DIR, comb)
        if not os.path.exists(folder):
            os.makedirs(folder)
    if error_comb:
        for comb in error_comb:
            combinations.remove(comb)
    
    PERF_EXP_VAL_DIR = os.path.join(EXPS_DIR, "perf_exp_vals")
    if not os.path.exists(PERF_EXP_VAL_DIR):
        os.makedirs(PERF_EXP_VAL_DIR)
    
    with open(os.path.join(EXPS_DIR, "config.ini"), "w") as f:
        config.write(f)

    with open(os.path.join(EXPS_DIR, "circuits_configurations.json"), "w") as f:
        json.dump(circuits_configurations, f)
    
    circuits = []
    for circuit_conf in circuits_configurations:
        if isinstance(circuit_conf, str):
            circuit_qasm = circuit_conf
            circuit_qubits = QuantumCircuit.from_qasm_str(circuit_qasm).num_qubits
            circuit_name = f"QASM_HASH:{hash_circuit(circuit_qasm)}"

        else:
            penny_circuit = generate_qaoa_maxcut_circuit(**circuit_conf)
            circuit_qasm = str(penny_circuit.to_openqasm())
            circuit_qubits = penny_circuit.num_wires
            circuit_name = f"n{circuit_conf['n']}_r{circuit_conf['r']}_k{circuit_conf['k']}_p{circuit_conf['layers']}_s{circuit_conf['seed']}"
        
        observable_string = "Z"*circuit_qubits #TODO: maybe give it as input (uno per circuito nel file dei circuiti)?
        circuits.append((circuit_qasm, observable_string, circuit_qubits, circuit_name, circuit_conf))

    experiment_settings = {
        "exp_id": exp_id,
        "combinations": combinations,
        "git_push": git_push,
        "stop_on_error": stop_on_error,
        "retry_on_error": retry_on_error,
        "resume": resume,
        "circuits": circuits,
        "backends": (backend_distribution_module,all_backends),
        "cut_strategy": cut_strategy_module,
        "shotwise_policies": sw_policies_modules,
        "shots_allocations": shots_allocation_modules,
        "shots": shots,
        "parallel_execution": parallel_execution
    }

    experiment_settings["circuits"] = exp.all_theoretical_exp_values(PERF_EXP_VAL_DIR,experiment_settings["circuits"])

    experiments = exp.all_experiments(experiment_settings)
    tot_experiments = len(experiments)

    if resume:
        my_logger.info("Resuming experiments.")
        experiments = exp.resume_experiments(EXPS_DIR, experiments)
        my_logger.info(f"Resumed {len(experiments)} experiments.")
        tot_experiments = len(experiments)
    
    failures = 0
    for experiment in experiments:
        if timing and not (start_time <= datetime.now().time() <= stop_time):
            my_logger.info("Out of time. Exiting.")
            break
        failures += exp.launch_experiment(EXPS_DIR, experiment, experiment_settings, tot_experiments) 
        tot_experiments -= 1
    
    if failures > 0:
        my_logger.error(f"Failed {failures} experiments.")

    if git_push:
        my_logger.info("Pushing the experiments.")
        subprocess.run(["bash", "push_exp.sh"])
    time = datetime.now()-start
    total_seconds = int(time.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    my_logger.info(f"Finished in {hours:02}:{minutes:02}:{seconds:02}.")

