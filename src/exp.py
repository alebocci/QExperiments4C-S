import sys, subprocess,os,json, logging
import pennylane as qml

# create logger
my_logger = logging.getLogger('main')
logging.basicConfig(level=logging.ERROR)
my_logger.setLevel(logging.INFO)

TEMP_CONF_FILE = ".temp.ini"
TEMP_QASM_FILE = ".temp.qasm"
TEMP_FOLDER = "temp"

max_tries = 3#TODO from input

def qasm_to_pennylane(qasm, observable_string):
    qasm_circuit = qml.from_qasm(qasm)
    def fun():
        qasm_circuit()
        return qml.expval(qml.pauli.string_to_pauli_word(observable_string))
    return fun

def compute_theoretical_exp_value(qasm_circuit, observable_string, nqubits):
    dev = qml.device('default.qubit', wires=nqubits)    
    node = qml.QNode(qasm_to_pennylane(qasm_circuit, observable_string), dev)
    return float(node())

def all_theoretical_exp_values(PERF_EXP_VAL_DIR,circuits):
    new_circuits = []
    for circ, obs, qubits, circuit_name, circuit_conf in circuits:
        perf_exp_val = compute_theoretical_exp_value(circ, obs, qubits)
        new_circuits.append((circ, obs, qubits, circuit_name, circuit_conf, perf_exp_val))
        file = os.path.join(PERF_EXP_VAL_DIR, f"{circuit_name}_{qubits}_{obs}.json")
        if not os.path.exists(file):
            to_save = {"circuit_conf": circuit_conf, "circuit_name": circuit_name, "n_qubits":qubits, "observable":obs, "perf_exp_val": perf_exp_val}
            with open(file, "w") as f:
                json.dump(to_save, f)
    return new_circuits

def generate_backends_name(all_backends, beckends):
    return "_".join(["b"+str(all_backends.index(beckend)) for beckend in beckends])

def all_experiments(experiments):
    combinations = experiments["combinations"]
    circuits = experiments["circuits"]
    backends = experiments["backends"] # pair of backend_distribution_module and all_backends
    cut_strategy = experiments["cut_strategy"]
    shotwise_policies = experiments["shotwise_policies"]
    shots_allocations = experiments["shots_allocations"]
    shots = experiments["shots"]
    
    exps = []
    exp_n = 0
    backends_distr = backends[0].backend_distribution(backends[1])
    all_backends = backends[1]

    my_logger.info(f"Generating experiments.")
    my_logger.info(f"Number of circuits: {len(circuits)}")
    my_logger.info(f"Combinations: {combinations}")
    my_logger.info(f"Backends: {all_backends}")
    my_logger.info(f"Backends distribution: {backends[0].__name__}")
    my_logger.info(f"Cut strategy: {cut_strategy.__name__}")
    my_logger.info(f"Shotwise policies: {','.join(p.__name__ for p in shotwise_policies)}")
    my_logger.info(f"Shots allocation: {','.join(p.__name__ for p in shots_allocations)}")
    my_logger.info(f"Shots: {shots}")

    for circ, obs, qubits, name, _, perf_exp_val in circuits:
        for comb in combinations:
            if comb == "vanilla":
                for provider_backend in all_backends:
                    exp = {"circuit_name": name, "circuit": circ, "observable": obs, "n_qubits":qubits, "exp_type": comb, "shots": shots, "perf_exp_val": perf_exp_val}
                    exp["backends"] = [provider_backend]
                    exp_n += 1
                    bname = generate_backends_name(all_backends, [provider_backend])
                    exp["filename"] = f"{exp_n:04d}_q{qubits}_{comb}_{name}_{bname}.json"
                    exps.append(exp)
            elif comb == "cc":
                for provider_backend in all_backends:
                    for shots_allocation in shots_allocations:
                        exp = {"circuit_name": name, "circuit": circ, "observable": obs, "n_qubits":qubits, "exp_type": comb, "shots": shots, "perf_exp_val": perf_exp_val}
                        exp["backends"] = [provider_backend]
                        exp["cut_strategy"] = cut_strategy.__name__#TODO check all the names
                        exp["shots_allocation"] = shots_allocation.__name__ 
                        exp_n += 1
                        bname = generate_backends_name(all_backends, [provider_backend])
                        exp["filename"] = f"{exp_n:04d}_q{qubits}_{comb}_{name}_{bname}.json"
                        exps.append(exp)
            elif comb == "sw":
                for provider_backend_couples in backends_distr:
                    for sw_policy in shotwise_policies:
                        exp = {"circuit_name": name, "circuit": circ, "observable": obs, "n_qubits":qubits, "exp_type": comb, "shots": shots, "perf_exp_val": perf_exp_val}
                        exp["shotwise_policy"] = sw_policy.__name__
                        exp["backends"] = provider_backend_couples
                        exp_n += 1
                        bname = generate_backends_name(all_backends, provider_backend_couples)
                        exp["filename"] = f"{exp_n:04d}_q{qubits}_{comb}_{name}_{bname}.json"
                        exps.append(exp)
            else:
                for provider_backend_couples in backends_distr:
                    for sw_policy in shotwise_policies:
                        for shots_allocation in shots_allocations:
                            exp = {"circuit_name": name, "circuit": circ, "observable": obs, "n_qubits":qubits, "exp_type": comb, "shots": shots, "perf_exp_val": perf_exp_val}
                            exp["cut_strategy"] = cut_strategy.__name__
                            exp["shotwise_policy"] = sw_policy.__name__
                            exp["shots_allocation"] = shots_allocation.__name__
                            exp["backends"] = provider_backend_couples
                            exp_n += 1
                            bname = generate_backends_name(all_backends, provider_backend_couples)
                            exp["filename"] = f"{exp_n:04d}_q{qubits}_{comb}_{name}_{bname}.json"
                            exps.append(exp)

    my_logger.info(f"Total number of experiments: {len(exps)}")
    return  exps

def resume_experiments(EXPS_DIR, experiments):
    todo_experiments = []
    for exp in experiments:
        exp_type = exp["exp_type"]
        EXP_DIR = os.path.join(EXPS_DIR, exp_type)
        try:
            with open(os.path.join(EXP_DIR, exp["filename"]), "r") as f:
                _ = json.load(f)
        except FileNotFoundError:
            todo_experiments.append(exp)
        except json.JSONDecodeError:
            todo_experiments.append(exp)
            os.remove(os.path.join(EXP_DIR, exp["filename"]))
    return todo_experiments

def launch_experiment(EXPS_DIR, exp, experiment_settings, tot_experiments):
    my_logger.info("*********Starting*******"+exp["filename"]+"*****Exps left:"+str(tot_experiments)+"***")
    exp_type = exp["exp_type"]
    EXP_DIR = os.path.join(EXPS_DIR, exp_type)
    failures = 0
    if exp_type == "vanilla":
        failures += spawn_experiment(EXP_DIR, exp, experiment_settings, "-va")
    elif exp_type == "cc":
        failures +=spawn_experiment(EXP_DIR, exp, experiment_settings, "-cc")
    elif exp_type == "sw":
        failures +=spawn_experiment(EXP_DIR, exp, experiment_settings,  "-sw")
    else:
        #exp["cut_strategy"] = exp["cut_strategy"].removeprefix("cutshot.src.")
        #exp["shots_allocation"] = exp["shots_allocation"].removeprefix("cutshot.src.")
        #exp["shotwise_policy"] = exp["shotwise_policy"].removeprefix("cutshot.src.")
        EXP_DIR = os.path.join("../",EXP_DIR)
        failures +=spawn_experiment(EXP_DIR, exp, experiment_settings,None)
    return failures

def create_temp_files(exp, exp_type, parallel_execution):
    qasm_circuit = exp["circuit"]
  
    temp_path = os.path.join(os.path.dirname(__file__), TEMP_FOLDER)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    temp_qasm_path = os.path.join(temp_path, TEMP_QASM_FILE)

    with open(temp_qasm_path, "w") as f:
        f.write(qasm_circuit)

    circuit_name = exp["circuit_name"]
    observable = exp["observable"]
    cut_strategy = exp["cut_strategy"] if "cut_strategy" in exp else None
    shots_allocation = exp["shots_allocation"] if "shots_allocation" in exp else None
    shotwise_policy = exp["shotwise_policy"] if "shotwise_policy" in exp else None
    backends = list(exp["backends"])
    
    lines=[
        '[SETTINGS]'+'\n',
        f'circuit = "temp/.temp.qasm"'+'\n',
        f'observables = "{observable}"'+'\n',
        f"shots = {exp['shots']}"+'\n',
        f"parallel_execution = {parallel_execution}"+'\n',
        f'cut_strategy_module = "{cut_strategy}"'+'\n',
        f'shots_allocation_module = "{shots_allocation}"'+'\n',
        f"backends = {backends}".replace("'", '"')+'\n',
        f'sw_policy_module = "{shotwise_policy}"'+'\n',
        f'circuit_name = "{circuit_name}"'+'\n',
        f'perf_exp_val = {exp["perf_exp_val"]}'+'\n'
        f'n_qubits = {exp["n_qubits"]}'+'\n'
    ]

    if exp_type == "cc":
        lines.pop(8)#remove sw_policy
    elif exp_type == "sw":
        lines.pop(6)#remove cut_strategy
        lines.pop(5)#remove shots_allocation
    elif exp_type == "vanilla":
        lines.pop(8)#remove sw_policy
        lines.pop(6)#remove cut_strategy
        lines.pop(5)#remove shots_allocation
    else:
        lines[1] = f'circuit = "../temp/.temp.qasm"'+'\n' #cut and shoot has a different path
        for i in range(len(lines)):
            if  "cutshot." in lines[i]:
                line = lines[i]
                line = line.replace("cutshot.", "")
                lines[i] = line
    ini_path = os.path.join(temp_path, TEMP_CONF_FILE)
    with open(ini_path, "w") as f:
        f.writelines(lines) 

def spawn_experiment(EXP_DIR, exp, experiment_settings, exp_flag):
    exp_type = exp["exp_type"]
    stop_on_error = experiment_settings["stop_on_error"]
    retry_on_error = experiment_settings["retry_on_error"]
    parallel_execution = experiment_settings["parallel_execution"]
    create_temp_files( exp, exp_type, parallel_execution)
    
    file_output = os.path.join(EXP_DIR, exp["filename"])
    if exp_flag:
        args = ["python", "run_experiment.py", "-o", file_output, exp_flag]
    else:
        args = ["python", "cutshot/main.py", "-o", file_output,"-v", "-c", "./temp/.temp.ini"]
    
    return_code = 1
    tries = 0
    failures = 0
    while return_code != 0:
        tries += 1
        proc = subprocess.run(
            args,
            #capture_output=True,
            text=True,
            stdout=sys.stdout, 
            stderr=subprocess.STDOUT
            )
        return_code = proc.returncode
        if return_code != 0:
            failures += 1
            my_logger.error(f"Experiment {exp['filename']} failed.")
            if stop_on_error:
                logging.shutdown()
                exit(-1)
            if retry_on_error and tries < max_tries:
                my_logger.info(f"Retrying experiment {exp['filename']}.")
            else:
                my_logger.info(f"Skipping experiment {exp['filename']}.")
                return failures
    return failures