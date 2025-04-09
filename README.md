# Cut&Shoot experiments
Tool to make experiments using Cut&Shoot and the settings with only circuit cutting, only shot-wise and with none of them.

## Installation
The tool can be used after installing the dependencies with pip:
```bash
git clone [this repository]
cd QExperiments4C-S
pip install -r requirements.txt
```

## Usage

Cut&Shoot can be used as a command line tool.

```bash

cd src
python main.py [CONFIGURATION_FILE]

```

where [CONFIGURATION_FILE] is the path to the configuration file (default = config.ini).

## External files

The tool needs a configuration file (e.g. conf.ini) in input which specifies the parameters of the pipeline. The configuration file is in .ini format and contains the following sections:

```ini

[SETTINGS]
git_push = boolean flag that indicates if push the results on GitHub at the end, values: True or False
stop_on_error = boolean flag that indicates if stop the execution when there is an error in the experiments, values: True or False
retry_on_error = boolean flag the indicates if push retry an experiment that has failed, values: True or False
resume = boolean flag that indicates if resume previous not completed experiments. Warning: when False overwrite the experiments file for the same id: True or False
experiment_id = unique identifier of the experiment, it is also the name of the folder containing the results of the experiment, e.g test
circuits = file with the configurations to generate the QAOA circuits, e.g. "./mini_conf.json"
shots = number of overall shots of a single experiment, e.g. 8000
combinations = list of combinations of the experiments, values: ["vanilla","cc","sw","cc_sw"]
start_time = format hh:mm: starting time window when one experiment can start, allowed only in combination with stop_time, e.g. 08:00. When not specified there is no time window restriction.
stop_time = format hh:mm: stopping time window when one experiment can start, allowed only in combination with start_time, e.g. 20:00. When not specified there is no time window restriction.
parallel = boolean flag that indicates if each execution on a backend is on a different process, values: True or False

[BACKENDS]
backends = list of two-element list [provider, backend] used in the experiments, e.g. [["ibm_aer", "aer.fake_brisbane"], ["ibm_aer", "aer.fake_kyoto"]]. Currenlty only FakeBackendV2 are supported (https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake_provider#fake-v2-backends). For such backends, use "ibm_aer" ad provider and prefix their name with "aer", to specify to run them on the AerSimulator. Currenlty only AerSimulator is supported. To use a perfect backend use "ibm_aer" as provider and "aer.perfect" as backend.

[MODULES]
backend_distribution = name of the python script containing the policy to combine the backends for the shot-wise, e.g. "backends_powerset"
shotwise_policies = name of the python script containing the shot-wise policies, e.g. ["cutshot.policies.sw_policies"]
cut_strategy = name of the python script containing the cutting strategy, e.g."cutshot.pennylane_tool"
shots_allocation = name of the python script containing the shots allocation strategy, e.g. "cutshot.policies.qubit_proportional"
```

The backend distribution policy is used to select the backends to use in the only shot-wise and in the Cut & Shot experiments.
For instance, using the powerset policy means using all the combinations of backends (with at least two) in the experiments.
It must be a Python script (e.g cutting_tool.py), implementing the following interface:
```
- backend_distribution(backends) -> (backend_distribution)
    backends: list of tuples (provider, backend) representing all the considered backends
    backend_distribution: list of tuples (provider, backend) where each tuple will be used in the experiments
```
The cutting strategy must be a Python script (e.g cutshot/pennylane_tool.py), implementing the following interface:
```
- cut(circuit, observable_string) -> output, cut_data, cut_info
    circuit: QASM circuit to cut
    observable_string: observable to measure on the circuit
    output: list of tuples (fragment, observable) where fragment is a QASM circuit without basis changes and observable is a list of string representing the observables
    cut_data: dictionary containing data needed by the sew function
    cut_info: (optional) dictionary containing information about the cut recorded by the experiments, must be JSON serializable
- sew(qasm_obs_expvals, sew_data)  -> results
    qasm_obs_expvals: dictionary (fragment, observable) -> expected value, where (fragment, observable) are the tuples returned by the cut function and expected value is the expected value of the fragment execution
    sew_data: dictionary containing data needed by the sew function
    results: results of the sew function
```

The shots allocation strategy must be a Python script (e.g cutshot/policies/qubit_proportional.py), implementing the following interface:
```
- allocate_shots: (vcs, shots_assignment) -> vc_shots
    vcs: list of qukit.VirtualCircuit circuit where each circuit has a metadata field containing the number of qubits
    shots_allocation: parameters for the allocation
    vcs_shots: list of tuples (fragment, shots), where fragment is a qukit.VirtualCircuit circuit and shots is the number of shots to assign to the fragment
```

The shot-wise policies must be a Python script (e.g cutshot/policies/sw_policies.py), implementing the following interface:
```
- split(backends, shots) -> (dispatch, coefficients)
    backends: list of tuples (provider, backend)
    shots: number of shots to split
    dispatch: dictionary of dictionaries of integers, where dispatch[provider][backend] is the number of shots to send to the backend
    coefficients: dictionary of dictionaries of floats, where coefficients[provider][backend] is the weight of the backend in the policy
- merge(counts) -> (probs, coefficients)
    counts: dictionary of dictionaries of lists, where counts[provider][backend] is a list of tuples (circuit_id, observable, counts)
    probs: dictionary of dictionaries of floats, where probs[(circuit_id,observable)][state] is the probability of measuring the state in the circuit
    coefficients: dictionary of dictionaries of floats, where coefficients[provider][backend] is the weight of the backend in the policy
```
## OUTPUT
TODO
