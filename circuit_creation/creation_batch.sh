#!/bin/bash
min_qubits=14
max_qubits=14
step=2
ncircuits=4 #attualmente non funziona, vengono generati tutti i circuiti possibili
variations_max=10
max_var_qubits=10

declare -A try_conf

# Cicli nidificati per generare le combinazioni
for ((n = 1; n <= max_qubits; n++)); do
    for ((r = 1; r <= max_qubits; r++)); do
        for ((k = 1; k <= max_qubits; k++)); do
            nqubits=$((n*r + k*(r-1)))
            if ((nqubits < min_qubits || nqubits > max_qubits || (nqubits-min_qubits) % step != 0)); then
                continue
            fi
            if [[ -z "${try_conf[$nqubits]}" ]]; then
                try_conf[$nqubits]=""
            fi
            try_conf[$nqubits]+="($n,$r,$k) "
        done
    done
done

run_all() {
    echo -n "["
    for key in "${!try_conf[@]}"; do
        local values="${try_conf[$key]}"
        local n r k
        
        try=0
        for value in $values; do
            value="${value//[\(\)]/}"  # Rimuove parentesi
            IFS=',' read -r n r k <<< "$value"
            n=$n
            r=$r
            k=$k
            for ((seed = 0; seed <= 100; seed=seed+4)); do
                if [ $try -lt $ncircuits ]; then
                    python creation_single.py $n $r $k $seed $variations_max $max_var_qubits &
                    pid1=$!
                fi
                if [ $try -lt $ncircuits ]; then
                    python creation_single.py $n $r $k $((seed+1)) $variations_max $max_var_qubits &
                    pid2=$!
                fi
                if [ $try -lt $ncircuits ]; then
                    python creation_single.py $n $r $k $((seed+2)) $variations_max $max_var_qubits &
                    pid3=$!
                fi
                if [ $try -lt $ncircuits ]; then
                    python creation_single.py $n $r $k $((seed+3)) $variations_max $max_var_qubits &
                    pid4=$!
                fi

                sleep 5 &
                
                memp1=$(ps -eo pid,%mem | grep $pid1 | awk '{print $2}' | awk -F. '{ print ($1) }')
                if [ ! -z "$memp1" ]; then
                    if [ "$memp1" -gt 40 ]; then
                        kill $pid1 > /dev/null 2>&1
                    fi
                fi

                memp2=$(ps -eo pid,%mem | grep $pid2 | awk '{print $2}' | awk -F. '{ print ($1) }')
                if [ ! -z "$memp2" ]; then
                    if [ "$memp2" -gt 40 ]; then
                        kill $pid2 > /dev/null 2>&1
                    fi
                fi

                memp3=$(ps -eo pid,%mem | grep $pid3 | awk '{print $2}' | awk -F. '{ print ($1) }')
                if [ ! -z "$memp3" ]; then
                    if [ "$memp3" -gt 40 ]; then
                        kill $pid3 > /dev/null 2>&1
                    fi
                fi

                memp4=$(ps -eo pid,%mem | grep $pid4 | awk '{print $2}' | awk -F. '{ print ($1) }')
                if [ ! -z "$memp4" ]; then
                    if [ "$memp4" -gt 40 ]; then
                        kill $pid4 > /dev/null 2>&1
                    fi
                fi

                wait $pid1
                wait $pid2
                wait $pid3
                wait $pid4
            done
        done
        if [ $try -ge $ncircuits ]; then
            break
        fi
    done
}

run_all