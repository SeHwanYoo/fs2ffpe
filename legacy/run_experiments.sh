#!/bin/bash
# bash run_experiments.sh baseline
# bash run_experiments.sh parallel core
# bash run_experiments.sh parallel all

SCRIPT="train_fs2ffpe_v2.py"
BS=4
EPOCHS=200
LOGDIR="logs"
DATA_PATH="${DATA_PATH:-/home/sehwan001/datasets/FS2FFPE}"

mkdir -p ${LOGDIR}

run_one() {
    local preset=$1
    local logfile="${LOGDIR}/${preset}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date +%H:%M:%S)] Starting: ${preset}"
    python ${SCRIPT} --preset ${preset} --batch-size ${BS} --epochs ${EPOCHS} --data-path "${DATA_PATH}" 2>&1 | tee ${logfile}
    echo "[$(date +%H:%M:%S)] Done: ${preset} (exit: $?)"
}

run_bg() {
    local preset=$1
    local logfile="${LOGDIR}/${preset}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date +%H:%M:%S)] Starting (bg): ${preset}"
    nohup python ${SCRIPT} --preset ${preset} --batch-size ${BS} --epochs ${EPOCHS} --data-path "${DATA_PATH}" > ${logfile} 2>&1 &
    echo "  PID: $!"
}

get_presets() {
    case "$1" in
        core)     echo "baseline conch conch-full" ;;
        ablation) echo "conch sc-only conch-sc" ;;
        all)      echo "baseline conch sc-only rag-only conch-sc conch-rag conch-full" ;;
        *)        echo "$@" ;;
    esac
}

if [ "$1" = "parallel" ]; then
    shift; PRESETS=$(get_presets "$@")
    for p in ${PRESETS}; do run_bg ${p}; done
    echo "All launched. tail -f ${LOGDIR}/*.log"
    wait
else
    PRESETS=$(get_presets "$@")
    for p in ${PRESETS}; do run_one ${p}; echo ""; done
fi
echo "All done!"
