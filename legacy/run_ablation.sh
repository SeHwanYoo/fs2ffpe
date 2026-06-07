#!/bin/bash
# ============================================================
# DeepThaw v2 Ablation Experiments
# ============================================================
# 사용법:
#   export DATA_PATH=/path/to/FS2FFPE
#   bash run_ablation.sh e1
#   bash run_ablation.sh e2
#   bash run_ablation.sh e4
#   bash run_ablation.sh all
# ============================================================

SCRIPT="train_fs2ffpe_v2.py"
BS=4
EPOCHS=200
LOGDIR="logs"

# ⚠️ DATA_PATH 설정 필수!
DATA_PATH="${DATA_PATH:-/home/sehwan001/datasets/FS2FFPE}"

if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: DATA_PATH=$DATA_PATH 없음!"
    echo "  export DATA_PATH=/your/path/to/FS2FFPE"
    exit 1
fi

mkdir -p ${LOGDIR}

run() {
    local name=$1; shift
    local logfile="${LOGDIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== ${name} ==="
    echo "  Args: $@"
    python ${SCRIPT} "$@" --data-path "${DATA_PATH}" 2>&1 | tee ${logfile}
    echo "  Exit: $?"
    echo ""
}

# E1: RAG Ablation
run_e1() {
    echo "============================================"
    echo "E1: 5-way RAG Ablation"
    echo "============================================"
    run e1_full       --preset conch-full --batch-size ${BS} --epochs ${EPOCHS} --label v2-e1-full
    run e1_no_rag     --preset conch-sc   --batch-size ${BS} --epochs ${EPOCHS} --label v2-e1-no-rag
    run e1_random_rag --preset conch-full --batch-size ${BS} --epochs ${EPOCHS} --rag-random --label v2-e1-random-rag
    run e1_baseline   --preset baseline   --batch-size ${BS} --epochs ${EPOCHS} --label v2-e1-baseline
    echo "E1 Complete."
}

# E2: Data Scaling
run_e2() {
    echo "============================================"
    echo "E2: Data Scaling Curve"
    echo "============================================"
    for pct in 10 30 100; do
        echo "--- ${pct}% data ---"
        run e2_full_${pct}pct     --preset conch-full --batch-size ${BS} --epochs ${EPOCHS} --data-percent ${pct} --label v2-e2-full-${pct}pct
        run e2_no_rag_${pct}pct   --preset conch-sc   --batch-size ${BS} --epochs ${EPOCHS} --data-percent ${pct} --label v2-e2-norag-${pct}pct
        run e2_baseline_${pct}pct --preset baseline    --batch-size ${BS} --epochs ${EPOCHS} --data-percent ${pct} --label v2-e2-baseline-${pct}pct
    done
    echo "E2 Complete."
}

# E4: Wrong-RAG
run_e4() {
    echo "============================================"
    echo "E4: Wrong-RAG Stress Test"
    echo "============================================"
    run e4_correct --preset conch-full --batch-size ${BS} --epochs ${EPOCHS} --label v2-e4-correct
    run e4_wrong   --preset conch-full --batch-size ${BS} --epochs ${EPOCHS} --rag-wrong --label v2-e4-wrong
    echo "E4 Complete."
}

case "${1:-all}" in
    e1)  run_e1 ;;
    e2)  run_e2 ;;
    e4)  run_e4 ;;
    all) run_e1; run_e2; run_e4 ;;
    *)   echo "Usage: bash run_ablation.sh {e1|e2|e4|all}"; exit 1 ;;
esac
echo "Done!"
