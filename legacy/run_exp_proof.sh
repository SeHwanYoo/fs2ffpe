#!/bin/bash
# ================================================================
# E1-E4 RAG Ablation 학습 실행 스크립트
# ================================================================
#
# E1: 5-way ablation (학습 후 proof_experiments.py rag_ablation으로 비교)
# E2: Data scaling (10%, 30%, 100%)
# E3: OOD split (site 기반)
# E4: Wrong-RAG (farthest neighbor)
#
# 사용법:
#   bash run_ablation.sh e1          # 5-way ablation 전부
#   bash run_ablation.sh e2          # Data scaling
#   bash run_ablation.sh e1_parallel # E1 병렬 실행
#   bash run_ablation.sh all         # 전부
#
# E1 완료 후 분석:
#   python proof_experiments.py rag_ablation \
#     --csv_file labels.csv \
#     --result_dirs \
#       full=outdir/v2-conch-full/eval/... \
#       no_rag=outdir/v2-conch/eval/... \
#       random_rag=outdir/v2-conch-random-rag/eval/... \
#       weak_retriever=outdir/v2-conch-weak-retriever/eval/... \
#       no_fm_no_rag=outdir/v2-baseline/eval/...
# ================================================================

SCRIPT="train_fs2ffpe_v2.py"
BS=4
EP=200
LOGDIR="logs/ablation"
mkdir -p ${LOGDIR}

run() {
    local name=$1; shift
    echo "=== ${name} ==="
    echo "  Args: $@"
    python -u ${SCRIPT} "$@" 2>&1 | tee ${LOGDIR}/${name}.log
}

run_bg() {
    local name=$1; shift
    echo "=== ${name} (background) ==="
    nohup python -u ${SCRIPT} "$@" > ${LOGDIR}/${name}.log 2>&1 &
    echo "  PID: $!"
}

# ================================================================
# E1: 5-way Ablation
# ================================================================
e1() {
    echo "============================================"
    echo "E1: 5-way RAG Ablation"
    echo "============================================"
    echo ""
    echo "1) Full: CONCH + SC + RAG"
    echo "2) No-RAG: CONCH + SC (no RAG)"
    echo "3) Random-RAG: CONCH + SC + random exemplars"
    echo "4) Weak-Retriever: ResNet retriever instead of CONCH"
    echo "5) No-FM/No-RAG: baseline UVCGAN2"
    echo ""

    # 1) Full (CONCH + SC + RAG)
    run "e1_full" \
        --preset conch-full \
        --batch-size ${BS} --epochs ${EP} \
        --label v2-e1-full

    # 2) No-RAG (CONCH + SC)
    run "e1_no_rag" \
        --preset conch-sc \
        --batch-size ${BS} --epochs ${EP} \
        --label v2-e1-no-rag

    # 3) Random-RAG (CONCH + SC + RAG with shuffled cache)
    # → train script에서 --rag-random 플래그 추가 필요
    # → 없으면 수동으로 rag_cache를 셔플한 버전 만들어서 사용
    run "e1_random_rag" \
        --preset conch-full \
        --batch-size ${BS} --epochs ${EP} \
        --rag-random \
        --label v2-e1-random-rag

    # 4) No-FM/No-RAG (baseline)
    run "e1_baseline" \
        --preset baseline \
        --batch-size ${BS} --epochs ${EP} \
        --label v2-e1-baseline

    echo ""
    echo "E1 Complete. 분석:"
    echo "  python proof_experiments.py rag_ablation --csv_file labels.csv \\"
    echo "    --result_dirs full=... no_rag=... random_rag=... baseline=..."
}

e1_parallel() {
    echo "E1: 5-way Ablation (parallel)"
    run_bg "e1_full"       --preset conch-full --batch-size ${BS} --epochs ${EP} --label v2-e1-full
    run_bg "e1_no_rag"     --preset conch-sc   --batch-size ${BS} --epochs ${EP} --label v2-e1-no-rag
    run_bg "e1_baseline"   --preset baseline    --batch-size ${BS} --epochs ${EP} --label v2-e1-baseline
    echo "Monitor: tail -f ${LOGDIR}/e1_*.log"
    wait
}

# ================================================================
# E2: Data Scaling (10%, 30%, 100%)
# ================================================================
e2() {
    echo "============================================"
    echo "E2: Data Scaling Curve"
    echo "============================================"

    for pct in 10 30 100; do
        echo ""
        echo "--- ${pct}% data ---"

        # Full
        run "e2_full_${pct}pct" \
            --preset conch-full \
            --batch-size ${BS} --epochs ${EP} \
            --data-percent ${pct} \
            --label v2-e2-full-${pct}pct

        # No-RAG
        run "e2_no_rag_${pct}pct" \
            --preset conch-sc \
            --batch-size ${BS} --epochs ${EP} \
            --data-percent ${pct} \
            --label v2-e2-norag-${pct}pct

        # Baseline
        run "e2_baseline_${pct}pct" \
            --preset baseline \
            --batch-size ${BS} --epochs ${EP} \
            --data-percent ${pct} \
            --label v2-e2-baseline-${pct}pct
    done
}

# ================================================================
# E3: OOD Split (site-based)
# ================================================================
e3() {
    echo "============================================"
    echo "E3: OOD Evaluation (cross-site)"
    echo "============================================"
    echo "  → CSV에 OOD split column 추가 필요"
    echo "  → 또는 test set을 specific site로 제한"

    # Full
    run "e3_full_ood" \
        --preset conch-full \
        --batch-size ${BS} --epochs ${EP} \
        --ood-eval \
        --label v2-e3-full-ood

    # Baseline
    run "e3_baseline_ood" \
        --preset baseline \
        --batch-size ${BS} --epochs ${EP} \
        --ood-eval \
        --label v2-e3-baseline-ood
}

# ================================================================
# E4: Wrong-RAG
# ================================================================
e4() {
    echo "============================================"
    echo "E4: Wrong-RAG Stress Test"
    echo "============================================"
    echo "  → Farthest neighbor retrieval"

    # Correct RAG
    run "e4_correct" \
        --preset conch-full \
        --batch-size ${BS} --epochs ${EP} \
        --label v2-e4-correct

    # Wrong RAG (farthest neighbor)
    run "e4_wrong" \
        --preset conch-full \
        --batch-size ${BS} --epochs ${EP} \
        --rag-wrong \
        --label v2-e4-wrong

    echo ""
    echo "E4 분석:"
    echo "  python proof_experiments.py wrong_rag --csv_file labels.csv \\"
    echo "    --correct_dir=... --wrong_dir=..."
}

# ================================================================
# Dispatch
# ================================================================
case "$1" in
    e1)          e1 ;;
    e1_parallel) e1_parallel ;;
    e2)          e2 ;;
    e3)          e3 ;;
    e4)          e4 ;;
    all)
        e1
        e2
        e4
        ;;
    *)
        echo "Usage: $0 {e1|e1_parallel|e2|e3|e4|all}"
        echo ""
        echo "  e1: 5-way ablation (Full/No-RAG/Random-RAG/Baseline)"
        echo "  e2: Data scaling (10%/30%/100%)"
        echo "  e3: OOD evaluation"
        echo "  e4: Wrong-RAG stress test"
        echo "  all: e1 + e2 + e4"
        ;;
esac

echo ""
echo "Done!"