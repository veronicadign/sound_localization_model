#!/usr/bin/env bash
# Run all LFP reconstructions needed for the slides.
# Execute from the repo root:  bash LFP_reconstruction/run_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV=sl_env
RUN="conda run -n $CONDA_ENV mpiexec -n 4 python LFP_reconstruction/main_reconstruct.py"

BASELINE="RESULTS/baseline_simulation.pic"
CLICK="RESULTS/click_train_70dB&Zilany&baseline.pic"
N=3600

echo "========================================================"
echo " MSO LFP Reconstruction — all runs"
echo "========================================================"

echo -e "\n--- [1/6] baseline | angle=90 | L | binaural ---"
$RUN --pic-file "$BASELINE" --angle 90 --side L --n-cells $N

echo -e "\n--- [2/6] baseline | angle=90 | R | binaural ---"
$RUN --pic-file "$BASELINE" --angle 90 --side R --n-cells $N

echo -e "\n--- [3/6] baseline | angle=90 | R | binaural ---"
$RUN --pic-file "$BASELINE" --angle 90 --side R --n-cells $N

echo -e "\n--- [4/6] click train | angle=90 | L | binaural ---"
$RUN --pic-file "$CLICK"    --angle 90 --side L --n-cells $N

echo -e "\n--- [5/6] click train | angle=90 | R | binaural ---"
$RUN --pic-file "$CLICK"    --angle 90 --side R --n-cells $N

echo -e "\n--- [6/6] baseline | angle=90 | L | monaural ---"
$RUN --pic-file "$BASELINE" --angle 90 --side L --n-cells $N --monaural

echo -e "\n--- [7/7] baseline | angle=90 | R | monaural ---"
$RUN --pic-file "$BASELINE" --angle 90 --side R --n-cells $N --monaural

echo -e "\n========================================================"
echo " All runs complete."
echo "========================================================"
