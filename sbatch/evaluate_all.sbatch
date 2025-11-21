#!/bin/bash

#SBATCH --account=<LUMI project account>
#SBATCH --job-name=evaluate-all
#SBATCH -o %x-%j.out
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard

# Base folders
BASE="../"
RUNS_DIR="$BASE/runs"
EVAL_DIR="$BASE/evaluations"


ANNOT="$BASE/annotations/manual_annotations.csv"
SCRIPT="$BASE/evaluation.py"

# You can override SPACY_MODEL here if you want
SPACY_MODEL="en_core_web_lg"

module load cray-python

## Loop over models and pipelines
for MODEL_DIR in "$RUNS_DIR"/*; do
  [ -d "$MODEL_DIR" ] || continue  # skip non-dirs


MODEL_NAME=$(basename "$MODEL_DIR")
  echo
  echo "=== MODEL: $MODEL_NAME ==="

  
  
  for PIPE_DIR in "$MODEL_DIR"/*; do
    [ -d "$PIPE_DIR" ] || continue
    PIPE_NAME=$(basename "$PIPE_DIR")

    OUTDIR="$EVAL_DIR/$MODEL_NAME/$PIPE_NAME"
    mkdir -p "$OUTDIR"

    echo "→ Evaluating $MODEL_NAME / $PIPE_NAME"
    python "$SCRIPT" \
      -a "$ANNOT" \
      -o "$PIPE_DIR" \
      --outdir "$OUTDIR" \
      --spacy-model "$SPACY_MODEL" \
      --precision 3 \
      -r "eval.csv"
  done
done

echo "=== DONE: All runs evaluated ==="

