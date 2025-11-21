#!/bin/bash
#SBATCH --account=<LUMI project account>
#SBATCH --job-name=claude37-pipeline
#SBATCH -o %x-%j.out

#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

########################
# Python environment   #
########################

module load cray-python
python --version


VENV_DIR=../../venvs/claude37
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null

# Install required libs (no-op if already present)
python - <<'PY'
import sys, subprocess
req = ["anthropic", "httpx>=0.27", "openai>=1.40", "pandas"]  
for pkg in req:
    try:
        __import__(pkg.split("==")[0].split(">=")[0])
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
PY

# Sanity check
python -c "import anthropic, httpx, openai; print('Deps OK')"

############################
# Anthropic model settings #
############################
MODEL="claude-3-7-sonnet-20250219"
BASE_URL="anthropic"          # triggers the Anthropic path in runner/llm.py

export ANTHROPIC_API_KEY="YOUR KEY HERE"

############################
# Multi-corpus run section #
############################

FILTER_PIPELINE_BASE="../../filter-pipeline"

PIPELINES=(
  main_merged_translate_table_claude
)

LANGS=(fi est)

declare -A CORPUS_fi IDS_fi
CORPUS_fi[jr]="$FILTER_PIPELINE_BASE/data/work/jr/verses_cl.csv"
IDS_fi[jr]="j45771,j25563"
CORPUS_fi[skvr]="$FILTER_PIPELINE_BASE/data/work/skvr/verses_cl.csv"
IDS_fi[skvr]="skvr04117570"

declare -A CORPUS_est IDS_est
CORPUS_est[erab]="$FILTER_PIPELINE_BASE/data/work/erab/verses_cl.csv"
IDS_est[erab]="era203000970003,era207801520003"

CHUNK_SIZE=4
MAX_RETRIES=2
TEMP=0.0
MAX_TOKENS=8000
RUNS_BASE="tests"
RUN_LABEL_PREFIX="claude37"
MAX_CONNECTIONS=2


echo "Using MODEL=${MODEL}"
echo "BASE_URL=${BASE_URL}"
echo "RUNS_BASE=${RUNS_BASE}"
echo "RUN_LABEL_PREFIX=${RUN_LABEL_PREFIX}"
echo "Pipelines: ${PIPELINES[*]}"
echo

run_one() {
  local pipeline="$1" lang="$2" tag="$3" corpus="$4" ids="$5"
  local label="${RUN_LABEL_PREFIX}/${pipeline}/${tag}"
  echo "[$(date '+%F %T')] Running ${label}"
  python scripts/run_pipeline.py \
    --corpus "$corpus" \
    --id "$ids" \
    --pipeline "pipelines/${pipeline}.json" \
    --model "$MODEL" \
    --base_url "$BASE_URL" \
    --api_key "$ANTHROPIC_API_KEY" \
    --runs_base "$RUNS_BASE" \
    --run_label "$label" \
    --chunk_size "$CHUNK_SIZE" \
    --max_retries "$MAX_RETRIES" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    --max_connections "$MAX_CONNECTIONS" \
    --lang "$lang"
}

for p in "${PIPELINES[@]}"; do
  for lang in "${LANGS[@]}"; do
    eval "tags=(\"\${!CORPUS_${lang}[@]}\")"
    for tag in "${tags[@]}"; do
      eval "corpus=\${CORPUS_${lang}[$tag]}"
      eval "ids=\${IDS_${lang}[$tag]}"
      run_one "$p" "$lang" "$tag" "$corpus" "$ids"
    done
  done
done

echo "All done."
