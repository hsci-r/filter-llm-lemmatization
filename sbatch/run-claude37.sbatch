#!/bin/bash
#SBATCH --account=<LUMI project account>
#SBATCH --job-name=claude37-full
#SBATCH -o %x-%j.out
#SBATCH --partition=standard
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

###################################
# 1. Python environment setup     #
###################################

module load cray-python
python --version

VENV_DIR=../../venvs/claude37
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null

# Ensure dependencies
python - <<'PY'
import sys, subprocess
req = ["anthropic", "httpx>=0.27", "openai>=1.40", "pandas"]
for pkg in req:
    try:
        __import__(pkg.split("==")[0].split(">=")[0])
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
PY

python -c "import anthropic, httpx, openai; print('Deps OK')"

###################################
# 2. Anthropic model config       #
###################################

MODEL="claude-3-7-sonnet-20250219"
BASE_URL="anthropic"
export ANTHROPIC_API_KEY="YOUR KEY HERE"

###################################
# 3. Run parameters               #
###################################

# Each pipeline = full job pass over all corpora
CHUNK_SIZE=25
MAX_RETRIES=1
MAX_TOKENS=8000
TEMP=0.0
MAX_CONNECTIONS=1
RUNS_BASE="runs"
RUN_LABEL_PREFIX="claude37"

# File with all target IDs
IDS_LIST="$(cat ../annotations/ids.txt)"

###################################
# 4. Corpora and language mapping #
###################################

FILTER_PIPELINE_BASE="../../filter-pipeline"

declare -A CORPUS LANG
CORPUS[jr]="$FILTER_PIPELINE_BASE/data/work/jr/verses_cl.csv"
LANG[jr]="fi"

CORPUS[skvr]="$FILTER_PIPELINE_BASE//data/work/skvr/verses_cl.csv"
LANG[skvr]="fi"

CORPUS[erab]="$FILTER_PIPELINE_BASE/data/work/erab/verses_cl.csv"
LANG[erab]="est"

###################################
# 5. Pipelines to run             #
###################################

#    bare_merged_translate_table
#    main_only_table
#    main_merged_translate_table
#    bare_seq_translate_then_table
#    main_seq_translate_then_table

#    bare_only_table
#    bare_seq_translate_fix_table_fix


#    bare_seq_translate_fix_table
#    main_seq_translate_fix_table
#    main_seq_translate_fix_table_fix

PIPELINES=(
    bare_merged_translate_table
    bare_only_table
    main_only_table
    main_merged_translate_table
)

###################################
# 6. Execution function           #
###################################

run_one() {
  local pipeline="$1" tag="$2"
  local corpus="${CORPUS[$tag]}"
  local lang="${LANG[$tag]}"
  local label="${RUN_LABEL_PREFIX}/${pipeline}/${tag}"

  echo "[$(date '+%F %T')] Running ${label}"
  python scripts/run_pipeline.py \
    --corpus "$corpus" \
    --id "$IDS_LIST" \
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

###################################
# 7. Main run loop                #
###################################

echo "Using MODEL=${MODEL}"
echo "BASE_URL=${BASE_URL}"
echo "RUNS_BASE=${RUNS_BASE}"
echo "RUN_LABEL_PREFIX=${RUN_LABEL_PREFIX}"
echo "Pipelines: ${PIPELINES[*]}"
echo

for p in "${PIPELINES[@]}"; do
  for tag in "${!CORPUS[@]}"; do
    run_one "$p" "$tag"
  done
done

echo "All done."
