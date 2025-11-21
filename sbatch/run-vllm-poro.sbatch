#!/bin/bash

#SBATCH --account=<LUMI project account>
#SBATCH --job-name=run-poro
#SBATCH -o %x-%j.out


#SBATCH --partition=standard-g                                                                                              
#SBATCH --time=48:00:00                
#SBATCH --nodes=5 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8 
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G

SIF=docker://rocm/vllm:rocm6.3.1_instinct_vllm0.8.3_20250410

MODEL="LumiOpen/Llama-Poro-2-70B-Instruct"

# Alternative models
# MODEL=unsloth/DeepSeek-R1-BF16
# MODEL="meta-llama/Llama-3.3-70B-Instruct"
# MODEL="mistralai/Mixtral-8x22B-Instruct-v0.1"
# MODEL="databricks/dbrx-instruct"


export VLLM_PORT=8100

export SINGULARITY_TMPDIR=../singularity 
export SINGULARITY_CACHEDIR=../singularity 

export SINGULARITYENV_VLLM_LOGGING_LEVEL=ERROR

export HF_HOME=../../hf

mkdir -p "$HF_HOME/models"

export HF_TOKEN="SOME MODELS REQUIRE HUGGING FACE TOKEN"


export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export SINGULARITYENV_HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN"

export SINGULARITYENV_HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN"
export SINGULARITYENV_HF_HOME="$HF_HOME"

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/rocm/lib/:/opt/cray/libfabric/1.15.2.0/lib64
export SINGULARITY_CONTAINLIBS=/usr/lib64/libcxi.so.1,/usr/lib64/libjson-c.so.3,/opt/rocm/lib/librocm_smi64.so.6
export SINGULARITY_BIND=/opt/cray,/pfs,/users,/scratch,/projappl,/flash,/project,app:/app
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

#MODEL_DIR=$(echo $MODEL | tr '/' '_')
#MODEL_DIR="$HF_HOME/models/$MODEL_DIR"
MODEL_DIR="$HF_HOME/models/$MODEL"
echo "!!!MODEL DIR: $MODEL_DIR"


RAY_PORT=6379
HEAD_NODE=$(hostname)



singularity exec $SIF sh -c "ray start --head --port=$RAY_PORT --disable-usage-stats && python -m vllm.entrypoints.openai.api_server --distributed-executor-backend=ray --trust-remote-code --gpu-memory-utilization=0.9  --max-num-seqs=128 --tensor-parallel-size=8 --pipeline-parallel-size=$SLURM_NNODES --model=$MODEL_DIR --max-model-len=28000 --enable-chunked-prefill --enable-prefix-caching --port $VLLM_PORT " &

WORKER_NNODES=$(( SLURM_NNODES - 1 ))
echo "Start the $WORKER_NNODES worker node(s)"
srun --ntasks=$WORKER_NNODES --nodes=$WORKER_NNODES --exclude=$HEAD_NODE singularity exec $SIF ray start --block --address=$HEAD_NODE:${RAY_PORT} &


### run once for each model
singularity exec $SIF huggingface-cli download $MODEL --local-dir $MODEL_DIR --local-dir-use-symlinks False
echo "!!!Done downloading"
ls "$MODEL_DIR"/model-*.safetensors | wc -l

# Wait until vLLM is running properly
sleep 20
while ! curl "http://localhost:${VLLM_PORT}" >/dev/null 2>&1
do
  sleep 10
done

#############FULL-SCALE

# ---- Common run params ----
CHUNK=25
RETRIES=2
MAX_TOKENS=4000
TEMP=0.0
RUNS_BASE="runs"

# Load the same ID list for all corpora (change if you ever want per-corpus lists)
IDS_LIST="$(cat ../annotations/ids.txt)"

FILTER_PIPELINE_BASE="../../filter-pipeline"

# ---- Define corpora and their languages ----
# tags: jr (Finnish JR), skvr (Finnish SKVR), erab (Estonian ERAB)
declare -A CORPUS LANG
CORPUS[jr]="../../filter-pipeline/data/work/jr/verses_cl.csv"
LANG[jr]="fi"

CORPUS[skvr]="../../filter-pipeline/data/work/skvr/verses_cl.csv"
LANG[skvr]="fi"

CORPUS[erab]="../../filter-pipeline/data/work/erab/verses_cl.csv"
LANG[erab]="est"

# ---- Choose which pipelines to run (activate/deactivate by editing this list) ----
# Available: main_only_table, main_merged_translate_table, main_seq_translate_then_table,
#            main_seq_translate_fix_table, main_seq_translate_fix_table_fix
#     main_seq_translate_fix_table main_seq_translate_fix_table_fix

# bare_merged_translate_table
# bare_only_table
# bare_seq_translate_fix_table
# bare_seq_translate_fix_table_fix
# bare_seq_translate_then_table
# main_merged_translate_table
# main_only_table
# main_seq_translate_fix_table
# main_seq_translate_fix_table_fix
# main_seq_translate_then_table     

PIPELINES=(
prompt8
)

run_one() {
    local pipeline="$1" tag="$2"
    local corpus="${CORPUS[$tag]}"
    local lang="${LANG[$tag]}"
    local label="poro/${pipeline}/${tag}"
    echo "[$(date '+%F %T')] Running ${label}"
    singularity exec "$SIF" python scripts/run_pipeline.py \
		--corpus "$corpus" \
		--id "$IDS_LIST" \
		--pipeline "pipelines/${pipeline}.json" \
		--model "$MODEL_DIR" \
		--base_url "http://localhost:${VLLM_PORT}/v1" \
		--runs_base "$RUNS_BASE" \
		--run_label "$label" \
		--chunk_size "$CHUNK" \
		--max_retries "$RETRIES" \
		--max_tokens "$MAX_TOKENS" \
		--temperature "$TEMP" \
		--lang "$lang"
}

# ---- Execute: pipelines × corpora ----
for p in "${PIPELINES[@]}"; do
    for tag in "${!CORPUS[@]}"; do
	run_one "$p" "$tag"
    done
done


