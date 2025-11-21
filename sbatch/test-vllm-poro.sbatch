#!/bin/bash

#SBATCH --account=<LUMI project account>
#SBATCH --job-name=test-poro
#SBATCH -o %x-%j.out

#SBATCH --dependency=singleton                
#SBATCH --partition=standard-g                                                                                              
#SBATCH --time=00:30:00                
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


export VLLM_PORT=8101

export SINGULARITY_TMPDIR=../singularity 
export SINGULARITY_CACHEDIR=../singularity 

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

MODEL_DIR="$HF_HOME/models/$MODEL"
echo "!!!MODEL DIR: $MODEL_DIR"


RAY_PORT=6379
HEAD_NODE=$(hostname)

singularity exec $SIF sh -c "ray start --head --port=$RAY_PORT --disable-usage-stats && python -m vllm.entrypoints.openai.api_server --distributed-executor-backend=ray --trust-remote-code --gpu-memory-utilization=0.9  --max-num-seqs=128 --tensor-parallel-size=8 --pipeline-parallel-size=$SLURM_NNODES --model=$MODEL_DIR --max-model-len=28000 --enable-chunked-prefill --enable-prefix-caching --port $VLLM_PORT " &
### --rope-scaling '{\"rope_type\":\"dynamic\",\"factor\":16}'


WORKER_NNODES=$(( SLURM_NNODES - 1 ))
echo "Start the $WORKER_NNODES worker node(s)"
srun --ntasks=$WORKER_NNODES --nodes=$WORKER_NNODES --exclude=$HEAD_NODE singularity exec $SIF ray start --block --address=$HEAD_NODE:${RAY_PORT} &


## run once for each model
singularity exec $SIF huggingface-cli download $MODEL --local-dir $MODEL_DIR --local-dir-use-symlinks False
echo "!!!Done downloading"
ls "$MODEL_DIR"/model-*.safetensors | wc -l

# Wait until vLLM is running properly
sleep 20
while ! curl "http://localhost:${VLLM_PORT}" >/dev/null 2>&1
do
  sleep 10
done
 
##############TEST
FILTER_PIPELINE_BASE="../../filter-pipeline"

singularity exec $SIF python scripts/run_pipeline.py --corpus $FILTER_PIPELINE_BASE/data/work/jr/verses_cl.csv --id j45771,j25563 --pipeline pipelines/main_merged_translate_table.json --model $MODEL_DIR --base_url "http://localhost:${VLLM_PORT}/v1" --runs_base tests  --run_label main_merged_translate_table_fi --chunk_size 4 --max_retries 2 --max_tokens 4000 --temperature 0.0 --lang fi


singularity exec $SIF python scripts/run_pipeline.py --corpus $FILTER_PIPELINE_BASE/data/work/erab/verses_cl.csv --id era203000970003,era207801520003 --pipeline pipelines/main_merged_translate_table.json --model $MODEL_DIR --base_url "http://localhost:${VLLM_PORT}/v1" --runs_base tests  --run_label main_merged_translate_table_est --chunk_size 4 --max_retries 2 --max_tokens 4000 --temperature 0.0 --lang est
