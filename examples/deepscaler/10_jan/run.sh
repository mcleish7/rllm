uenv start --view=modules prgenv-gnu/25.6:v2
module load aws-ofi-nccl cuda nccl libfabric gcc # note no python here
source /capstor/scratch/cscs/smcleish/daint_291_129_uenv_venv_rllm_2/.venv/bin/activate
cd rllm_daint_291_129_uenv_venv_rllm_2/

uv pip install -e .[verl]

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-reproduce-first-run \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=1440 \
    --pass_run_name=False \
    --custom_invocation='DS_NAME="deepscaler_math" bash examples/deepscaler/7_jan/dataset_env_var_launch.sh'

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-reproduce-first-run-with-shuffle \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=1440 \
    --pass_run_name=False \
    --custom_invocation='DS_NAME="deepscaler_math" bash examples/deepscaler/7_jan/dataset_env_var_launch_with_shuffle.sh'
