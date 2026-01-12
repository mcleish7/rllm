uenv start --view=modules prgenv-gnu/25.6:v2
module load aws-ofi-nccl cuda nccl libfabric gcc # note no python here
source /capstor/scratch/cscs/smcleish/daint_291_129_uenv_venv_rllm_2/.venv/bin/activate
cd rllm_daint_291_129_uenv_venv_rllm_2/

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-reproduce-untouched \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=1440 \
    --pass_run_name=False \
    --custom_invocation='bash examples/deepscaler/11_jan.sh/reproduce_untouched.sh'

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-easy-first-run-with-shuffle \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=1440 \
    --pass_run_name=False \
    --custom_invocation='DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_easy" bash examples/deepscaler/7_jan/dataset_env_var_launch_with_shuffle.sh'

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-hard-first-run-with-shuffle \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=1440 \
    --pass_run_name=False \
    --custom_invocation='DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_hard" bash examples/deepscaler/7_jan/dataset_env_var_launch_with_shuffle.sh'
