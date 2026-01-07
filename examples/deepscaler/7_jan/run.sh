uenv start --view=modules prgenv-gnu/25.6:v2
module load aws-ofi-nccl cuda nccl libfabric gcc # note no python here
source /capstor/scratch/cscs/smcleish/daint_291_129_uenv_venv_rllm/.venv/bin/activate
cd rllm_daint_291_129_uenv_venv_rllm/

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-easy-test-3 \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=30 \
    --pass_run_name=False \
    --custom_invocation='DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_easy" bash examples/deepscaler/7_jan/dataset_env_var_launch.sh' --partition=debug

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-easy-first-run \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=1440 \
    --pass_run_name=False \
    --custom_invocation='DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_easy" bash examples/deepscaler/7_jan/dataset_env_var_launch.sh'

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-hard-first-run \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=1440 \
    --pass_run_name=False \
    --custom_invocation='DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_hard" bash examples/deepscaler/7_jan/dataset_env_var_launch.sh'

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-random-first-run \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=1440 \
    --pass_run_name=False \
    --custom_invocation='DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_random" bash examples/deepscaler/7_jan/dataset_env_var_launch.sh'

