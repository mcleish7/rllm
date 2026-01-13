python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=720 \
    --pass_run_name=False \
    --custom_invocation='MODEL_PATH="/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step400-hf" DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_hard" bash examples/deepscaler/13_jan/dataset_env_var_launch_with_shuffle_16k.sh'

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-460-chkpt \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=720 \
    --pass_run_name=False \
    --custom_invocation='MODEL_PATH="/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step460-hf" DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_hard" bash examples/deepscaler/13_jan/dataset_env_var_launch_with_shuffle_16k.sh'

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=720 \
    --pass_run_name=False \
    --custom_invocation='MODEL_PATH="/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step500-hf" DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_hard" bash examples/deepscaler/13_jan/dataset_env_var_launch_with_shuffle_16k.sh'


python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=720 \
    --pass_run_name=False \
    --custom_invocation='MODEL_PATH="/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step400-hf" DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_easy" bash examples/deepscaler/13_jan/dataset_env_var_launch_with_shuffle_16k.sh'

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-460-chkpt \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=720 \
    --pass_run_name=False \
    --custom_invocation='MODEL_PATH="/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step460-hf" DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_easy" bash examples/deepscaler/13_jan/dataset_env_var_launch_with_shuffle_16k.sh'

python /capstor/scratch/cscs/smcleish/llnl-tools/launch_daint.py \
    --output_dir=/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs \
    --run_name=deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-500-chkpt \
    --nodes=2 \
    --gpus_per_node=4 \
    --ntasks_per_node=1 \
    --cpus_per_task=72 \
    --minutes=720 \
    --pass_run_name=False \
    --custom_invocation='MODEL_PATH="/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step500-hf" DS_NAME="deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_easy" bash examples/deepscaler/13_jan/dataset_env_var_launch_with_shuffle_16k.sh'
