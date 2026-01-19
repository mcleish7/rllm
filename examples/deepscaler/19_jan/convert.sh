python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-500-chkpt/checkpoints/global_step_400/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-500-chkpt/checkpoints/deepscaler-1.5b-8k-step400-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt/checkpoints/global_step_400/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt/checkpoints/deepscaler-1.5b-8k-step400-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt/checkpoints/global_step_400/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt/checkpoints/deepscaler-1.5b-8k-step400-hf

# python -m verl.model_merger merge \
#   --backend fsdp \
#   --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt/checkpoints/global_step_400/actor \
#   --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt/checkpoints/deepscaler-1.5b-8k-step400-hf