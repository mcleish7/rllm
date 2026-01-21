
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/global_step_200/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step200-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/global_step_300/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step300-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/global_step_400/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step400-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/global_step_500/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step500-hf