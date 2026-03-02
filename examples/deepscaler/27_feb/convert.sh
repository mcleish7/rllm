python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-easy/checkpoints/global_step_200/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-easy/deepscaler-1.5b-8k-dapo-step200-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-easy/checkpoints/global_step_400/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-easy/deepscaler-1.5b-8k-dapo-step400-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-easy/checkpoints/global_step_600/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-easy/deepscaler-1.5b-8k-dapo-step600-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-hard/checkpoints/global_step_200/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-hard/deepscaler-1.5b-8k-dapo-step200-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-hard/checkpoints/global_step_400/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-hard/deepscaler-1.5b-8k-dapo-step400-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-random/checkpoints/global_step_200/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-random/deepscaler-1.5b-8k-dapo-step200-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-random/checkpoints/global_step_400/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-random/deepscaler-1.5b-8k-dapo-step400-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-random/checkpoints/global_step_500/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-random/deepscaler-1.5b-8k-dapo-step500-hf