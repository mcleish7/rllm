uenv start --view=modules prgenv-gnu/25.6:v2
module load aws-ofi-nccl cuda nccl libfabric gcc # note no python here
source /capstor/scratch/cscs/smcleish/daint_291_129_uenv_venv_rllm_2/.venv/bin/activate
cd rllm_daint_291_129_uenv_venv_rllm_2/

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-500-chkpt/checkpoints/global_step_200/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-500-chkpt/checkpoints/deepscaler-1.5b-8k-step200-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-460-chkpt/checkpoints/global_step_200/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-460-chkpt/checkpoints/deepscaler-1.5b-8k-step200-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt/checkpoints/global_step_200/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt/checkpoints/deepscaler-1.5b-8k-step200-hf


python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt/checkpoints/global_step_200/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt/checkpoints/deepscaler-1.5b-8k-step200-hf

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt/checkpoints/global_step_200/actor \
  --target_dir /capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt/checkpoints/deepscaler-1.5b-8k-step200-hf
