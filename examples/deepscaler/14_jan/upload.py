from transformers import AutoModelForCausalLM, AutoTokenizer

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step400-hf"
# repo_name = "deepscaler-1.5b-8k-easy-first-run-with-shuffle-step400"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step400-hf"
# repo_name = "deepscaler-1.5b-8k-hard-first-run-with-shuffle-step400"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-500-chkpt/checkpoints/deepscaler-1.5b-8k-step200-hf"
# repo_name = "deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-500-chkpt-step-200"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt/checkpoints/deepscaler-1.5b-8k-step200-hf"
# repo_name = "deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt-step-200"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step500-hf"
# repo_name = "deepscaler-1.5b-8k-easy-first-run-with-shuffle-step500"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step500-hf"
# repo_name = "deepscaler-1.5b-8k-hard-first-run-with-shuffle-step500"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt/checkpoints/deepscaler-1.5b-8k-step200-hf"
# repo_name = "deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt-step-200"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt/checkpoints/deepscaler-1.5b-8k-step200-hf"
# repo_name = "deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt-step-200"


# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-500-chkpt/checkpoints/deepscaler-1.5b-8k-step400-hf"
# repo_name = "deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-500-chkpt-step-400"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt/checkpoints/deepscaler-1.5b-8k-step400-hf"
# repo_name = "deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt-step-400"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt/checkpoints/deepscaler-1.5b-8k-step400-hf"
# repo_name = "deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt-step-400"

local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt/checkpoints/deepscaler-1.5b-8k-step400-hf"
repo_name = "deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt-step-400"

model = AutoModelForCausalLM.from_pretrained(local_path)
tokenizer = AutoTokenizer.from_pretrained(local_path)

username = "smcleish"
model.push_to_hub(f"{username}/{repo_name}")
tokenizer.push_to_hub(f"{username}/{repo_name}")
