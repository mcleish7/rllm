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

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt/checkpoints/deepscaler-1.5b-8k-step400-hf"
# repo_name = "deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-500-chkpt-step-400"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step200-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-step-200"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step300-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-step-300"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step400-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-step-400"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle/checkpoints/deepscaler-1.5b-8k-step500-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-step-500"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/eepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt-16k-400-chkpt/deepscaler-1.5b-8k-step200-hf"
# repo_name = "deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt-16k-400-chkpt-step-200"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/eepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt-16k-400-chkpt/deepscaler-1.5b-8k-step400-hf"
# repo_name = "deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt-16k-400-chkpt-step-400"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/eepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt-16k-200-chkpt/deepscaler-1.5b-8k-step200-hf"
# repo_name = "deepscaler-1.5b-8k-easy-first-run-with-shuffle-8k-400-chkpt-16k-200-chkpt-step-200"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt-16k-400-chkpt/deepscaler-1.5b-8k-step200-hf"
# repo_name = "deepscaler-1.5b-8k-hard-first-run-with-shuffle-8k-400-chkpt-16k-400-chkpt-step-200"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-400-chkpt/deepscaler-1.5b-8k-step300-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-400-chkpt-step-300"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-400-chkpt/deepscaler-1.5b-8k-step200-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-400-chkpt-step-200"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-400-chkpt/deepscaler-1.5b-8k-step100-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-400-chkpt-step-100"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-300-chkpt/deepscaler-1.5b-8k-step400-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-300-chkpt-step-400"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-300-chkpt/deepscaler-1.5b-8k-step300-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-300-chkpt-step-300"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-300-chkpt/deepscaler-1.5b-8k-step200-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-300-chkpt-step-200"

# local_path = "/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-300-chkpt/deepscaler-1.5b-8k-step100-hf"
# repo_name = "deepscaler-1.5b-8k-reproduce-first-run-with-shuffle-8k-300-chkpt-step-100"

lst = [
    ("/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-easy/deepscaler-1.5b-8k-dapo-step200-hf", "deepscaler-1.5b-8k-dapo-easy-step200-hf"),
    ("/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-easy/deepscaler-1.5b-8k-dapo-step400-hf", "deepscaler-1.5b-8k-dapo-easy-step400-hf"),
    ("/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-hard/deepscaler-1.5b-8k-dapo-step200-hf", "deepscaler-1.5b-8k-dapo-hard-step200-hf"),
    ("/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-hard/deepscaler-1.5b-8k-dapo-step400-hf", "deepscaler-1.5b-8k-dapo-hard-step400-hf"),
    ("/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-random/deepscaler-1.5b-8k-dapo-step200-hf", "deepscaler-1.5b-8k-dapo-random-step200-hf"),
    ("/capstor/scratch/cscs/smcleish/rllm_daint_291_129_uenv_venv_rllm_2/tuo_outputs/deepscaler-1.5b-8k-dapo-random/deepscaler-1.5b-8k-dapo-step400-hf", "deepscaler-1.5b-8k-dapo-random-step400-hf"),
]

for local_path, repo_name in lst:
    model = AutoModelForCausalLM.from_pretrained(local_path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    username = "smcleish"
    model.push_to_hub(f"{username}/{repo_name}")
    tokenizer.push_to_hub(f"{username}/{repo_name}")
