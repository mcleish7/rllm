from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/capstor/scratch/cscs/smcleish/deepscaler-1.5b-8k-step20-hf"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
