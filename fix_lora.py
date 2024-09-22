import torch
from safetensors import safe_open
from safetensors.torch import save_file

tensors = {}
with safe_open("../exllamav2/Llama-3-8B-instruct-r128/adapter_model.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key.removeprefix("base_model.")] = f.get_tensor(key)
       print(key)

tensors = {}
save_file(tensors, "model.safetensors")