
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.attn import ExLlamaV2Attention

import chat_prompts

import torch

nouns = [
  "red apples",
  "tall trees",
  "blue flowers",
  "black cats",
  "hot dogs",
  "soft pillows",
  "white clouds",
  "green leaves",
  "wild animals",
  "tiny insects",
  "big elephants",
  "yellow bananas",
  "purple grapes",
  "slippery rocks",
  "old books",
  "happy children",
  "noisy birds",
  "smooth stones",
  "shiny diamonds",
  "thick blankets",
  "bright lights",
  "colorful balloons",
  "heavy backpacks",
  "furry kittens",
  "rusty nails",
  "hard bricks",
  "crispy fries",
  "bumpy roads",
  "sour candies",
  "young puppies",
  "cool sunglasses",
  "sharp knives",
  "sweet desserts",
  "salty snacks",
  "sticky notes",
  "hot peppers",
  "tender meats",
  "crunchy chips",
  "spicy sauces",
  "silky scarves",
  "sturdy chairs",
  "fragrant flowers",
  "fresh vegetables",
  "juicy fruits",
  "creamy soups",
  "fluffy clouds",
  "warm fires",
  "frozen treats",
  "smooth creams",
  "crispy crackers",
  "sweet cookies",
  "cold drinks",
  "soft blankets",
  "hard candies",
  "colorful crayons",
  "sharp thorns",
  "tasty treats",
  "muddy puddles",
  "foggy days",
  "sunny beaches",
  "cloudy skies",
  "stormy nights",
  "windy days",
  "rainy afternoons",
  "snowy winters",
  "icy roads",
  "frosty mornings",
  "thundering storms",
  "lightning strikes",
  "hailing stones",
  "chirping birds",
  "howling winds",
  "buzzing bees",
  "croaking frogs",
  "purring cats",
  "barking dogs",
  "snarling wolves",
  "fluttering butterflies",
  "slithering snakes",
  "galloping horses",
  "hoofed deer",
  "sleek cheetahs",
  "roaring lions",
  "gentle giraffes",
  "stubborn donkeys",
  "curious monkeys",
  "playful dolphins",
  "majestic whales"
]

pairs = [
    [f"Are {q} good?", f"Are {q} bad?"] for q in nouns
]

def format_prompt(p):
    return chat_prompts.PromptFormat_llama3().first_prompt() \
        .replace("<|system_prompt|>", chat_prompts.PromptFormat_llama3().default_system_prompt()) \
        .replace("<|user_prompt|>", p)

model_directory =  "../exllamav2/Llama-3-8B-exl2/"

config = ExLlamaV2Config(model_directory)
config.max_output_len = 1
config.max_batch_size = 2
model = ExLlamaV2(config)
print("Loading model: " + model_directory)

cache = ExLlamaV2Cache(model, lazy = True, batch_size=2)
model.load_autosplit(cache)

lora_directory = "../exllamav2/Llama-3-8B-instruct-r128/"
lora = ExLlamaV2Lora.from_directory(model, lora_directory)

tokenizer = ExLlamaV2Tokenizer(config)

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0

layers = defaultdict(list)

def hook(act, idx, mod):
    if isinstance(mod, (ExLlamaV2MLP, ExLlamaV2Attention)) and 20 <= idx <= 60:
        difference = act[1, -1, :] - act[0, -1, :]
        #print(torch.linalg.norm(difference).cpu().item(), act.shape, idx)
        layers[idx].append(difference.cpu().numpy())

for b, (positive, negative) in enumerate(tqdm(pairs)):
    print(format_prompt(positive))
    result = generator.generate_simple([
        format_prompt(positive), format_prompt(negative)
    ], settings, 8, seed = 1234, add_bos = True, activation_edit_hooks=[hook], loras=[lora], encode_special_tokens=True, decode_special_tokens=True)
    print(*result, sep="\n-\n")

'''
control_vector = {}

for idx, vectors in layers.items():
    vectors = np.vstack(vectors)
    relvectors = vectors - np.mean(vectors, axis=0)
    pca = PCA(n_components=1, whiten=False).fit(relvectors)
    principal_component = pca.components_[0]
    control_vector[idx] = torch.Tensor(principal_component).to("cuda").half()

for b, (positive, negative) in enumerate(pairs):
    print(f"{b + 1} of {len(pairs)}...")

    used = set()
    def hook(act, idx, mod):
        global used
        control = control_vector.get(idx)
        if control is not None and idx not in used:
            used.add(idx)
            #ctrl_components = act @ control
            #projection = ctrl_components.unsqueeze(-1).expand_as(act) * control.unsqueeze(0).unsqueeze(0).expand_as(act)
            #return act - projection
            #print(act.shape, control.shape)

            return act + (1 * control)

    result = generator.generate_simple([
        format_prompt(positive), format_prompt(negative)
    ], settings, 32, seed = 1234, add_bos = True, activation_edit_hooks=[hook])
    print(*result, sep="\n-\n")
'''