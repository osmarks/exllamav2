import torch
from typing import Callable
import enum

ExLlamaV2ActivationEditingHook = Callable[[torch.Tensor, int, torch.nn.Module], torch.Tensor | None]