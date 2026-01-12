from typing import TYPE_CHECKING

import torch
from transformers.utils import is_torch_cuda_available, is_torch_bf16_gpu_available

from ..logger import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer


logger = logging.get_logger(__name__)


_is_fp16_available = is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available()
except Exception:
    _is_bf16_available = False


def count_parameters(model: "torch.nn.Module") -> tuple[int, int]:
    r"""Return the number of trainable parameters and number of all parameters in the model."""
    trainable_params, all_params = 0, 0
    
    for param in model.parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return trainable_params, all_params


def infer_optim_dtype(model_dtype: "torch.dtype") -> "torch.dtype":
    r"""Infer the optimal dtype according to the model_dtype and device compatibility."""
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32
    

def register_autoclass(config: "PretrainedConfig", model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()
