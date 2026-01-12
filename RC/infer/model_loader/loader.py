import os
from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from ..logger import logging
from .patcher import patch_tokenizer, patch_config, patch_model
from .adapter import init_adapter
from .utils import register_autoclass, count_parameters


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedModel
    
    from ..args import ModelArgs


logger = logging.get_logger(__name__)    


def load_tokenizer(model_args) -> "PreTrainedTokenizer":
    init_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer. Please check the `model_name_or_path`.") from e
    
    patch_tokenizer(tokenizer, model_args)
        
    return tokenizer


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArgs",
) -> "PreTrainedModel":
    init_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    patch_config(config, model_args, init_kwargs)
    
    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
    
    model = AutoModelForCausalLM.from_pretrained(**init_kwargs)
    patch_model(model)
    register_autoclass(config, model, tokenizer)  # register self-define modules for Auto classes
    
    model = init_adapter(model, model_args)
    
    model.requires_grad_(False)
    for param in model.parameters():
        if param.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
            param.data = param.data.to(model_args.compute_dtype)
    model.eval()

    _, all_params = count_parameters(model)

    param_stats = f"all params: {all_params:,}"
    logger.info_rank0(param_stats)
    
    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")
    
    return model
    