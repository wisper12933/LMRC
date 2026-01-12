from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from transformers import GenerationMixin, PreTrainedTokenizerBase

from ..logger import logging
from .utils import infer_optim_dtype


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PretrainedConfig, PreTrainedModel
    
    from ..args import ModelArgs


logger = logging.get_logger(__name__)


def _print_attn_implementation(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info_rank0("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info_rank0("Using torch SDPA for faster training and inference.")
    else:
        logger.info_rank0("Using vanilla attention implementation.")


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArgs") -> None:
    if "PreTrainTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length        


def patch_config(
    config: "PretrainedConfig",
    model_args: "ModelArgs",
    init_kwargs: dict[str, Any],
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto":
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "dtype", None))
    
    # set KV cache
    setattr(config, "use_cache", model_args.use_cache)
    if hasattr(config, "text_config"):
        setattr(config.text_config, "use_cache", model_args.use_cache)
    
    if model_args.use_cache:
        logger.info_rank0("KV cache is enabled for faster generation.")
    else:
        logger.info_rank0("KV cache is disabled.")
    
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage
    init_kwargs["torch_dtype"] = model_args.compute_dtype
    if init_kwargs["low_cpu_mem_usage"]:
        if "device_map" not in init_kwargs and model_args.device_map:
            logger.info_rank0(f"Setting device_map to {model_args.device_map}")
            init_kwargs["device_map"] = model_args.device_map


def patch_model(
    model: "PreTrainedModel",
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True
    
    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(GenerationMixin.generate, model)
    
    _print_attn_implementation(model.config)
