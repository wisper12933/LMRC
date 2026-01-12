from typing import TYPE_CHECKING

from peft import PeftModel

from ..logger import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    
    from ..args import ModelArgs


logger = logging.get_logger(__name__)
    

def _setup_lora_tuning(
    model: "PreTrainedModel",
    model_args: "ModelArgs",
) -> "PeftModel":
    if model_args.adapter_name_or_path is not None:
        # inference or training with pre-trained adapter
        adapter_to_merge = model_args.adapter_name_or_path
        
        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
        }
        
        model = PeftModel.from_pretrained(model, adapter_to_merge, is_trainable=False, **init_kwargs)
        model = model.merge_and_unload()
        
        logger.info_rank0(f"Loaded adapter from {adapter_to_merge}.")
    
    else:
        raise ValueError("Inference with LoRA adapter requires `adapter_name_or_path`.")
    
    return model


def init_adapter(
    model: "PreTrainedModel",
    model_args: "ModelArgs",
) -> "PreTrainedModel":
    r"""Initialize the adapters.

    Support full-parameter and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    
    if model_args.load_type == "full":
        pass
    elif model_args.load_type == "lora":
        model = _setup_lora_tuning(model, model_args)
    else:
        raise ValueError(f"Unsupported training type: {model_args.load_type}")
    
    return model
    