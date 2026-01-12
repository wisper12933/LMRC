from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union
import yaml

import torch
from transformers import HfArgumentParser, GenerationConfig
from transformers.utils import is_torch_cuda_available

from .logger import logging


logger = logging.get_logger(__name__)


@dataclass
class DataArgs:
    r"""Args pertaining to what data we are going to input our model for training and evaluation."""
    
    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    default_system: Optional[str] = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
            
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    
@dataclass
class ModelArgs:
    r"""Args pertaining to what model we are going to use for MAML."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    adapter_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    load_type: Literal["full", "lora"] = field(
        default="lora",
        metadata={"help": "The type of model to load."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co."},
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."},
    )
    # Do not specify these args, they are derived from other args
    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        init=False,
        metadata={"help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    device_map: Optional[Union[str, dict[str, Any]]] = field(
        default=None,
        init=False,
        metadata={"help": "Device map for model placement, derived from training stage. Do not specify it."},
    )
    model_max_length: Optional[int] = field(
        default=None,
        init=False,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    
    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationArgs:
    r"""Args pertaining to generation configuration."""
    
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."},
    )
    temperature: float = field(
        default=0.1,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: float = field(
        default=0.9,
        metadata={
            "help": (
                "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
            )
        },
    )
    top_k: int = field(
        default=20,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."},
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."},
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."},
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "Whether or not to remove special tokens in the decoding."},
    )
    
    def to_dict(self, obey_generation_config: bool = False) -> dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)

        if obey_generation_config:
            generation_config = GenerationConfig()
            for key in list(args.keys()):
                if not hasattr(generation_config, key):
                    args.pop(key)

        return args


def read_specify_task_eval_args(args) -> tuple[DataArgs, ModelArgs, GenerationArgs]:
    r"""Read and parse the configuration file for specify-task evaluation. (No need for loading Datasets)"""
    with open(args.config, "r", encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    parser = HfArgumentParser([DataArgs, ModelArgs, GenerationArgs])
    data_args, model_args, generation_args = parser.parse_dict(config_dict)
    
    if model_args.adapter_name_or_path is not None and model_args.load_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")
    
    if is_torch_cuda_available():
        model_args.device_map = "auto"
    else:
        model_args.device_map = {"": torch.device("cpu")}
    model_args.model_max_length = data_args.cutoff_len
    
    return data_args, model_args, generation_args