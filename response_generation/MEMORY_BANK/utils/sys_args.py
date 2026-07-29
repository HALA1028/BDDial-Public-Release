from transformers import Trainer, HfArgumentParser
from dataclasses import dataclass, field
import os

DEFAULT_MEMORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "memories"))

@dataclass
class DataArguments:
    memory_search_top_k: int = field(default=2)
    memory_basic_dir: str = field(default=DEFAULT_MEMORY_DIR)
    memory_file: str = field(default='update_memory.json')
    language: str = field(default='cn')
    max_history: int = field(default=7,metadata={"help": "maximum number for keeping current history"},)
    enable_forget_mechanism: bool = field(default=False)
    HY_dir: str = field(default='HY_origin.json')
@dataclass
class ModelArguments:
    model_type: str = field(
        default="chatglm",
        metadata={"help": "model type: chatglm / belle"},
    )
    base_model: str = field(
        default="THUDM/chatglm-6b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    adapter_model: str = field(
        default=None,
        metadata={"help": "Path to lora adapter model"},
    )
    ptuning_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to pretrained prefix embedding of ptuning"},
    )
    

    # prompt_column

data_args,model_args = HfArgumentParser(
    (DataArguments,ModelArguments)
).parse_args_into_dataclasses()
