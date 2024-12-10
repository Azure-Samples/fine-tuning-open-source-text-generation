from pydantic.dataclasses import dataclass
from typing import Dict, Any
from pydantic import Field
import yaml


@dataclass
class RootConfig:
    """Base class for managing configurations."""

    @classmethod
    def load_from_yaml(cls, filepath: str):
        """Load configuration from a YAML file."""
        with open(filepath, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        return cls(**yaml_data)

    @classmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the TrainingArgsConfig to a dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


@dataclass
class ModelingConfig(RootConfig):
    """Configuration for the Modeling Process."""

    model_name: str = ""
    evaluation_metric: str = ""
    track_experiment: bool = False
    experiment_name: str = ""
    run_name: str = ""
    task: str = ""


@dataclass
class DeploymentConfig(ModelingConfig):
    metric_order: str = ""
    mode: str = "last"
    environment: str = "sandbox"


@dataclass
class DeepSpeedConfig(RootConfig):
    fp16: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    )

    optimizer: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        }
    )

    scheduler: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        }
    )

    zero_optimization: Dict[str, Any] = Field(
        default_factory=lambda: {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        }
    )

    # gradient_accumulation_steps: str = "auto"
    gradient_clipping: str = "auto"
    steps_per_print: int = 1000
    train_batch_size: str = 1
    # train_micro_batch_size_per_gpu: str = "auto"
    wall_clock_breakdown: bool = False


@dataclass
class TrainingArgsConfig(RootConfig):
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 1
    eval_accumulation_steps: int = -1
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    fp16: bool = True
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "linear"
    max_steps: int = (
        1  # If set to a positive number, the total number of training steps to perform. Overrides `epochs`."
    )
    # # In case of using a finite iterable dataset the training may stop before reaching the set number of steps"
    # # when all data is exhausted
    logging_steps: int = 10
    warmup_steps: int = 5
    ddp_find_unused_parameters: bool = False
    eval_strategy: str = "epoch"  # To calculate metrics per epoch
    save_strategy: str = "epoch"
    logging_strategy: str = "steps"  # Extra: to log training data stats for loss
    load_best_model_at_end: bool = (
        True  # load_best_model_at_end requires the saving steps to be a round multiple of the evaluation steps
    )
    num_train_epochs: int = 3
    metric_for_best_model: str = "loss"
    predict_with_generate = True
    deepspeed = DeepSpeedConfig()


@dataclass
class ModelLoraConfig:
    task_type: str = "CAUSAL_LM"
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    bias: str = "none"
