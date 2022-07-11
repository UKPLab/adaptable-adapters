from typing import List, Optional
from dataclasses import dataclass, field

import os
import wandb
import math
import time
import logging

from transformers import MultiLingAdapterArguments, TrainingArguments

SWITCH_INPUT_CHOICES = ["minimal", "pfeiffer"]

logger = logging.getLogger(__name__)


LOAD_DATASET_ARGS = {
    "rte": ("glue", "rte"),
    "qqp": ("glue", "qqp"),
    "cola": ("glue", "cola"),
    "mnli": ("glue", "mnli"),
    "mrpc": ("glue", "mrpc"),
    "qnli": ("glue", "qnli"),
    "sst2": ("glue", "sst2"),
    "stsb": ("glue", "stsb"),
    "trivia_qa": ("trivia_qa", "rc")
}


TASKS = {'rte', 'qqp', 'cola', 'mnli', 'mrpc', 'qnli', 'sst2', 'stsb'}

METRIC_FOR_BEST_MODEL = {
    "mnli": "eval_accuracy",
    "qqp": "eval_f1",
    "qnli": "eval_accuracy",
    "sst2": "eval_accuracy",
    "cola": "eval_matthews_correlation",
    "stsb": "eval_spearmanr",
    "mrpc": "eval_f1",
    "rte": "eval_accuracy",
    "wnli": "eval_accuracy",
}


@dataclass
class SwitchArgsMixin:

    # Check directly with wandb.
    check_switches_from_wandb: bool = False

    # Drop the skip-connections of adapters when using switch.
    adapter_drop_skip_connections: bool = False
    adapter_drop_skip_connections_training_only: bool = False


    # Put adapters at some fixed locations.
    adapters_at: List[int] = field(default_factory=list)
    # Number of layers in the base transformer
    # Default is set to 24 that is the number of layers in BERT-large used in the paper
    layer_num: int = 24


    # Temperature control.
    temp_N: int = 1
    temp_r: float = None
    temp_initial: float = 0.1
    temp_min: float = 0.1

    # Where to put switches.
    switches_at: List[int] = field(default_factory=list)

    # Fixed switch positions.
    fixed_configuration: List[int] = None


    # If switches are used they use the same inputs.
    use_switches: bool = False
    switch_inputs: List[str] = field(default_factory=list)


    # Default adapter.
    default_adapter: str = "rational"

    # Fix the switches
    fix_rational_switch: bool = False

    # Probability for soft fixed
    prob_for_soft_fixed: float = 0.9

    # Learning rate for probabilities.
    lr_for_switches: float = 0.05


    # Learning rate for the rational adapters.
    lr_for_rational_activations: float = 0.01

    default_adapter_non_linearity: str = "rational:one"



    def __post_init__(self):
        super().__post_init__()

        if self.check_switches_from_wandb and len(self.adapters_at) == 0:
            filters = {
                'config.seed': self.seed,
                'config.task_name': self.task_name,
                'config.baseline': False,
                'config.baseline_bert': False,
                'config.baseline_leave_out_all': False,
            }

            api = wandb.Api()
            runs = api.runs(os.environ['WANDB_PROJECT'], filters=filters)
            assert len(runs) == 1, "We expect only one run with this configuration."
            hist = runs[0].history()
            for i in range(12):
                tag = f'train/layer.{i}.prob.1'
                if hist[tag][hist[tag].notna()].iloc[-1] > 0.5:
                    self.adapters_at.append(i)

        if self.use_switches:
            if len(self.switch_inputs) == 0:
                raise ValueError(
                    "Please provide the inputs to the switches with `switch_inputs`"
                )

            for switch_input in self.switch_inputs:
                if switch_input.split(":")[0] not in SWITCH_INPUT_CHOICES:
                    raise ValueError("Incorrect switch options")



        if self.fixed_configuration is not None:
            assert len(self.fixed_configuration) == self.layer_num

        if self.lr_for_switches is None:
            self.lr_for_switches = self.learning_rate

        if self.lr_for_rational_activations is None:
            self.lr_for_rational_activations = self.learning_rate

        # Default value for temp_r.
        if self.temp_r is None:
            n = self.num_train_epochs / 2
            self.temp_r = -math.log(self.temp_min / self.temp_initial) / n



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    baseline: bool = False
    baseline_bert: bool = False
    baseline_leave_out_all: bool = False

    layer_num: int = field(
        default=24,
        metadata={
            "help": "The number of layers in the specified model in model_name_or_path"
        }
    )

    model_name_or_path: str = field(
        default="bert-large-uncased",
        metadata={
            "help": "Path to pretrained model or model identifier "
            "from huggingface.co/models",
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory to store the pretrained models "
            "downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the "
            "tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch "
            "name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running "
            "`transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class SimpleDataTrainingArguments:

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate "
            "the number of training examples to this value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate "
            "the number of validation examples to this value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate "
            "the number of test examples to this value if set."
        },
    )


@dataclass
class BaseArgs(
    SwitchArgsMixin,
    MultiLingAdapterArguments,
    ModelArguments,
    SimpleDataTrainingArguments,
    TrainingArguments
):

    # Data arguments:
    task_name: str = field(
        default=None,
        metadata=dict(help="The name of the task to train on: " + ", ".join(TASKS))
    )

    # Save the rational plots.
    save_rational_plots: bool = False

    # Use the validaton splt for testing.
    low_resources: int = None

    # Some extra defaults.
    load_best_model_at_end: bool = True
    num_train_epochs: int = 10
    learning_rate: float = 1e-4
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 2
    switch_inputs: List[str] = field(
        default_factory=lambda: ['minimal:identity', 'pfeiffer:rational:one']
    )

    def __post_init__(self):
        if self.task_name in METRIC_FOR_BEST_MODEL:
            self.metric_for_best_model = METRIC_FOR_BEST_MODEL[self.task_name]
        super().__post_init__()

@dataclass
class GLUEArgs(BaseArgs):

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after "
            "tokenization. Sequences longer than this will be truncated, "
            "sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to "
            "the maximum length in the batch."
        },
    )

    # Shuffle the samples
    shuffle_samples: bool = False

    def __post_init__(self):

        # Output dir requires some post-processing.
        self.output_dir = self.output_dir.replace("%t", str(int(1e7 * time.time())))

        TASK_TO_KEYS = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }

        if self.task_name is not None:
            self.task_name = self.task_name.lower()

        if self.task_name not in TASK_TO_KEYS:
            raise ValueError(
                "Unknown task_name, you should pick one in " + ",".join(TASK_TO_KEYS)
            )

        super().__post_init__()
