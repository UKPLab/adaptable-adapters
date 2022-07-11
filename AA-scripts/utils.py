from typing import Optional, List, Dict
import re
import math
import logging
import hashlib
import numpy as np

import torch
from torch import nn

import wandb
from tqdm.auto import tqdm

from transformers import TrainerState, TrainerCallback, AdapterConfig
from transformers.integrations import WandbCallback
import transformers.adapters.composition as ac

from rational.torch import Rational

from datasets import Dataset
from dataclasses import asdict

from args import BaseArgs
import json


logger = logging.getLogger(__name__)


def format_args(args: BaseArgs) -> str:
    msg = []
    for name, value in sorted(asdict(args).items()):
        msg.append(f" - {name} = {value}")
    return "\n".join(msg)


def split_datasets(train_ds, eval_ds, n: int = None):
    logger.info(
        "Spliting the train/eval datasets into train/eval/test by "
        "using 75% and 25% of train as train and eval and eval as test."
    )
    if n is None:
        n = len(train_ds)
        logger.info(f"Using the whole train dataset of {n} samples.")
    else:
        logger.info(f"Reducing the train dataset to only {n} samples.")

    split_at = int(n * 0.75)
    train_ds = train_ds.shuffle()
    new_eval_ds = train_ds.select(range(split_at, n))
    new_train_ds = train_ds.select(range(split_at))
    return new_train_ds, new_eval_ds, eval_ds


def hash_dataset(ds: Dataset, keys: List[str] = []) -> str:
    m = hashlib.sha256()
    for sample in tqdm(ds, desc='Hashing the train dataset'):
        for key in keys:
            if key is None:
                continue
            v = sample[key]
            if isinstance(v, str):
                m.update(v.encode('utf-8'))
    return m.hexdigest()


def _rewrite_logs(d):
    new_d = {}

    re_eval = re.compile("^eval_")
    re_train = re.compile("^train_")
    re_test = re.compile("^test_")

    for k, v in d.items():
        if re_eval.match(k):
            new_d[re_eval.sub("eval/", k)] = v
        elif re_test.match(k):
            new_d[re_test.sub("test/", k)] = v
        elif re_train.match(k):
            new_d[re_train.sub("train/", k)] = v
        else:
            new_d["train/" + k] = v

    return new_d


class CustomWandbCallback(WandbCallback):
    _re = re.compile(r".*\.layer\.([0-9]+)\.output.*\.switch_logits")

    def on_step_begin(self, args: BaseArgs, state: TrainerState, control, **kwargs):
        if args.save_rational_plots:
            if state.global_step % args.logging_steps == 0:
                Rational.save_all_inputs(True)

    def on_step_end(self, args: BaseArgs, state: TrainerState, control, **kwargs):
        if args.save_rational_plots:
            if state.global_step % args.logging_steps == 0 and len(Rational.list) > 0:
                Rational.capture_all(f"Global Step {state.global_step}")
                filename = f"{args.output_dir}/ra_{state.global_step}.png"
                Rational.export_graphs(filename)
                self._wandb.log(
                    {
                        "train/rational_activations": wandb.Image(filename),
                        "train/global_step": state.global_step,
                    }
                )
                Rational.save_all_inputs(False)

    def on_log(self, args: BaseArgs, state, control, **kwargs):
        if self._wandb is None:
            return

        # Capture the model.
        model: nn.Module = kwargs.pop("model", None)

        if not self._initialized:
            self.setup(args, state, model)

        # Capture the logs.
        logs = kwargs.pop("logs", {}) or {}

        # Detect a prefix.
        prefix = None
        if any(n.startswith('eval_final_') for n in logs):
            prefix = 'eval_final'
        elif any(n.startswith('eval_') for n in logs):
            prefix = 'eval'
        elif any(n.startswith('test_') for n in logs):
            prefix = 'test'
        elif any(n.startswith('train_') for n in logs):
            prefix = 'train'
        elif 'loss' in logs:
            prefix = 'train'

        # Get the number of parameters that need training.
        params_t = [p for p in model.parameters() if p.requires_grad]

        # Total number of parameters.
        logs["num_params"] = sum(math.prod(p.size()) for p in params_t)
        if prefix is not None:
            logs[f'{prefix}_num_params'] = logs["num_params"]

        # Get the current temperature.
        temp = None
        for name, buf in model.named_buffers():
            if name.endswith(".switch_temperature"):
                temp = buf[0].item()
                break

        # Log the temperature.
        if temp is not None:
            logs["temperature"] = temp
            if prefix is not None:
                logs[f"{prefix}_temperature"] = temp

        # Logs probs.
        for name, param in model.named_parameters():
            match = self._re.match(name)
            if match:
                layer_idx = int(match.groups()[0])
                prob = torch.softmax(param, dim=-1)
                #prob = torch.softmax(param / temp, dim=-1)
                for i in range(prob.size()[0]):
                    p = prob[i].item()
                    logs[f"layer.{layer_idx}.prob.{i}"] = p
                    if prefix is not None:
                        logs[f"{prefix}_layer.{layer_idx}.prob.{i}"] = p

        if state.is_world_process_zero:
            logs = _rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})


class AdapterDropTrainerCallback(TrainerCallback):
  def on_step_begin(self, args, state, control, **kwargs):
    skip_layers = list(range(np.random.randint(0, 11)))
    kwargs['model'].set_active_adapters("pfeiffer", skip_layers=skip_layers)

  def on_evaluate(self, args, state, control, **kwargs):
    # Deactivate skipping layers during evaluation (otherwise it would use the
    # previous randomly chosen skip_layers and thus yield results not comparable
    # across different epochs)
    if kwargs['model'].training:
        kwargs['model'].set_active_adapters("pfeiffer", skip_layers=None)

class TemperatureControl(TrainerCallback):
    """
    This callback controls the temperature according to the rule
    in arxiv:1611.01144v5.
    """

    def on_init_end(self, args: BaseArgs, state, control, model=None, **kwargs):
        self._temp_buffers = []
        for name, buf in model.named_buffers():
            if name.endswith(".switch_temperature"):
                buf[0] = args.temp_initial
                self._temp_buffers.append(buf)

    def on_step_begin(self, args: BaseArgs, state: TrainerState, control, **kwargs,):

        # Use the epoch to control the temperature.
        t = args.temp_N * (state.epoch // args.temp_N)

        # Compute the temperature
        temp = max(args.temp_min, args.temp_initial * math.exp(-args.temp_r * t))

        # Set all te temperature buffers at the same value.
        for buf in self._temp_buffers:
            buf[0] = temp


def get_optimizer(model, args):
    # Split the parameters for differential learning rates.
    params_rational = []
    params_switches = []
    params_rest = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".switch_logits"):
            params_switches.append(p)
        elif n.endswith(".f.numerator"):
            params_rational.append(p)
        elif n.endswith(".f.denominator"):
            params_rational.append(p)
        else:
            params_rest.append(p)

    # Parameters for the optimizers
    params_optim = [
        {"params": params_rest},
        {"params": params_switches, "lr": args.lr_for_switches},
        {"params": params_rational, "lr": args.lr_for_rational_activations},
    ]

    return torch.optim.Adam(
        params_optim, lr=args.learning_rate, weight_decay=args.weight_decay
    )


def load_extra_adapters(model, args):
    # Add the adapters to the model according to the configuration in args.

    if args.baseline:
        logger.info("Using the baseline configuration.")
        model.add_adapter("pfeiffer", config=AdapterConfig.load("pfeiffer"))
        model.train_adapter("pfeiffer")

    elif args.use_switches:
        # Configurations
        switch_inputs = []
        # First collect the inputs for the switches.
        for s_input in args.switch_inputs:

            adapter_identifier = s_input.replace(":", "_")
            switch_inputs.append(adapter_identifier)

            adapter_name = s_input.split(":")[0]
            if adapter_name == "minimal":
                adapter_name = "rational"

            adapter_activation = None
            if len(s_input.split(":")) > 1:
                adapter_activation = s_input.split(":", 1)[1]

            # Define the configuration.
            config = {
                "non_linearity": adapter_activation,
                "drop_skip_connections": args.adapter_drop_skip_connections,
                "drop_skip_connections_training_only": args.adapter_drop_skip_connections_training_only
            }

            config = AdapterConfig.load(adapter_name, **config)

            # Add the switch adapters.
            model.add_adapter(adapter_identifier, config=config)

        adapter_switch = ac.Switch(*switch_inputs)
        switch_config = {
            "strategy": "global",
        }
        model.add_adapter_switch(adapter_switch, config=switch_config)
        model.train_adapter_switch(adapter_switch)

    elif len(args.adapters_at) > 0 and args.layer_num:
        leave_out = [i for i in range(args.layer_num) if i not in args.adapters_at]
        model.add_adapter(
            'pfeiffer', config=AdapterConfig.load(
                "pfeiffer",
                leave_out=leave_out,
                non_linearity=args.default_adapter_non_linearity,
            )
        )
        model.train_adapter("pfeiffer")

    # Get the number of parameters that need training.
    params = model.named_parameters()
    params_t = {n: p for n, p in params if p.requires_grad}

    # Total number of parameters.
    num_params = sum(math.prod(p.size()) for p in params_t.values())

    logger.info(f"Training {len(params_t)} params of total size {num_params}.")
    logger.info(
        "The following parameters will be trained:\n"
        + "\n".join(f" - {n}" for n in params_t)
    )

    return model


def make_jsonable(dict):
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except:
            return False
    arg_dict = vars(dict)
    _to_delete = []
    for key,value in arg_dict.items():
        if not is_jsonable(value):
            _to_delete.append(key)
    for key in _to_delete:
        arg_dict.pop(key)
        print(f"Dropped {key} to make jsonable")
    return arg_dict
