from transformers import Trainer
from transformers.trainer import WEIGHTS_NAME
from transformers.trainer_utils import PredictionOutput, speed_metrics


from typing import Optional, Dict, List
import re
import time
import math

import torch

from datasets import Dataset

from args import BaseArgs
import os

class CustomTrainerMixin(Trainer):

    args: BaseArgs



    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
    ):
        results = super().predict(
            test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )


        return results

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        prefix = metric_key_prefix

        # Evaluate the position of the switches.
        metrics = {}

        if eval_dataset is not None:
            metrics[f"{metric_key_prefix}_samples"] = len(eval_dataset)

        # Call the original evaluation loop.
        metrics.update(
            super().evaluate(
                eval_dataset, ignore_keys, metric_key_prefix=metric_key_prefix
            )
        )

        self.log(metrics)
        self.log_metrics("eval", metrics)
        self.save_metrics("eval", metrics)
        return metrics


class GLUETrainer(CustomTrainerMixin):
    def _load_checkpoint(self, checkpoint_folder):
        if self.do_save_full_model:
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(os.path.join(checkpoint_folder, WEIGHTS_NAME), map_location="cpu")
            # If the model is on the GPU, it still works!
            self._load_state_dict_in_model(state_dict)
        if self.do_save_adapters:
            adapter_loaded = False
            if os.path.isdir(checkpoint_folder):
                for file_name in os.listdir(checkpoint_folder):
                    if os.path.isdir(os.path.join(checkpoint_folder, file_name)):
                        if "," in file_name:
                            self.model.load_adapter_fusion(os.path.join(checkpoint_folder, file_name))
                            adapter_loaded = True
                        else:
                            self.model.load_adapter(
                                os.path.join(os.path.join(checkpoint_folder, file_name))
                            )
                            adapter_loaded = True

            if not adapter_loaded:
                raise Exception("Can't find a valid checkpoint at {}".format(checkpoint_folder))
