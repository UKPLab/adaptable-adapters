from transformers import Trainer
from transformers.trainer_utils import PredictionOutput, speed_metrics


from typing import Optional, Dict, List
import re
import time
import math

import torch

from datasets import Dataset

from args import BaseArgs


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
    pass


