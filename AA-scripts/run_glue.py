#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task.
# Pointers for this are left as comments.

import logging
import json
import os
import random
import sys
import numpy as np
from datetime import datetime

from rational.torch import Rational
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from transformers.integrations import INTEGRATION_TO_CALLBACK
from transformers.adapters.layer import Adapter, AdapterSwitch

from utils import (
    load_extra_adapters,
    get_optimizer,
    CustomWandbCallback,
    TemperatureControl,
    hash_dataset,
    split_datasets,
    format_args,
    make_jsonable
)
from args import GLUEArgs as Args
from trainer import GLUETrainer as Trainer

Rational.use_kde = False

# Will error if the minimal version of Transformers is not installed. Remove at
# your own risks.
check_min_version("4.5.0")

task_to_keys = {
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


logger = logging.getLogger(__name__)


def _main():
    INTEGRATION_TO_CALLBACK["wandb"] = CustomWandbCallback
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
        force=True
    )
    parser = HfArgumentParser(Args)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 1:
        # Arguments passed by stdin as json.
        data = sys.stdin.read()
        print("Loading from STDIN: ", data)
        args = parser.parse_dict(json.loads(data))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]
    main(args)


import re
def get_adapters_switches(model):
    switches = []
    selected_layers = []
    useless_layers = []
    for name, param in model.named_parameters():
        if "switch_logits" in name:
            switches.append((name, param))
            layer_n = int(re.findall(r'\d+', name[11:-1])[0])
            #If the corresponding logit for the adapter layer is higher than that of the skip connection 
            if param[1].item() > param[0].item():
                selected_layers.append(layer_n)
            else:
                useless_layers.append(layer_n)

    return selected_layers, useless_layers, switches


def main(args: Args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    logger.info(f"Running a GLUE Task with arguments:\n{format_args(args)}")
    summary = {}

    # Hack for the custom wandb callback used.
    INTEGRATION_TO_CALLBACK["wandb"] = CustomWandbCallback

    # Detecting last checkpoint.
    last_checkpoint = None

    if (
        os.path.isdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists "
                "and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the `--output_dir` or "
                "add `--overwrite_output_dir` to train from scratch."
            )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training
    # and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded
    # automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column
    # called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such
    # column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script
    # does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that
    # only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {
            "train": args.train_file,
            "validation": args.validation_file,
        }

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if args.do_predict:
            if args.test_file is not None:
                train_extension = args.train_file.split(".")[-1]
                test_extension = args.test_file.split(".")[-1]
                if test_extension != train_extension:
                    raise ValueError(
                        "`test_file` should have the same extension "
                        "(csv or json) as `train_file`."
                    )
                data_files["test"] = args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`."
                )

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee
    # that only one local process can concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        args.config_name
        if args.config_name
        else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name
        else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )

    # Add the adapters.
    load_extra_adapters(model, args)

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    # Padding strategy
    if args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max
        # sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure
    # we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warn(
                "Your model seems to have been trained with labels, "
                "but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, "
                f"dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result."
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({args.max_seq_length}) "
            "is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using "
            f"max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):

        # Tokenize the texts
        sentences = (examples[sentence1_key],)
        if sentence2_key is not None:
            sentences += (examples[sentence2_key],)

        result = tokenizer(
            *sentences, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = []
            for label in examples["label"]:
                result["label"].append(-1 if label == -1 else label_to_id[label])
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not args.overwrite_cache,
    )

    # Default values for the datasets.
    train_dataset = eval_dataset = test_dataset = None

    if args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if args.max_train_samples is not None:
            if args.shuffle_samples:
                train_dataset = train_dataset.shuffle()
            train_dataset = train_dataset.select(range(args.max_train_samples))

    if args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")

        if args.task_name == "mnli":
            eval_dataset = datasets["validation_matched"]
        else:
            eval_dataset = datasets["validation"]

        if args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    if (
        args.do_predict
        or args.task_name is not None
        or args.test_file is not None
    ):
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets[
            "test_matched" if args.task_name == "mnli" else "test"
        ]
        if args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(args.max_test_samples))
        test_dataset.remove_columns_("label")

    # Split the datasets following the 75% for training and 25% for validation.
    train_dataset, eval_dataset, test_dataset = split_datasets(
        train_dataset, eval_dataset, args.low_resources
    )

    # Log a few random samples from the training set:
    if train_dataset is not None:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    metric = load_metric("glue", args.task_name)

    # compute_metrics
    # You can define your custom compute_metrics function. It takes
    # an `EvalPrediction` object (a namedtuple with a predictions and
    # label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        if is_regression:
            preds = np.squeeze(preds)
        else:
            preds = np.argmax(preds, axis=1)
        result = {}
        if args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
        elif is_regression:
            result["mse"] = ((preds - p.label_ids) ** 2).mean().item()
        else:
            result["accuracy"] = (preds == p.label_ids).astype(np.float32).mean().item()
        return result

    # Data collator will default to DataCollatorWithPadding, so we change it
    # if we already did the padding.
    if args.pad_to_max_length:
        data_collator = default_data_collator
    elif args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Default optimizer.
    optimizer = get_optimizer(model, args)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        do_save_full_model=not args.train_adapter,
        do_save_adapters=args.train_adapter,
        optimizers=(optimizer, None),
        #callbacks=[TemperatureControl],
    )

    # Training
    if train_dataset is not None:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(args.model_name_or_path):
            # Check the config from that potential checkpoint has the right
            # number of labels before using it as a checkpoint.
            model_cand = AutoConfig.from_pretrained(args.model_name_or_path)

            if model_cand.num_labels == num_labels:
                checkpoint = args.model_name_or_path

        # Compute a hash of the train dataset to confirm seeds create the same split.
        train_hash = hash_dataset(train_dataset, keys=task_to_keys[args.task_name])
        trainer.log({'train/dataset_hash': train_hash})

        train_result = trainer.train(resume_from_checkpoint=checkpoint)


        metrics = train_result.metrics
        max_train_samples = (
            args.max_train_samples
            if args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # Save the last model.
        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        summary["train_metrics"] = metrics

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        #trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval_final")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [args.task_name]
        eval_datasets = [eval_dataset]
        if args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            #max_val_samples = len(eval_dataset) #args.max_val_samples if args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = len(eval_dataset)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        summary["eval_metrics"] = metrics


    if args.do_predict:
        logger.info("*** Test ***")
        task = args.task_name
        result = trainer.predict(test_dataset=test_dataset)
        metrics = result.metrics.copy()

        max_test_samples = len(test_dataset)
        if args.max_test_samples is not None:
            max_test_samples = args.max_test_samples
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))

        print(metrics)
        trainer.log(metrics)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        summary["test_metrics"] = metrics

        predictions = result.predictions
        if is_regression:
            predictions = np.squeeze(predictions)
        else:
            predictions = np.argmax(predictions, axis=1)

        output_test_file = os.path.join(args.output_dir, f"test_results_{task}.txt")

        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info(f"***** Test results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")

    selected_layers, useless_layers, adapters_switches = get_adapters_switches(model)
    summary["task_name"] = args.task_name
    jsonable_args = make_jsonable(args)
    summary["args"] = jsonable_args
    summary["selected_layers"] = selected_layers
    summary["useless_layers"] = useless_layers

    timestamp = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    sum_save_folder = "summaries"
    os.makedirs(sum_save_folder, exist_ok=True)
    filename = f'{args.task_name}_s{args.seed}_{timestamp}.json'
    save_path = f"{sum_save_folder}/{filename}"
    with open(save_path, 'w') as fp:
        json.dump(summary, fp, indent=4, sort_keys=True)
    print(f"Saved a summary in {save_path}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    _main()


if __name__ == "__main__":
    _main()
