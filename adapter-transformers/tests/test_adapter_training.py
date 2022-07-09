import copy

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelWithHeads,
    AutoTokenizer,
    GlueDataset,
    GlueDataTrainingArguments,
    Trainer,
    TrainingArguments,
)
from transformers.adapters.composition import Fuse
from transformers.testing_utils import require_torch


def filter_parameters(model, filter_string):
    return {k: v for (k, v) in model.named_parameters() if filter_string in k}


@require_torch
class AdapterTrainingTestMixin:
    def test_train_single_adapter(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelWithHeads.from_config(self.config())

        # add two adapters: one will be trained and the other should be frozen
        model.add_adapter("mrpc")
        model.add_adapter("dummy")
        model.add_classification_head("mrpc")

        self.assertIn("mrpc", model.config.adapters.adapters)
        self.assertIn("dummy", model.config.adapters.adapters)

        # train the mrpc adapter -> should be activated & unfreezed
        model.train_adapter("mrpc")
        self.assertEqual(set(["mrpc"]), model.active_adapters.flatten())

        # all weights of the adapter should be activated
        for k, v in filter_parameters(model, "adapters.mrpc.").items():
            self.assertTrue(v.requires_grad, k)
        # all weights of the adapter not used for training should be freezed
        for k, v in filter_parameters(model, "adapters.dummy.").items():
            self.assertFalse(v.requires_grad, k)
        # weights of the model should be freezed (check on some examples)
        for k, v in filter_parameters(model, "encoder.layer.0.attention").items():
            self.assertFalse(v.requires_grad, k)

        state_dict_pre = copy.deepcopy(model.state_dict())

        # setup dataset
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        training_args = TrainingArguments(
            output_dir="./examples", do_train=True, learning_rate=0.1, max_steps=7, no_cuda=True
        )

        # evaluate
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

        for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items()):
            if "mrpc" in k1:
                self.assertFalse(torch.equal(v1, v2))
            else:
                self.assertTrue(torch.equal(v1, v2))

    def test_train_adapter_fusion(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_config(self.config())

        # add the adapters to be fused
        model.add_adapter("a")
        model.add_adapter("b")
        model.add_adapter("c")

        self.assertIn("a", model.config.adapters.adapters)
        self.assertIn("b", model.config.adapters.adapters)
        self.assertIn("c", model.config.adapters.adapters)

        # setup fusion
        adapter_setup = Fuse("a", "b", "c")
        model.add_adapter_fusion(adapter_setup)
        model.train_adapter_fusion(adapter_setup)
        model.set_active_adapters(adapter_setup)
        self.assertEqual(adapter_setup, model.active_adapters)

        # all weights of the adapters should be frozen (test for one)
        for k, v in filter_parameters(model, "adapters.a.").items():
            self.assertFalse(v.requires_grad, k)
        # all weights of the fusion layer should be activated
        for k, v in filter_parameters(model, "adapter_fusion_layer").items():
            self.assertTrue(v.requires_grad, k)
        # weights of the model should be freezed (check on some examples)
        for k, v in filter_parameters(model, "encoder.layer.0.attention").items():
            self.assertFalse(v.requires_grad, k)

        state_dict_pre = copy.deepcopy(model.state_dict())

        # Since our config has a value matrix, make sure it is regularized.
        # We do this by patching the fusion regularization function.
        regularization_called = False
        orig_fusion_regularization_loss = model.base_model.get_fusion_regularization_loss

        def patched_fusion_reg_loss():
            nonlocal regularization_called
            regularization_called = True
            return orig_fusion_regularization_loss()

        model.base_model.get_fusion_regularization_loss = patched_fusion_reg_loss

        # setup dataset
        data_args = GlueDataTrainingArguments(
            task_name="mrpc", data_dir="./tests/fixtures/tests_samples/MRPC", overwrite_cache=True
        )
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
        training_args = TrainingArguments(
            output_dir="./examples", do_train=True, learning_rate=0.1, max_steps=7, no_cuda=True
        )

        # evaluate
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

        for ((k1, v1), (k2, v2)) in zip(state_dict_pre.items(), model.state_dict().items()):
            if "adapter_fusion_layer" in k1 or "classifier" in k1 or "classification_head" in k1 or "score" in k1:
                self.assertFalse(torch.equal(v1, v2), k1)
            else:
                self.assertTrue(torch.equal(v1, v2), k1)

        self.assertTrue(regularization_called)
