import tempfile

import torch

from transformers import MODEL_WITH_HEADS_MAPPING, AutoModelForSequenceClassification, AutoModelWithHeads
from transformers.adapters.composition import BatchSplit, Stack
from transformers.testing_utils import require_torch, torch_device

from .test_adapter_common import create_twin_models


@require_torch
class PredictionHeadModelTestMixin:

    batch_size = 1
    seq_length = 128

    def run_prediction_head_test(
        self, model, compare_model, head_name, input_shape=None, output_shape=(1, 2), label_dict=None
    ):
        # first, check if the head is actually correctly registered as part of the pt module
        self.assertTrue(f"heads.{head_name}" in dict(model.named_modules()))

        # save & reload
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_head(temp_dir, head_name)

            compare_model.load_head(temp_dir)

        # check if adapter was correctly loaded
        self.assertTrue(head_name in compare_model.heads)

        # make a forward pass
        model.active_head = head_name
        input_shape = input_shape or (self.batch_size, self.seq_length)
        in_data = self.get_input_samples(input_shape, config=model.config)
        if label_dict:
            for k, v in label_dict.items():
                in_data[k] = v
        output1 = model(**in_data)
        self.assertEqual(output_shape, tuple(output1[1].size()))
        # check equal output
        compare_model.active_head = head_name
        output2 = compare_model(**in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[1], output2[1]))

    def test_classification_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_classification_head"):
            self.skipTest("No classification head")

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        model1.add_classification_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(model1, model2, "dummy", label_dict=label_dict)

    def test_multiple_choice_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_multiple_choice_head"):
            self.skipTest("No multiple choice head")

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        model1.add_multiple_choice_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.ones(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(
            model1, model2, "dummy", input_shape=(self.batch_size, 2, self.seq_length), label_dict=label_dict
        )

    def test_tagging_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_tagging_head"):
            self.skipTest("No tagging head")

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        model1.add_tagging_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(
            model1, model2, "dummy", output_shape=(1, self.seq_length, 2), label_dict=label_dict
        )

    def test_qa_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_qa_head"):
            self.skipTest("No QA head")

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        model1.add_qa_head("dummy")
        label_dict = {}
        label_dict["start_positions"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        label_dict["end_positions"] = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(
            model1, model2, "dummy", output_shape=(1, self.seq_length), label_dict=label_dict
        )

    def test_causal_or_seq2seq_lm_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_causal_lm_head"):
            if hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_seq2seq_lm_head"):
                seq2seq_head = True
            else:
                self.skipTest("No causal or seq2seq language model head")
        else:
            seq2seq_head = False

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        if seq2seq_head:
            model1.add_seq2seq_lm_head("dummy")
        else:
            model1.add_causal_lm_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(
            model1, model2, "dummy", output_shape=(1, self.seq_length, model1.config.vocab_size), label_dict=label_dict
        )

    def test_masked_lm_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_masked_lm_head"):
            self.skipTest("No causal or seq2seq language model head")

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        model1.add_masked_lm_head("dummy")
        label_dict = {}
        label_dict["labels"] = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        self.run_prediction_head_test(
            model1, model2, "dummy", output_shape=(1, self.seq_length, model1.config.vocab_size), label_dict=label_dict
        )

    def test_dependency_parsing_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_dependency_parsing_head"):
            self.skipTest("No dependency parsing head")

        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        model1.add_dependency_parsing_head("dummy")
        label_dict = {}
        label_dict["labels_arcs"] = torch.zeros(
            (self.batch_size, self.seq_length), dtype=torch.long, device=torch_device
        )
        label_dict["labels_rels"] = torch.zeros(
            (self.batch_size, self.seq_length), dtype=torch.long, device=torch_device
        )
        label_dict["word_starts"] = torch.zeros(
            (self.batch_size, self.seq_length), dtype=torch.long, device=torch_device
        )
        self.run_prediction_head_test(
            model1, model2, "dummy", output_shape=(1, self.seq_length, self.seq_length + 1, 2), label_dict=label_dict
        )

    def test_delete_head(self):
        model = AutoModelWithHeads.from_config(self.config())
        model.eval()

        name = "test_head"
        model.add_classification_head(name)
        self.assertTrue(name in model.heads)
        self.assertTrue(name in model.config.prediction_heads)
        self.assertEqual(name, model.active_head)

        model.delete_head(name)
        self.assertFalse(name in model.heads)
        self.assertFalse(name in model.config.prediction_heads)
        self.assertNotEqual(name, model.active_head)

    def test_adapter_with_head(self):
        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        name = "dummy"
        model1.add_adapter(name)
        model1.add_classification_head(name, num_labels=3)
        model1.set_active_adapters(name)
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter(temp_dir, name)

            model2.load_adapter(temp_dir)
            model2.set_active_adapters(name)
        # check equal output
        in_data = self.get_input_samples((1, 128), config=model1.config)
        output1 = model1(**in_data)
        output2 = model2(**in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))
        self.assertEqual(3, output1[0].size()[1])

    def test_adapter_with_head_load_as(self):
        model1, model2 = create_twin_models(AutoModelWithHeads, self.config)

        name = "dummy"
        model1.add_adapter(name)
        model1.add_classification_head(name, num_labels=3)
        model1.set_active_adapters(name)
        with tempfile.TemporaryDirectory() as temp_dir:
            model1.save_adapter(temp_dir, name)

            # reload using a different name
            model2.load_adapter(temp_dir, load_as="new_name")
            model2.set_active_adapters("new_name")

        # check equal output
        in_data = self.get_input_samples((1, 128), config=model1.config)
        output1 = model1(**in_data)
        output2 = model2(**in_data)
        self.assertEqual(len(output1), len(output2))
        self.assertTrue(torch.equal(output1[0], output2[0]))
        self.assertEqual(3, output1[0].size()[1])

    def test_load_full_model(self):
        model = AutoModelWithHeads.from_config(self.config())
        model.add_classification_head("dummy", layers=1)

        true_config = model.get_prediction_heads_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            # save
            model.save_pretrained(temp_dir)
            # reload
            model = AutoModelWithHeads.from_pretrained(temp_dir)
        self.assertIn("dummy", model.heads)
        self.assertDictEqual(true_config, model.get_prediction_heads_config())

    def test_batch_split_head(self):
        model = AutoModelWithHeads.from_config(self.config())
        model.add_classification_head("a")
        model.add_classification_head("b")
        model.active_head = BatchSplit("a", "b", batch_sizes=[1, 2])
        in_data = self.get_input_samples((3, 128), config=model.config)

        out = model(**in_data)
        self.assertEqual(2, len(out))
        self.assertEqual((1, 2), out[0][0].shape)
        self.assertEqual((2, 2), out[1][0].shape)

    def test_batch_split_adapter_head(self):
        model = AutoModelWithHeads.from_config(self.config())
        model.add_classification_head("a")
        model.add_classification_head("b")
        model.add_adapter("a")
        model.add_adapter("b")
        model.add_adapter("c")
        model.set_active_adapters(BatchSplit(Stack("c", "a"), "b", batch_sizes=[2, 1]))

        in_data = self.get_input_samples((3, 128), config=model.config)
        out = model(**in_data)

        self.assertEqual(2, len(out))
        self.assertTrue(isinstance(model.active_head, BatchSplit))

    def test_reload_static_to_flex_head(self):
        static_head_model = AutoModelForSequenceClassification.from_config(self.config())
        flex_head_model = AutoModelWithHeads.from_pretrained(
            None, config=self.config(), state_dict=static_head_model.state_dict()
        )
        static_head_model.eval()
        flex_head_model.eval()

        static_head_model.add_adapter("test")

        with tempfile.TemporaryDirectory() as temp_dir:
            static_head_model.save_adapter(temp_dir, "test")

            loading_info = {}
            flex_head_model.load_adapter(temp_dir, loading_info=loading_info)

            # Load the adapter a second time to make sure our conversion script doesn't break anything
            flex_head_model.load_adapter(temp_dir, loading_info=loading_info)
        self.assertEqual(0, len(loading_info["missing_keys"]))
        self.assertEqual(0, len(loading_info["unexpected_keys"]))

        # adapter and head were loaded
        self.assertIn("test", flex_head_model.config.adapters)
        self.assertIn("test", flex_head_model.heads)

        # check equal output
        in_data = self.get_input_samples((1, 128), config=flex_head_model.config)
        output1 = static_head_model(**in_data, adapter_names=["test"])
        output2 = flex_head_model(**in_data, adapter_names=["test"], head="test")
        self.assertTrue(torch.all(torch.isclose(output1.logits, output2.logits)))

    def test_invertible_adapter_with_head(self):
        if not hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_masked_lm_head"):
            if hasattr(MODEL_WITH_HEADS_MAPPING[self.config_class], "add_causal_lm_head"):
                causal_lm_head = True
            else:
                self.skipTest("No masked or causel language model head")
        else:
            causal_lm_head = False

        model = AutoModelWithHeads.from_config(self.config())
        model.add_adapter("test", config="pfeiffer+inv")
        if causal_lm_head:
            model.add_causal_lm_head("test")
        else:
            model.add_masked_lm_head("test")
        model.set_active_adapters("test")

        # Set a hook before the invertible adapter to make sure it's actually called twice:
        # Once after the embedding layer and once in the prediction head.
        calls = 0

        def forward_pre_hook(module, input):
            nonlocal calls
            calls += 1

        inv_adapter = model.base_model.get_invertible_adapter()
        self.assertIsNotNone(inv_adapter)
        inv_adapter.register_forward_pre_hook(forward_pre_hook)

        in_data = self.get_input_samples((self.batch_size, self.seq_length), config=model.config)
        out = model(**in_data)

        self.assertEqual((self.batch_size, self.seq_length, model.config.vocab_size), out[0].shape)
        self.assertEqual(2, calls)
