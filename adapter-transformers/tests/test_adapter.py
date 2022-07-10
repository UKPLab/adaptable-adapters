import random
import unittest

import torch

from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    BartConfig,
    BertConfig,
    DistilBertConfig,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    GPT2Config,
    MBartConfig,
    RobertaConfig,
)
from transformers.testing_utils import require_torch, torch_device

from .test_adapter_common import AdapterModelTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_adapter_training import AdapterTrainingTestMixin


def make_config(config_class, **kwargs):
    return staticmethod(lambda: config_class(**kwargs))


class AdapterTestBase:
    # If not overriden by subclass, AutoModel should be used.
    model_class = AutoModel

    def get_model(self):
        if self.model_class == AutoModel:
            return AutoModel.from_config(self.config())
        else:
            return self.model_class(self.config())

    def get_input_samples(self, shape, vocab_size=5000, config=None):
        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(random.randint(0, vocab_size - 1))
        input_ids = torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()
        # this is needed e.g. for BART
        if config and config.eos_token_id is not None:
            input_ids[input_ids == config.eos_token_id] = random.randint(0, config.eos_token_id - 1)
            input_ids[:, -1] = config.eos_token_id
        in_data = {"input_ids": input_ids}

        if config and config.is_encoder_decoder:
            in_data["decoder_input_ids"] = input_ids.clone()
        return in_data


class BertAdapterTestBase(AdapterTestBase):
    config_class = BertConfig
    config = make_config(
        BertConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    tokenizer_name = "bert-base-uncased"


@require_torch
class BertAdapterTest(
    AdapterModelTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    AdapterTrainingTestMixin,
    ParallelAdapterInferenceTestMixin,
    BertAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class BertClassConversionTest(
    ModelClassConversionTestMixin,
    BertAdapterTestBase,
    unittest.TestCase,
):
    pass


class RobertaAdapterTestBase(AdapterTestBase):
    config_class = RobertaConfig
    config = make_config(
        RobertaConfig,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )


@require_torch
class RobertaAdapterTest(
    AdapterModelTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    RobertaAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class RobertaClassConversionTest(
    ModelClassConversionTestMixin,
    RobertaAdapterTestBase,
    unittest.TestCase,
):
    pass


class DistilBertAdapterTestBase(AdapterTestBase):
    config_class = DistilBertConfig
    config = make_config(
        DistilBertConfig,
        dim=32,
        n_layers=4,
        n_heads=4,
        hidden_dim=37,
    )
    tokenizer_name = "distilbert-base-uncased"


@require_torch
class DistilBertAdapterTest(
    AdapterModelTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    AdapterTrainingTestMixin,
    ParallelAdapterInferenceTestMixin,
    DistilBertAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class DistilBertClassConversionTest(
    ModelClassConversionTestMixin,
    DistilBertAdapterTestBase,
    unittest.TestCase,
):
    pass


class BartAdapterTestBase(AdapterTestBase):
    config_class = BartConfig
    config = make_config(
        BartConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
    )
    tokenizer_name = "facebook/bart-base"


@require_torch
class BartAdapterTest(
    AdapterModelTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    AdapterTrainingTestMixin,
    ParallelAdapterInferenceTestMixin,
    BartAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class BartClassConversionTest(
    ModelClassConversionTestMixin,
    BartAdapterTestBase,
    unittest.TestCase,
):
    pass


class MBartAdapterTestBase(AdapterTestBase):
    config_class = MBartConfig
    config = make_config(
        MBartConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
    )


@require_torch
class MBartAdapterTest(
    AdapterModelTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    MBartAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class MBartClassConversionTest(
    ModelClassConversionTestMixin,
    MBartAdapterTestBase,
    unittest.TestCase,
):
    pass


class GPT2AdapterTestBase(AdapterTestBase):
    config_class = GPT2Config
    config = make_config(
        GPT2Config,
        n_embd=32,
        n_layer=4,
        n_head=4,
        # set pad token to eos token
        pad_token_id=50256,
    )
    tokenizer_name = "gpt2"


@require_torch
class GPT2AdapterTest(
    AdapterModelTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    AdapterTrainingTestMixin,
    ParallelAdapterInferenceTestMixin,
    GPT2AdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class GPT2ClassConversionTest(
    ModelClassConversionTestMixin,
    GPT2AdapterTestBase,
    unittest.TestCase,
):
    pass


class EncoderDecoderAdapterTestBase(AdapterTestBase):
    model_class = EncoderDecoderModel
    config_class = EncoderDecoderConfig
    config = staticmethod(
        lambda: EncoderDecoderConfig.from_encoder_decoder_configs(
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
            ),
            BertConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=37,
                is_decoder=True,
                add_cross_attention=True,
            ),
        )
    )
    tokenizer_name = "bert-base-uncased"


@require_torch
class EncoderDecoderAdapterTest(
    AdapterModelTestMixin,
    AdapterFusionModelTestMixin,
    EncoderDecoderAdapterTestBase,
    unittest.TestCase,
):
    def test_invertible_adapter_with_head(self):
        """This test class is copied and adapted from the identically-named test in test_adapter_heads.py."""
        model = AutoModelForSeq2SeqLM.from_config(self.config())
        model.add_adapter("test", config="pfeiffer+inv")
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

        in_data = self.get_input_samples((1, 128), config=model.config)
        out = model(**in_data)

        self.assertEqual((1, 128, model.config.decoder.vocab_size), out[0].shape)
        self.assertEqual(2, calls)
