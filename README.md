# Adaptable Adapters

Adaptable adapters contain (1) learnable activation functions, i.e., Rational, at each adapter layer, and (2) a learnable switch to select and only use the beneficial adapter layers.

The implementation is based on [AdapterHub](https://adapterhub.ml/) and [Rational Activations](https://github.com/ml-research/rational_activations).


## Project structure

* `adapter-transformers` -- this folder contains the adapter transformer code from AdapterHub that is modified to include rational activation function and Gumbel Softmax 
* `rational_activations` -- this folder contains the code of the rational activation
* `AA-scripts` -- this folder contains the code for running the baseline, AA, and AA-focused experiments




## Installation

First, install the dependencies at `requirements_cu111.txt` or `requirements_cu102.txt` (feel free to use a virtualenv). 
Then, install the adapter-transformers and rational_activations from the local directory.


```bash
pip install --upgrade -r requirements_cu111.txt
pip install -e ./adapter-transformers ./rational_activations
```
_Note:_ Installing `rational_activations` is non-trivial in some cases.

## Running the Experiments 

For training the baseline/AA/AA-focused models you can use the following command
```bash
pyhon AA-scripts/run.py ADAPTER-MODE --task_name TASK-NAME --seed RANDOM-SEED 
                        --model_name_or_path TRANSFORMER-MODEL --layer_num TRANSFORMER-LAYERS 
                        [--adapter_non_linearity ACTIVATION-FUNCTION] [--aa_summary AA-SUMMARY-FILE]
                        [--low_resource AMOUNT-OF-TRAINING-DATA] [--batch_size BATCH-SIZE]
```

* `ADAPTER-MODE`: one of the 'baseline', 'switch', and 'AA-focused' modes
  *   `baseline` adds an adapter layer on top of each of the transformer layers
  *   `switch` builds AA architecture by adding a Gumbel Softmax switch on each transformer layer to select beween the adapter layer and the skip function. The summary of each AA experiment will be saved in a .json file in the summaries directory.
  *   `AA-focused` builds a customized adapter architecture that only adds an adapter layer on the selected layers by an AA that is specified using the --aa_summary argument 

* `TASK-NAME`: one of the 'mnli', 'qqp', 'qnli', 'sst2, 'cola', 'stsb', 'mrpc', 'rte', and 'wnli' values to specify the corresponding GLUE task.
* `RANDOM-SEED`: the random seed to be used for training the model. We have used 42, 92, 111, 245, and 651 random seeds for the experiments of the paper.
* `TRANSFORMER-MODEL`: the base transformer model from the HuggingFace models, e.g., bert-large-uncased or bert-base-uncased.
* `TRANSFORMER-LAYERS`: the number of layers in `TRANSFORMER-MODEL`, e.g., 24 in bert-large-uncased and 12 in bert-base-uncased.
* `ACTIVATION-FUNCTION`: the activation function of adapter layers. You don't need to use this argument for the baseline adapter models. We use 'rational:one' for `AA` ans `AA-focused` experiments. 'rational:one' is a rational activation function that is initialized with f(x) =1. You can initialize rational by any known constant activation, e.g., 'rational:gelu' will initialize rational activations with gelu.
* `AA-SUMMARY-FILE`: The json summary of an AA experiment that is created by one of the `switch` experiments. The 'selected_layers' filed from this json file will be used for adding the adapter layers in AA-focused experiments.
* `AMOUNT-OF-TRAINING-DATA`: The amount of available training data for the low-data settings.
* `BATCH-SIZE`: batch-size for training the model



## Citation
Please use the following citation:

```
@inproceedings{moosavi-etal-2022-adaptable,
    title = "Adaptable Adapters",
    author = "Moosavi, Nafise  and
      Delfosse, Quentin  and
      Kersting, Kristian  and
      Gurevych, Iryna",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.274",
    pages = "3742--3753",
    abstract = "State-of-the-art pretrained NLP models contain a hundred million to trillion parameters. Adapters provide a parameter-efficient alternative for the full finetuning in which we can only finetune lightweight neural network layers on top of pretrained weights. Adapter layers are initialized randomly. However, existing work uses the same adapter architecture{---}i.e., the same adapter layer on top of each layer of the pretrained model{---}for every dataset, regardless of the properties of the dataset or the amount of available training data. In this work, we introduce adaptable adapters that contain (1) learning different activation functions for different layers and different input data, and (2) a learnable switch to select and only use the beneficial adapter layers. We show that adaptable adapters achieve on-par performances with the standard adapter architecture while using a considerably smaller number of adapter layers. In addition, we show that the selected adapter architecture by adaptable adapters transfers well across different data settings and similar tasks. We propose to use adaptable adapters for designing efficient and effective adapter architectures. The resulting adapters (a) contain about 50{\%} of the learning parameters of the standard adapter and are therefore more efficient at training and inference, and require less storage space, and (b) achieve considerably higher performances in low-data settings.",
}
```

