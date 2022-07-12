import json
import sys
import os
import argparse
import time
import logging
import wandb
from run_glue import main as main_glue
from run_drop_glue import main as main_drop_glue
from args import GLUEArgs


os.environ["WANDB_DISABLED"] = "true"


logger = logging.getLogger(__name__)


PER_TASK = {
    "rte": ("RTE", "accuracy", "acc", "Accuracy"),
    "wnli": ("WNLI", "accuracy", "acc", "Accuracy"),
    "mrpc": ("MRPC", "f1", "f1", "F1 Score"),
    "qqp": ("QQP", "f1", "f1", "F1 Score"),
    "sst2": ("SST-2", "accuracy", "acc", "Accuracy"),
    "qnli": ("QNLI", "accuracy", "acc", "Accuracy"),
    "mnli": ("MNLI", "accuracy", "acc", "Accuracy"),
    "cola": ("CoLA", "matthews_correlation", "mat_cor", "Matthews Correlation"),
    "stsb": ("STS-B", "spearmanr", "spearmanr", "Spearmanr"),
}



def _get_last_notna_value(df):
    df = df[df.notna()]
    return df.iloc[-1]


# def _get_wandb_api(timeout=11):
#     return wandb.Api(timeout=timeout)


def get_selected_layers(path):
    jdata = json.load(open(path))
    print(f"Loaded data from {path}")
    selected_layers = jdata["best_model"]["selected_layers"]
    print(f"Using selected layers: {selected_layers}")
    return selected_layers


def default_args(args):
    params = {
        'model_name_or_path': args.model_name_or_path,
        'layer_num': args.layer_num,
        'task_name': args.task_name,
        'train_adapter': True,
        'do_train': True,
        'do_eval': True,
        'do_predict': True,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'output_dir': f'output/{args.task_name}_{args.seed}_{int(time.time() * 1e7)}',
        'overwrite_output_dir': True,
        'seed': args.seed,
        'logging_steps': 20,
        'low_resources': args.low_resources,
        'save_total_limit': 2,
        'evaluation_strategy': 'epoch',
        'learning_rate': 1e-4,
        'num_train_epochs': 20,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_accuracy',
        'lr_for_switches': args.lr_for_switches,
        'lr_for_rational_activations': 0.01,
        'switch_inputs': ['minimal:identity', 'pfeiffer:rational:one']
    }

    if hasattr(args, 'adapter_non_linearity'):
        params['switch_inputs'] =  [
            'minimal:identity',
            f'pfeiffer:{args.adapter_non_linearity}'
        ]

    return params


def switches(args):
    glue_args = GLUEArgs(
        baseline=False,
        use_switches=True,
        adapter_drop_skip_connections=True,
        **default_args(args)
    )
    main_glue(glue_args)


def AA_focused(args):
    adapters_at = get_selected_layers(args.aa_summary)

    glue_args = GLUEArgs(
        baseline=False,
        use_switches=False,
        adapters_at=adapters_at,
        **default_args(args)
    )
    main_glue(glue_args)


def baseline(args):
    glue_args = GLUEArgs(
        baseline=True,
        baseline_bert=args.bert_only,
        baseline_leave_out_all=args.leave_out_all,
        **default_args(args)
    )
    main_glue(glue_args)


def drop(args):
    glue_args = GLUEArgs(
            baseline=True,
            **default_args(args)
        )
    main_drop_glue(glue_args)



class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if envvar in os.environ:
            default = os.environ[envvar]
        if required and default:
            required = False
        super().__init__(default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


def _add_id_arguments(parser, single=True):
    if single:
        parser.add_argument('--task_name', type=str, required=True)
        parser.add_argument('--seed', type=int, required=True)
    else:
        parser.add_argument('--seed', type=int, nargs='+')
        parser.add_argument('--task_name', type=str, nargs='+')
    parser.add_argument('--low_resources', type=int, default=None)
    parser.add_argument('--lr_for_switches', type=float, default=0.05)


def _add_train_arguments(parser):
    parser.add_argument(
        '--layer_num',
        type=int,
        default=24,
        help="The number of layers in the specified model in model_name_or_path"
    )

    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default="bert-large-uncased",
        help= "Path to pretrained model or model identifier "
            "from huggingface.co/models",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Set the 'per_device_{train,eval}_batch_size' argument."
    )


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
        force=True
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wandb_project',
        action=EnvDefault,
        envvar='WANDB_PROJECT',
        default="huggingface"
    )

    # We need some subparsers.
    subparsers = parser.add_subparsers(required=True)

    # Baseline.
    baseline_p = subparsers.add_parser('baseline')
    _add_id_arguments(baseline_p)
    _add_train_arguments(baseline_p)
    baseline_p.add_argument('--bert_only', action='store_true')
    baseline_p.add_argument('--leave_out_all', action='store_true')
    baseline_p.set_defaults(func=baseline)
    baseline_p.add_argument('--adapter_non_linearity', type=str, default='relu')


    # Drop.
    dbaseline_p = subparsers.add_parser('drop')
    _add_id_arguments(dbaseline_p)
    _add_train_arguments(dbaseline_p)
    dbaseline_p.set_defaults(func=drop)

    # Switches.
    switches_p = subparsers.add_parser('switch')
    _add_id_arguments(switches_p)
    _add_train_arguments(switches_p)
    switches_p.add_argument('--adapter_non_linearity', type=str, default='rational:one')
    switches_p.set_defaults(func=switches)


    #AA-focused adapter architecture.
    AA_focused_p = subparsers.add_parser('AA-focused')
    _add_id_arguments(AA_focused_p)
    _add_train_arguments(AA_focused_p)
    AA_focused_p.add_argument('--adapter_non_linearity', type=str, default='rational:one')
    AA_focused_p.add_argument('--aa_summary', help="path to summary file from which to get selected layers", type=str, required=True)
    AA_focused_p.set_defaults(func=AA_focused)

    # Parse and call the right function.
    try:
        args = parser.parse_args()
    except TypeError:
        parser.print_help()
        exit(-1)

    args.func(args)
