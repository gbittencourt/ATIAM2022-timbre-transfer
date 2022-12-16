import os
import click
import torch
import argparse


from timbre_transfer.config import load_config


# set GPU usage
CUDA_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES", None)

use_gpu = 1 if (CUDA_VISIBLE and int(CUDA_VISIBLE) >= 0) or torch.cuda.is_available() else 0
if CUDA_VISIBLE and int(CUDA_VISIBLE) == -1:
    use_gpu = 0

device = torch.device("cuda" if use_gpu else "cpu")


@click.argument("config_filepath")
@click.option("--verbose", "-v", is_flag=True)
def train(config_filepath, verbose):

    if verbose:
        if use_gpu:
            print(f"Using GPU: {CUDA_VISIBLE}.")
        print(f"Loading config file: {config_filepath}")
    
    config = load_config(config_filepath)


