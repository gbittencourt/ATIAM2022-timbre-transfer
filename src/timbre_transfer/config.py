import yaml
from pydantic import BaseModel
from typing import List


def load_yaml(yaml_filepath):
    with open(yaml_filepath, 'r') as s:
        try:
            parsed_yaml = yaml.safe_load(stream=s)
            return parsed_yaml
        except yaml.YAMLError as e:
            print(e)


class Params(BaseModel):
    # data params
    data_dir: str = None
    out_dir: str = None
    n_signal: int = 2 ** 18
    truncate: bool = False
    batch: int = 8
    n_workers: int = 2

    # signal features
    sr: int = 44100
    n_fft: int = 1024
    hop_length: int = 256
    type_transform: str = "dgt"  # "stft" or "dgt"
    inversion_mode: str = "pghi"  # "keep_input" or "random" or "griffin_lim" or "pghi"

    # model params
    latent_size: int = 16
    hidden_dim: int = 32
    ratios: List[int] = [4, 2]
    lr: float = 1e-4
    beta: float = 1.
    warmup: int = 10000
    cached: bool = False

    # train params
    models_dir: str = None
    name: str = "vae"
    max_steps: int = 10000
    val_steps: int = 1000
    n_ex: int = 2
    ckpt: str = None


def load_config(filepath):
    config = load_yaml(filepath)
    params = Params(**config)
    return params