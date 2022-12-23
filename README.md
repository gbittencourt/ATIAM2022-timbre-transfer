# Can you speak like a violin ?

## How to install dependencies ?

`pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

***

## How to load the dataset ?

Use `src\timbre_transfer\datasets\NSynthDataset\NSynthDataset.py` :

Example :
```Python
from src.timbre_transfer.datasets.NSynthDataset import NSynthDataset
root_dir = 'data'
train_dataset = NSynthDataset(root_dir = root_dir, usage = 'train', select_class='vocal_acoustic', transform=None)
```

Dataset files hierarchy (here, the root_dir is named "data", as shown on the Python example above) :

- data/
    - nsynth/
        - nsynth-test/
            - examples.json
            - audio/
                - All audio files (.wav) used for testing
        - nsynth-train/
            - examples.json
            - audio/
                - All audio files (.wav) used for training
        - nsynth-valid/
            - examples.json
            - audio/
                - All audio files (.wav) used for validating
***
`source /miniconda/bin/activate`


***

## How to train the model ?

2VAEs : `python train_2VAEs.py`

2VAEs + CC : `python train_2VAEs_CC.py`

2VAEs + CC + GAN : `python train_2VAEs_CC_GAN.py`

***

## How to export some sounds and spectrograms ?

2VAEs : `python exports_2VAE.py`

2VAEs + CC : `python exports_2VAE_CC.py`

2VAEs + CC + GAN : `python exports_2VAE_CC_GAN.py`