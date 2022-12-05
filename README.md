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
    - NSynth/
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