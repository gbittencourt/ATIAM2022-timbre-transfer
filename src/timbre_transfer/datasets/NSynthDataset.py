from torch.utils.data import Dataset, DataLoader
import json
import torchaudio
import os
import torch

class NSynthDataset(Dataset):
    """Use with dataloader to load the NSynth dataset
    args :
        - root_dir [str]: directory where the NSynth dataset is
        - usage [str] : which dataset is required. usages are : 
            - train
            - test
            - valid
        - transform [function] : transforms that are to be applied on the dataset
        - select_class [str] : if you want only one class of sounds. Sound classes are : 
            - bass
            - keyboard
            - brass
            - flute
            - guitar
            - keyboard
            - mallet
            - organ
            - reed
            - string
            - vocal"""
    def __init__(self, root_dir, usage = 'train', transform = None, select_class = None):
        self.root_dir = root_dir
        train_valid_test = {
            'train' : 'nsynth-train',
            'test' :  'nsynth-test',
            'valid' : 'nsynth-valid',
        }
        
        self.set_dir = os.path.join(self.root_dir, train_valid_test[usage])
        self.audio_dir = os.path.join(self.set_dir, 'audio')
        self.file_names = os.listdir(self.audio_dir)
        if select_class != None:
            self.file_names = list(filter(lambda x: select_class in x, self.file_names))
        
        print(self.file_names)

        self.labels = json.load(open(os.path.join(self.set_dir,'examples.json')))
        self.transform = transform
       
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        audio_path = os.path.join(self.audio_dir, self.file_names[idx])
        waveform, samplerate = torchaudio.load(audio_path)
        print(audio_path)
        return waveform