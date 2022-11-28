# Author : Come Peladeau

from torch.utils.data import Dataset, DataLoader
import json
import torchaudio
import os
import torch

class NSynthDataset(Dataset):
    def __init__(self, root_dir, usage = 'train', transform = None):
        self.root_dir = root_dir
        train_valid_test = {
            'train' : 'nsynth-train',
            'test' : 'nsynth-test',
            'valid' : 'nsynth-valid',
        }
        
        self.set_dir = os.path.join(self.root_dir, train_valid_test[usage])
        self.audio_dir = os.path.join(self.set_dir, 'audio')
        self.file_names = os.listdir(self.audio_dir)

        self.labels = json.load(open(os.path.join(self.set_dir,'examples.json')))
        self.transform = transform
       
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.file_names[idx])
        waveform, samplerate = torchaudio.load(audio_path)
        if waveform.size()!=torch.Size([1, 64000]):
            print(waveform.size())
        label = self.labels[self.file_names[idx][:-4]]
        
        print(label)
        return waveform, label['instrument_family']