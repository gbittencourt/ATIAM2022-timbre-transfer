from torch.utils.data import Dataset, DataLoader
import json
import torchaudio
import os
import torch

class NSynthDataset(Dataset):
    """Use with dataloader to load the NSynth dataset
    args :
        - root_dir [str]: directory where the NSynth folder is.
            The file architercture should be :

            root_dir/
                NSynth/
                    nsynth-test/
                        examples.json
                        audio/
                            "all audio (.wav) used for testing"
                    nsynth-train/
                        examples.json
                        audio/
                            "all audio (.wav) used for training"
                    nsynth-valid/
                        examples.json
                        audio/
                            "all audio (.wav) used for validating"

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
            - vocal
        
        - normalization [float]:
            If None, no normalization is applied
            If float, samples x are normalised by normalisation (x = x/normalization)

            For training, you should run through the dataset once to find the maximum value, the normalise by this value during the training process
            """
    
    def __init__(self, root_dir : str, usage = 'train', transform = None, filter_key = None, normalization = None):
        self.root_dir = root_dir
        train_valid_test = {
            'train' : 'nsynth/nsynth-train',
            'test' :  'nsynth/nsynth-test',
            'valid' : 'nsynth/nsynth-valid',
        }
        
        self.set_dir = os.path.join(self.root_dir, train_valid_test[usage])
        self.audio_dir = os.path.join(self.set_dir, 'audio')
        self.file_names = os.listdir(self.audio_dir)
        self.transform = transform
        self.normalization = normalization
        if filter_key != None:
            self.file_names = list(filter(lambda x: filter_key in x, self.file_names))
        
        self.labels = json.load(open(os.path.join(self.set_dir,'examples.json')))
        
        self.transform = transform
       
    def __len__(self):
        return len(self.file_names)
    

    def __getitem__(self, idx : int):
        audio_path = os.path.join(self.audio_dir, self.file_names[idx])
        label = self.labels[self.file_names[idx][:-4]]['instrument_family']
        
        waveform, samplerate = torchaudio.load(audio_path)
        if self.transform==None:
            output = torch.Tensor(waveform)
        else:
            output = self.transform(torch.Tensor(waveform))
        
        if self.normalization !=None:
            output = output/self.normalization
        return output, label