import torch
import torch.nn as nn
import numpy as np
import torchaudio.transforms as ta
import torchaudio.functional as tf
import torchaudio

class AudioTransform(torch.nn.Module):
    def __init__(self,input_freq = 16000, n_fft=1024,n_mel=128,stretch_factor=0.8):
        super().__init__()
        self.spec = ta.Spectrogram(n_fft=n_fft, power=2)

        self.spec_aug = torch.nn.Sequential(
            ta.TimeStretch(stretch_factor, fixed_rate=True),
            ta.FrequencyMasking(freq_mask_param=80),
            ta.TimeMasking(time_mask_param=80),
        )

        self.mel_scale = ta.MelScale(
            n_mels=n_mel, sample_rate=input_freq, n_stft=n_fft // 2 + 1)
        
        self.inverse_melscale = ta.InverseMelScale(sample_rate=input_freq, n_stft=n_fft // 2 + 1,n_mels=n_mel)
        
        self.griffin = ta.GriffinLim(n_fft=n_fft)
        
        #self.mel_spec = ta.MelSpectrogram(sample_rate = input_freq,n_fft=n_fft,n_mels=64)
        
    def forward(self, wav: torch.Tensor): #-> torch.Tensor:

        # Convert to power spectrogram
        spec = self.spec(wav)

        # Convert to mel-scale
        mel = self.mel_scale(spec)
        
        mel = np.log(1+mel)

        s = mel.size()
        #s[0] : 1
        #s[1] : mels
        #s[2] : temporel
        if s[2]!=s[1]:
            n = s[2]//s[1]+1
            if n ==1:
                new_mel = torch.zeros((s[0], s[1], s[1]))
                new_mel[:,:,:s[2]] = mel
            else:
                new_mel = torch.zeros((n, s[0], s[1], s[1]))
                for i in range(n-1):
                    new_mel[i,:,:,:] = mel[:,:,i*s[1]:(i+1)*s[1]]
                new_mel[n-1,:,:,:s[2]-(n-1)*s[1]] = mel[:,:,(n-1)*s[1]:]
            mel = new_mel
        return mel

    def inverse(self,mel : torch.Tensor):
        inv = np.exp(mel)-1
        inv = self.inverse_melscale(inv)
        inv = self.griffin(inv)
        
        return inv