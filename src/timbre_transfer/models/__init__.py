import os
import torch
import torch.nn as nn
import torch.distributions as distrib
import torchvision.transforms as transforms
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")