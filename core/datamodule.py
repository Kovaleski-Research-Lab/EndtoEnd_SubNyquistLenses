#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import logging
from typing import Optional
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

import sys
sys.path.append('../')
from core import custom_transforms as ct

import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#--------------------------------
# Initialize: MNIST Wavefront
#--------------------------------

class Wavefront_MNIST_DataModule(LightningDataModule):
    def __init__(self, params, transform = None):
        super().__init__() 
        logging.debug("datamodule.py - Initializing Wavefront_MNIST_DataModule")
        self.params = params.copy()
        self.Nx = self.params['Nxp']
        self.Ny = self.params['Nyp']
        self.n_cpus = self.params['n_cpus']
        self.path_data = self.params['path_data']
        self.path_root = self.params['path_root']
        self.path_data = os.path.join(self.path_root,self.path_data)
        logging.debug("datamodule.py - Setting path_data to {}".format(self.path_data))
        self.batch_size = self.params['batch_size']
        self.data_split = self.params['data_split']

        self.transform = transforms.Compose([
                            transforms.Resize((self.Nx, self.Ny)),
                            ct.Threshold(0.2),
                            ct.WavefrontTransform(self.params['transforms'])])
        
        self.initialize_cpus(self.n_cpus)

    def initialize_cpus(self, n_cpus):
        # Make sure default number of cpus is not more than the system has
        if n_cpus > os.cpu_count():
            n_cpus = 1
        self.n_cpus = n_cpus 
        logging.debug("Wavefront_MNIST_DataModule | Setting CPUS to {}".format(self.n_cpus))

    def prepare_data(self):
        MNIST(self.path_data, train=True, download=False)
        MNIST(self.path_data, train=False, download=False) 

    def setup(self, stage: Optional[str] = None):
        logging.debug("Wavefront_MNIST_DataModule | setup with datasplit = {}".format(self.data_split))
        train_file = f'MNIST/{self.data_split}.split'
        valid_file = f'MNIST/valid.split'
        test_file = 'MNIST/test.split'

        if stage == "fit" or stage is None:
            self.mnist_train = customDataset(torch.load(os.path.join(self.path_data, train_file)), self.transform)
            self.mnist_val = customDataset(torch.load(os.path.join(self.path_data, valid_file)), self.transform)
        if stage == "test" or stage is None:
            self.mnist_test = customDataset(torch.load(os.path.join(self.path_data, test_file)), self.transform)
        if stage == "predict" or stage is None:
            self.mnist_test = customDataset(torch.load(os.path.join(self.path_data, test_file)), self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.n_cpus, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.n_cpus, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=1, num_workers=self.n_cpus, persistent_workers=True, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.n_cpus, persistent_workers=True, shuffle=False)

#--------------------------------
# Initialize: CIFAR Wavefront
#--------------------------------

class customDataset(Dataset):
    def __init__(self, data, transform):
        logging.debug("datamodule.py - Initializing customDataset")
        self.samples, self.targets = data[0], data[1]

        if len(self.samples.shape) < 4:
            self.samples = torch.unsqueeze(self.samples, dim=1)
        
        if self.samples.shape[1] > 3:
            self.samples = torch.swapaxes(self.samples, 1,-1)

        self.transform = transform
        logging.debug("customDataset | Setting transform to {}".format(self.transform))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample,target = self.samples[idx], self.targets[idx]
        sample = self.transform(sample)
        #target = torch.nn.functional.one_hot(torch.tensor(target), num_classes=10)

        return sample,target

#--------------------------------
# Initialize: Select dataset
#--------------------------------

def select_data(params):
    if params['which'] == 'MNIST' :
        return Wavefront_MNIST_DataModule(params) 
    else:
        logging.error("datamodule.py | Dataset {} not implemented!".format(params['which']))
        exit()

#--------------------------------
# Initialize: Testing
#--------------------------------

if __name__=="__main__":
    import yaml
    import torch
    import matplotlib.pyplot as plt
    from pytorch_lightning import seed_everything
    from utils import parameter_manager
    logging.basicConfig(level=logging.DEBUG)
    seed_everything(1337)
    os.environ['SLURM_JOB_ID'] = '0'
    #plt.style.use(['science'])

    #Load config file   
    params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader).copy()
    params['model_id'] = 0

    #Parameter manager
    
    pm = parameter_manager.Parameter_Manager(params=params)

    #Initialize the data module
    dm = select_data(pm.params_datamodule)
    dm.prepare_data()
    dm.setup(stage="fit")

    #View some of the data

    images,labels = next(iter(dm.train_dataloader()))

    from IPython import embed; embed()
    print(images[0])
    print(dm.train_dataloader().__len__())
    print(images.shape)
    print(labels)

    #fig,ax = plt.subplots(1,3,figsize=(5,5))
    #for i,image in enumerate(images):
    #    ax[i].imshow(image.squeeze().abs())
    #    ax[i].axis('off')

    #plt.show()

