#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule
from torchvision.models import resnet50, resnet18, resnet34
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights

from torchmetrics import ConfusionMatrix, F1Score, Accuracy, Precision, Recall


class Classifier(LightningModule):
    def __init__(self, params_classifier):
        super().__init__()
 
        self.params = params_classifier
        self.name = self.params['name_classifier']
        self.num_classes = int(self.params['num_classes'])
        self.learning_rate = self.params['learning_rate']
        self.transfer_learn = self.params['transfer_learn']
        self.freeze_backbone = self.params['freeze_backbone']

        self.select_model()

        self.f1 = F1Score(task = 'multiclass', num_classes = self.num_classes, top_k = 1)
        self.acc = Accuracy(task = 'multiclass', num_classes = self.num_classes, top_k = 1)
        self.prec = Precision(task = 'multiclass', num_classes = self.num_classes, top_k = 1)
        self.rec = Recall(task = 'multiclass', num_classes = self.num_classes, top_k = 1)
        
        self.checkpoint_path = self.params['path_checkpoint_classifier']

        self.save_hyperparameters()

    def select_model(self):
        if self.name == 'resnet50':
            if self.transfer_learn:
                backbone = resnet50(weights = ResNet50_Weights.DEFAULT) 
            else:
                backbone = resnet50()

        elif self.name == 'resnet34':
            if self.transfer_learn:
                backbone = resnet34(weights = ResNet34_Weights.DEFAULT)
            else:
                backbone = resnet34()

        elif self.name == 'resnet18':
            if self.transfer_learn:
                backbone = resnet18(weights = ResNet18_Weights.DEFAULT)
            else:
                backbone = resnet18()

        if self.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters,  self.num_classes)

    def objective(self, output, target):
        target = torch.nn.functional.one_hot(target, self.num_classes).double()
        loss = nn.functional.cross_entropy(output, target)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
 
    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def shared_step(self, batch, batch_idx):
        sample, target = batch
        if sample.is_complex():
            sample = sample.abs()**2 
        
        sample = torch.cat((sample,sample,sample), 1)
        prediction = self(sample)

        return prediction, target

    def training_step(self, batch, batch_idx):
        output, target = self.shared_step(batch, batch_idx)
        loss = self.objective(output, target)
        self.log("train_loss", loss, on_step = False, on_epoch = True, sync_dist = True)
        return {'loss' : loss, 'output' : output, 'target' : target}

    def validation_step(self, batch, batch_idx):
        output, target = self.shared_step(batch, batch_idx)
        loss = self.objective(output, target)
        self.log("val_loss", loss, on_step = False, on_epoch = True, sync_dist = True)
        return {'loss' : loss, 'output' : output, 'target' : target}        


if __name__ == "__main__":
    import torch
    import yaml

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer, seed_everything
    from datamodule import Wavefront_MNIST_DataModule
    params = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)

    # Load: Paths 
    root = params['path_root']
    path_data = params['path_data']
    path_model = params['path_model']
    path_train = params['path_train']
    path_valid = params['path_valid']
    path_results = params['path_results']
    path_checkpoint = params['path_checkpoint']
 
    # Load: Trainer Params
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    valid_rate = params['valid_rate']
    accelerator = params['accelerator']
    learning_rate = params['learning_rate']
    gpu_flag, gpu_list = params['gpu_config']

    # Load: Model Params
    optimizer = params['optimizer']
    num_layers = params['num_layers']
    transfer_learn = params['transfer_learn']
    objective_function = params['objective_function']

    # Load: Datamodule Params
    which = params['which']
    n_cpus = params['n_cpus']
    data_split = params['data_split']
    transforms = params['wavefront_transform']
    
    # Load: Physical Params
    distance = params['distance']
    wavelength = params['wavelength']
    # Propagator
    Nxp = params['Nxp']
    Nyp = params['Nyp']
    Lxp = params['Lxp']
    Lyp = params['Lyp']
    #Modulator
    Nxm = params['Nxm']
    Nym = params['Nym']
    Lxm = params['Lxm']
    Lym = params['Lym']
    modulator_type = params['modulator_type'] 
    phase_initialization = params['phase_initialization']
    amplitude_initialization = params['amplitude_initialization']

    # Initalize: Global seeding
    seed_flag, seed_value = params['seed']
    if(seed_flag):
        seed_everything(seed_value, workers = True)



    # Collect: Datamodule params    
    params_datamodule = {'batch_size' : batch_size, 'data_split' : data_split,
                         'path_root' : root, 'path_data' : path_data, 'which' : which,
                         'transforms' : transforms,'Nxp' : Nxp, 'Nyp' : Nyp, 'n_cpus' : n_cpus}

    # Collect: Trainer params
    params_trainer = {'batch_size' : batch_size, 'num_epochs' : num_epochs, 
                      'learning_rate' : learning_rate, 'accelerator' : accelerator,
                      'valid_rate' : valid_rate, 'transfer_learn' : transfer_learn}


    #Collect: Classifer params 
    params_classifier = {'batch_size' : batch_size, 'learning_rate' : learning_rate}    


    logger = Logger(params_logger)
    model = Classifier(params_classifier)
    data = Wavefront_MNIST_DataModule(params_datamodule) 

    checkpoint_callback = ModelCheckpoint(dirpath = os.path.join(root,path_model))
    trainer = Trainer(logger = logger, accelerator = "cpu", max_epochs = num_epochs, 
                          num_sanity_val_steps = 0, default_root_dir = path_results, 
                          check_val_every_n_epoch = valid_rate, callbacks = [checkpoint_callback])


