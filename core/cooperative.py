#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import logging
import torchvision
import torchmetrics
from pytorch_lightning import LightningModule
from torchmetrics.functional import mean_squared_error as mse
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics import ConfusionMatrix, F1Score, Accuracy, Precision, Recall, ConfusionMatrix

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from core.lrn import LRN
from core.datamodule import * 
from core.classifiers import Classifier

#import matplotlib.pyplot as plt
#--------------------------------
# Model: Cooperative Optimization 
#--------------------------------

class CooperativeOptimization(LightningModule):
    def __init__(self, params_model_cooperative, params_model_lrn, params_propagator, 
                       params_modulator, params_model_classifier, all_paths):
        super().__init__()
        self.params = params_model_cooperative.copy()
        self.lrn = LRN(params_model_lrn, all_paths, params_propagator, params_modulator)
        self.classifier = Classifier(params_model_classifier)

        self.all_paths = all_paths
        
        root = all_paths['path_root']
        path_checkpoint_lrn = os.path.join(root,all_paths['path_checkpoint_lrn'])
        path_checkpoint_classifier = os.path.join(root,all_paths['path_checkpoint_classifier'])
        
        self.checkpoint_path = self.params['path_checkpoint_cooperative']

        if params_model_lrn['load_checkpoint_lrn']:
            logging.debug("cooperative | Loading LRN checkpoint")
            self.lrn.load_from_checkpoint(path_checkpoint_lrn,
                                           params_model = params_model_lrn,
                                           all_paths = all_paths,
                                           params_propagator = params_propagator, 
                                           params_modulator = params_modulator,
                                           strict = True)
        if params_model_classifier['load_checkpoint_classifier']:
            logging.debug("cooperative | Loading Classifier checkpoint")
            checkpoint = torch.load(all_paths['path_checkpoint_classifier'], map_location=self.device)
            self.classifier.load_from_checkpoint(path_checkpoint_classifier,
                                                params = params_model_classifier,
                                                strict = True)


        self.num_classes = self.classifier.num_classes

        self.f1 = F1Score(task = 'multiclass', num_classes = self.num_classes, top_k = 1).to(self.device)
        self.acc = Accuracy(task = 'multiclass', num_classes = self.num_classes, top_k = 1).to(self.device)
        self.prec = Precision(task = 'multiclass', num_classes = self.num_classes, top_k = 1).to(self.device)
        self.rec = Recall(task = 'multiclass', num_classes = self.num_classes, top_k = 1).to(self.device)
        self.cfm = ConfusionMatrix(task = 'multiclass', num_classes = self.num_classes, top_k = 1).to(self.device)

        self.lrn_metrics = []
        self.classifier_outputs = []

        self.save_hyperparameters()

    #--------------------------------
    # Initialize: Optimizer
    #--------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.classifier.learning_rate)
        return optimizer

    #--------------------------------
    # Initialize: Optimizer
    #--------------------------------

    def objective(self, lrn_output, prediction, sample, target):
        output_wavefronts, amplitudes, normalized_amplitudes, images, normalized_images, target_ = lrn_output

        loss_lrn = self.lrn.objective(images, target_)

        loss_classifier = self.classifier.objective(prediction, target)

        loss_combined = loss_lrn + loss_classifier
        
        return {'lrn' : loss_lrn, 'classifier' : loss_classifier, 
                'combined' : loss_combined} 
    #--------------------------------
    # Initialize: LRN Metrics
    #--------------------------------
    
    def run_lrn_metrics(self, lrn_outputs):
        wavefronts = lrn_outputs[0]
        amplitudes = lrn_outputs[1] 
        normalized_amplitudes = lrn_outputs[2]
        images = lrn_outputs[3]
        normalized_images = lrn_outputs[4]
        lrn_target = lrn_outputs[5]
        mse_vals = mse(normalized_images.detach(), lrn_target.detach())
        psnr_vals = psnr(normalized_images.detach(), lrn_target.detach())
        ssim_vals = ssim(normalized_images.detach(), lrn_target.detach()).detach()

        return {'mse' : mse_vals.cpu(), 'psnr' : psnr_vals.cpu(), 'ssim' : ssim_vals.cpu()}

    #--------------------------------
    # Initialize: Classifier Metrics
    #--------------------------------
    def run_classifier_metrics(self):
        self.classifier_outputs = torch.tensor(self.classifier_outputs)
        predictions = self.classifier_outputs[:,0].to(self.device)
        targets = self.classifier_outputs[:,1].to(self.device)

        precision = self.prec(preds= predictions, target=targets)
        recall = self.rec(preds = predictions, target=targets)
        f1_score = self.f1(preds = predictions, target=targets)
        accuracy = self.acc(preds = predictions, target=targets)
        confusion_matrix = self.cfm(preds=predictions, target = targets)  
        return {'prec':precision.cpu(), 'rec':recall.cpu(), 'f1':f1_score.cpu(), 'acc':accuracy.cpu(), 'cfm':confusion_matrix.cpu()}

    #--------------------------------
    # Create: Forward Pass
    #--------------------------------
   
    def forward(self, batch, batch_idx):
        #Image the object with our LRN
        lrn_output = self.lrn.shared_step(batch, batch_idx)
        
        output_wavefronts, amplitudes, normalized_amplitudes, images, normalized_images, target_ = lrn_output
        
        #Stack images for the classifer - expects 3 channels
        stacked_image = torch.cat((normalized_images.to(torch.float32), normalized_images.to(torch.float32), normalized_images.to(torch.float32)), 1)
        
        #Classify the image
        prediction = self.classifier(stacked_image)

        return lrn_output, prediction
 
    #--------------------------------
    # Create: Shared Step Train/Valid
    #--------------------------------
      
    def shared_step(self, batch, batch_idx):
        lrn_output, prediction = self(batch, batch_idx)
        return lrn_output,prediction
  
    #--------------------------------
    # Create: Training Step
    #--------------------------------
             
    def training_step(self, batch, batch_idx):
        sample,target = batch

        lrn_output, prediction = self.shared_step(batch, batch_idx)
        loss = self.objective(lrn_output, prediction, sample, target)
        
        self.log("train_loss_lrn", loss['lrn'], on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_loss_classifier", loss['classifier'], on_step = False, on_epoch = True, sync_dist = True)
        self.log("train_loss_combined", loss['combined'], on_step = False, on_epoch = True, sync_dist = True)

        return {'loss':loss['classifier'], 'lrn_output': lrn_output, 'prediction': prediction,
                'sample':sample, 'target': target}
   
    #--------------------------------
    # Create: Validation Step
    #--------------------------------
                
    def validation_step(self, batch, batch_idx):
        sample, target = batch
        lrn_output, prediction = self.shared_step(batch, batch_idx)
        loss = self.objective(lrn_output, prediction, sample, target)

        self.log("val_loss_lrn", loss['lrn'], on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_loss_classifier", loss['classifier'], on_step = False, on_epoch = True, sync_dist = True)
        self.log("val_loss_combined", loss['combined'], on_step = False, on_epoch = True, sync_dist = True)

        return {'loss':loss['classifier'], 'lrn_output': lrn_output, 'prediction': prediction,
                'sample':sample, 'target': target}

    #--------------------------------
    # Create: Test Step
    #--------------------------------
    
    def test_step(self, batch, batch_idx):
        sample, target = batch
        lrn_output, prediction = self.shared_step(batch, batch_idx)
        self.lrn_metrics.append(self.run_lrn_metrics(lrn_output))
        self.classifier_outputs.append((torch.argmax(prediction), target))
        return

    def on_test_end(self):
        classifier_metrics = self.run_classifier_metrics()
    
        #This is a little hacky, but it gets the model id...
        model_id = self.all_paths['path_results_cooperative'].split('/')[-2]
        filename = 'testResults_{}.pt'.format(model_id)
        save_dict = {'classifier' : classifier_metrics, 'lrn' : self.lrn_metrics}
        torch.save(save_dict, os.path.join(self.all_paths['path_root'],self.all_paths['path_results_cooperative'],filename))
        
        return


