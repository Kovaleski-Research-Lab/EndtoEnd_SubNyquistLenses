#--------------------------------
# Import: Python libraries
#--------------------------------

import torch
import logging

#--------------------------------
# Initialize: Wavefront transform
#--------------------------------

class WavefrontTransform(object):
    def __init__(self, params):
        self.params = params.copy()
        logging.debug("custom_transforms.py - Initializing WavefrontTransform")

        # Set initialization strategy for the wavefront
        self.phase_initialization_strategy = params['phase_initialization_strategy']

        if self.phase_initialization_strategy == 0:
            logging.debug("custom_transforms.py | WavefrontTransform | Phase Initialization : Phase = torch.ones(), Amplitude = Sample")
        else:
            logging.debug("custom_transforms.py | WavefrontTransform | Phase Initialization : Phase = Sample, Amplitude = torch.ones()")

    def __call__(self,sample):
        c,w,h = sample.shape 
        if self.phase_initialization_strategy == 0:
            phases = torch.ones(c,w,h)
            amplitude = sample
        else:
            phases = sample
            amplitude = torch.ones(c,w,h)

        return amplitude * torch.exp(1j*phases)

class Normalize(object):                                                                    
    def __init__(self, params):                                                             
        self.params = params.copy()                                                         
        logging.debug("custom_transforms.py - Initializing Normalize")
                                                                                            
    def __call__(self,sample):                                                              
                                                                                            
        min_val = torch.min(sample)                                                         
        sample = sample - min_val                                                           
        max_val = torch.max(sample)                                                         
                                                                                            
        return sample / max_val 

#--------------------------------
# Initialize: Threshold transform
#--------------------------------

class Threshold(object):
    def __init__(self, threshold):
        logging.debug("custom_transforms.py - Initializing Threshold")
        self.threshold = threshold
        logging.debug("custom_transforms.py | Threshold | Setting threshold to {}".format(self.threshold))

    def __call__(self, sample):
        return (sample > self.threshold)
