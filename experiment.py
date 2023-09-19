import os
import yaml
import torch
import logging
import argparse
import numpy as np

import train
#import evaluate

def read_experiment_file(params):
    model_id = params['model_id']

    hyperparams = torch.load('/develop/code/rcdo/hyperparams.pt') # list
    new_params = hyperparams[int(model_id)] #dictionary

    for k in new_params:
        params[k] = new_params[k]

    return params


def begin_experiment(params):
    os.environ['TORCH_HOME'] = params['torch_home']
    train.run(params)
    #evaluate.run(params)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help = "Experiment: Train and Eval LRN Network")
    parser.add_argument("-which", help = "Which dataset to use")
    parser.add_argument("-phase_initialization", help = "Which phase initialization for LRN")
    parser.add_argument("-objective_function_lrn", help = "Which objective function to train the LRN with")
    parser.add_argument("-transfer_learn_lrn", help = "Do you want to load in a pretrained lrn")
    parser.add_argument("-gpu_config", help = "Are you training with GPUs, and if so which ones")
    parser.add_argument("-num_epochs", help = "How many epochs to train for")
    parser.add_argument("-LRN", help = "Do you want to train with the LRN")
    parser.add_argument("-wavelength", help = "Which wavelength to run at?")
    parser.add_argument("-batch_size", help = "Batch size to use")
    parser.add_argument("-learning_rate_lrn", help = "learning rate for the LRN")
    parser.add_argument("-job_id", help = "SLURM job ID")
    parser.add_argument("-data_split", help = "The data split to use")
    parser.add_argument("-model_id", help = "The line from hyperparameters.txt that you want to run")
    parser.add_argument("-distance", help = "Distance of propagation")

    args = parser.parse_args()
    if(args.config == None):
        logging.error("\nAttach Configuration File! Run experiment.py -h\n")
        exit()

    if args.job_id is not None:
        os.environ["SLURM_JOB_ID"] = args.job_id
        logging.debug("Slurm ID : {}".format(os.environ['SLURM_JOB_ID']))

    params = yaml.load(open(args.config), Loader = yaml.FullLoader)
   
    # Overwrite CLI specified parameters - Used for SLURM
    for k in params.keys():
        if k in args.__dict__ and args.__dict__[f'{k}'] is not None:
            params[f'{k}'] = args.__dict__[f'{k}']
            logging.debug("experiment.py | Setting {0} to {1}".format(k, args.__dict__[f'{k}']))

    begin_experiment(params)
