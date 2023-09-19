import logging
import torch
import yaml
import traceback
import sys
import os
import numpy as np
class Parameter_Manager():
    def __init__(self, config = None,  params = None):
        logging.debug("parameter_manager.py - Initializing Parameter_Manager")

        if config is not None:
            self.open_config(config)
        if params is not None:
            self.params = params

        self.parse_params(self.params)

    def open_config(self, config_file):
        try:
            with open(config_file) as c:
                self.params = yaml.load(c, Loader = yaml.FullLoader)
        except Exception as e:
            logging.error(e)
            sys.exit()
            
    def parse_params(self, params):
        try:
            
            logging.debug("Parameter_Manager | parsing parameters")
                

            # Load: Paths 
            self.path_root = params['path_root']
            self.path_data = params['path_data']
            self.path_model = params['path_model']
            self.path_train = params['path_train']
            self.path_valid = params['path_valid']
            self.path_results = params['path_results']
            self._path_checkpoint_lrn = params['path_checkpoint_lrn']
 
            # Load: Trainer Params
            self.batch_size = params['batch_size']
            self.num_epochs = params['num_epochs']
            self.valid_rate = params['valid_rate']
            self.accelerator = params['accelerator']
            self.gpu_flag, self.gpu_list = params['gpu_config']

            # Load: LRN Model Params
            self.optimizer_lrn = params['optimizer_lrn']
            self.num_layers_lrn = params['num_layers_lrn']
            self.learning_rate_lrn = params['learning_rate_lrn']
            self.transfer_learn_lrn = params['transfer_learn_lrn']
            self.load_checkpoint_lrn = params['load_checkpoint_lrn']
            self.objective_function_lrn = params['objective_function_lrn']
            self.subset_lrn= params['subset_lrn']

            # Load: Classifier Model Params
            self.num_classes = params['num_classes']
            self.name_classifier = params['name_classifier']
            self.freeze_backbone = params['freeze_backbone']
            self.optimizer_classifier = params['optimizer_classifier']
            self.num_layers_classifier = params['num_layers_classifier']
            self.learning_rate_classifier = params['learning_rate_classifier']
            self.transfer_learn_classifier = params['transfer_learn_classifier']
            self.load_checkpoint_classifier = params['load_checkpoint_classifier']
            self.objective_function_classifier = params['objective_function_classifier']
            self.subset_classifier = params['subset_classifier']

            # Load: Cooperative Model Params
            self.load_checkpoint_cooperative = params['load_checkpoint_cooperative']
            self.subset_cooperative = params['subset_cooperative']

            # Load: Datamodule Params
            self._which = params['which']
            self.n_cpus = params['n_cpus']
            self._data_split = params['data_split']
            self.transforms = params['wavefront_transform']
            
            # Load: Physical Params
            self._distance = params['distance']
            if(not(isinstance(self._distance, torch.Tensor))):
                self._distance = torch.tensor(float(self._distance))
            self._wavelength = torch.tensor(float(params['wavelength']))

            # Propagator
            self.Nxp = params['Nxp']
            self.Nyp = params['Nyp']
            self.Lxp = params['Lxp']
            self.Lyp = params['Lyp']
            self._adaptive = params['adaptive']

            #Modulator
            self.Nxm = params['Nxm']
            self.Nym = params['Nym']
            self.Lxm = params['Lxm']
            self.Lym = params['Lym']
            self.modulator_type = params['modulator_type'] 
            self._phase_initialization = params['phase_initialization']
            self.amplitude_initialization = params['amplitude_initialization']

            # Determine the type of experiment we are running
            self.lrn = params['lrn']
            self.classifier = params['classifier']
            self.model_id = params['model_id']

            try:
                self.jobid = os.environ['SLURM_JOB_ID']
            except:
                self.jobid = 0

            #Checkpoint and results paths

            ######################################################################
            # IF SUBSETS GET UPDATED, THE CHECKPOINTS WILL ALSO NEED TO BE UPDATED
            ######################################################################

            checkpoint_distance = self.model_id.split('_')[-1]

            self.path_model_classifier = f"{self.path_model}/classifier/{self.model_id}/"
            self.path_results_classifier = f"{self.path_results}/classifier/{self.model_id}/"
            self.path_checkpoint_classifier = f"{self.path_model}/classifier/{self.subset_classifier}/{self.model_id}/"

            self.path_model_cooperative = f"{self.path_model}/cooperative/{self.model_id}/"
            self.path_results_cooperative = f"{self.path_results}/cooperative/{self.model_id}/"
            self.path_checkpoint_cooperative = f"{self.path_model}/cooperative/{self.subset_cooperative}/{self.model_id}/"

            self.path_model_lrn = f"{self.path_model}/lrn/{self.model_id}/"
            self.path_results_lrn = f"{self.path_results}/lrn/{self.model_id}/"
            self.path_checkpoint_lrn = f"{self.path_model}/lrn/{self.subset_lrn}/lrn_randomInit_{checkpoint_distance}/epoch=4-step=6250.ckpt"
            logging.debug(f"parameter_manager | path_checkpoint_lrn : {self.path_checkpoint_lrn}")


            # Seeding
            self.seed_flag, self.seed_value = params['seed']

            # Collect the parameters for the different modules
            self.collect_params()

        except Exception as e:

            logging.error(e)
            traceback.print_exc()
            sys.exit()

    def collect_params(self):
        logging.debug("Parameter_Manager | collecting parameters")
        self._params_model_lrn = {
                                'optimizer'             : self.optimizer_lrn,
                                'num_layers'            : self.num_layers_lrn, 
                                'learning_rate'         : self.learning_rate_lrn,
                                'transfer_learn'        : self.transfer_learn_lrn, 
                                'objective_function'    : self.objective_function_lrn,
                                'load_checkpoint_lrn'   : self.load_checkpoint_lrn,
                                'path_checkpoint_lrn'   : self.path_checkpoint_lrn,
                                'path_results_lrn'      : self.path_results_lrn,
                                'subset_lrn'            : self.subset_lrn,
                                }

        self._params_model_classifier = {
                                'num_classes'                   : self.num_classes, 
                                'optimizer'                     : self.optimizer_classifier,
                                'freeze_backbone'               : self.freeze_backbone,
                                'name_classifier'               : self.name_classifier, 
                                'num_layers'                    : self.num_layers_classifier, 
                                'learning_rate'                 : self.learning_rate_classifier, 
                                'transfer_learn'                : self.transfer_learn_classifier, 
                                'objective_function'            : self.objective_function_classifier,
                                'load_checkpoint_classifier'    : self.load_checkpoint_classifier,
                                'path_checkpoint_classifier'    : self.path_checkpoint_classifier,
                                'path_results_classifier'       : self.path_results_classifier,
                                'subset_classifier'             : self.subset_classifier,
                                }

        self._params_model_cooperative = {
                                'load_checkpoint_cooperative'   : self.load_checkpoint_cooperative,
                                'path_checkpoint_cooperative'   : self.path_checkpoint_cooperative,
                                'path_results_cooperative'      : self.path_results_cooperative, 
                                'subset_cooperative'            : self.subset_cooperative,
                                }
        
             
        self._params_propagator = {
                                'Nxp'           : self.Nxp, 
                                'Nyp'           : self.Nyp, 
                                'Lxp'           : self.Lxp, 
                                'Lyp'           : self.Lyp,
                                'distance'      : self._distance,
                                'adaptive'      : self.adaptive,
                                'batch_size'    : self.batch_size,
                                'wavelength'    : self._wavelength, 
                                }
                
        self._params_modulator = {
                                'Nxm'                       : self.Nxm, 
                                'Nym'                       : self.Nym, 
                                'Lxm'                       : self.Lxm, 
                                'Lym'                       : self.Lym,
                                'wavelength'                : self._wavelength, 
                                'modulator_type'            : self.modulator_type,
                                'phase_initialization'      : self._phase_initialization,
                                'amplitude_initialization'  : self.amplitude_initialization,
                                }

        self._params_datamodule = {
                                'Nxp'           : self.Nxp, 
                                'Nyp'           : self.Nyp, 
                                'which'         : self._which,
                                'n_cpus'        : self.n_cpus,
                                'path_root'     : self.path_root, 
                                'path_data'     : self.path_data, 
                                'batch_size'    : self.batch_size, 
                                'data_split'    : self.data_split,
                                'transforms'    : self.transforms,
                                }


        self._params_trainer = {
                            'num_epochs'    : self.num_epochs, 
                            'valid_rate'    : self.valid_rate,
                            'accelerator'   : self.accelerator, 
                            }
 

        self._all_paths = {
                        'path_root'                     : self.path_root, 
                        'path_data'                     : self.path_data, 
                        'path_model'                    : self.path_model,
                        'path_train'                    : self.path_train, 
                        'path_valid'                    : self.path_valid,
                        'path_results'                  : self.path_results, 
                        'path_model_lrn'                : self.path_model_lrn, 
                        'path_results_lrn'              : self.path_results_lrn, 
                        'path_checkpoint_lrn'           : self._path_checkpoint_lrn,
                        'path_model_classifier'         : self.path_model_classifier,
                        'path_model_cooperative'        : self.path_model_cooperative, 
                        'path_results_classifier'       : self.path_results_classifier,
                        'path_results_cooperative'      : self.path_results_cooperative,
                        'path_checkpoint_classifier'    : self.path_checkpoint_classifier,
                        }


    @property 
    def params_model_lrn(self):         
        return self._params_model_lrn

    @property
    def params_model_classifier(self):
        return  self._params_model_classifier

    @property 
    def params_model_cooperative(self):         
        return self._params_model_cooperative

    @property
    def params_propagator(self):
        return self._params_propagator                         

    @property
    def params_modulator(self):
        return self._params_modulator

    @property
    def params_datamodule(self):
        return self._params_datamodule

    @property 
    def params_trainer(self):
        return self._params_trainer

    @property
    def all_paths(self):
        return self._all_paths 

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        logging.debug("Parameter_Manager | setting distance to {}".format(value))
        self._distance = value
        
    @property
    def wavelength(self):
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value):
        logging.debug("Parameter_Manager | setting wavelength to {}".format(value))
        self._wavelength = value
    
    @property
    def path_checkpoint_lrn(self):
        return self._path_checkpoint_lrn

    @path_checkpoint_lrn.setter
    def path_checkpoint_lrn(self, value):
        logging.debug("Parameter_Manager | setting path_checkpoing_lrn to {}".format(value))
        self._path_checkpoint_lrn = value

    @property
    def which(self):
        return self._which

    @which.setter
    def which(self, value):
        logging.debug("Parameter_Manager | setting which to {}".format(value))
        self._which = value

    @property
    def adaptive(self):
        return self._adaptive

    @adaptive.setter
    def adaptive(self, value):
        logging.debug("Parameter_Manager | setting adaptive to {}".format(value))
        self._adaptive = value

    @property
    def phase_initialization(self):
        return self._phase_initialization

    @phase_initialization.setter
    def phase_initialization(self, value):
        logging.debug("Parameter_Manager | setting phase_initialization to {}".format(value))
        self._phase_initialization = value

    @property
    def data_split(self):
        return self._data_split

    @data_split.setter
    def data_split(self, value):
        logging.debug("Parameter_Manager | setting data_split to {}".format(value))
        self._data_split = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        logging.debug("Parameter_Manager | setting model_name to {}".format(value))
        self._model_name = value

if __name__ == "__main__":
    import yaml
    params = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)
    pm = Parameter_Manager(params = params)
    print(pm.path_model)

