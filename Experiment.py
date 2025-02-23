import os
from Models.CNN import *
from Experiments.classification import Experiment_Classification
from Utils import utils

base_project_dir = os.path.abspath(os.path.dirname(__file__))

model = Experiment_Classification(  epochs=10,
                                    network_structure='CNN',
                                    data_augmentation = False,
                                    base_project_dir=base_project_dir)


model.train()
