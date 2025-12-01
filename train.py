"""
Training Entrance
"""

from deepode.nn import Trainer
# from deepode.nn.DataLoad import loadData
from deepode.nn.utils import *
from deepode.nn.config import parser

import os 
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:scipy" 

def run_training_entry(args):
    trainer = Trainer()
    trainer.init_dataloaders(args)
    trainer.build_model(args)
    logging_args(args)        ## print hyper-parameters in log file
    trainer.run_training(args)


def main():
    args = parser().parse_args()

    args.model_root = "model"
    args.modelname = f"DRM19-0D1DPert-ckv8-deepode"
    args.input_path = "/home/yiyuxiao/data/AI4S/DeePCK/Data/DRM19/DRM19_2200wFlameMFPert_X.npy"
    args.label_path = "/home/yiyuxiao/data/AI4S/DeePCK/Data/DRM19/DRM19_2200wFlameMFPert_Y.npy"
    args.mech_path = "mechanism/DRM19.cti"
    args.zero_input = ["ar"]
    args.zero_gradient = ["p", "N2"]

    setup_current_time(args)
    setup_device(args)
    create_model_path(args)
    setup_logging(args)
    run_training_entry(args)


if __name__ == '__main__':
    """
    >>> python train.py -cuda 0,1,2,3,4,5,6,7 -ddp --delta_t 1e-6 -note "this is a test of DDP training"
    >>> python train.py --device="cuda:5" --delta_t 1e-6 -note "a test of single-GPU training"
    """
    main()
