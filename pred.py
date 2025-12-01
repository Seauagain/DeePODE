# -*- coding: utf-8 -*-
"""
DeePODE Prediction Interface

This script provides a command-line interface for prediction based on trained neural network models, supporting the following functions:
1. One-step prediction: Perform single-step prediction based on input npy file and save results
2. Continuous prediction: Simulate chemical reaction time evolution and compare with Cantera results
3. Model export: Export model to torch script format
4. Visualization prediction: Perform single-step prediction and generate comparison plots


"""

import os
import sys
import argparse
import numpy as np
from deepode.nn import Predictor
from deepode.utils.parser import parse_keys, dict_to_namespace


import warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:scipy" 



def quick_prediction(args):
    """
    Fast validation of DNN tested on all-ones vector. 
    """
    print(f"Executing one-step prediction: Model={args.modelname}, Epoch={args.epoch}")
    # Initialize model
    model = Predictor()
    model.init_kinetics_model(args.mech_path)
    model.load_model(args.modelname, args.epoch)
    
    # Execute prediction
    inputs = np.ones(model.args.input_dim)
    print("inputs: ", inputs)
    
    prediction = model.net_state2state(inputs)
    print("prediciton: ", prediction)


def one_step_prediction(args):
    """
    One-step prediction based on npy file
    
    Parameters:
        args: Command line arguments, including model name, epoch, mechanism file path, and input file path
    """
    print(f"Executing one-step prediction: Model={args.modelname}, Epoch={args.epoch}")
    
    # Check if input file exists
    # Load input data
    input_data = np.load(args.input_path)
    
    # Determine output file path
    if args.pred_path:
        pred_path = args.pred_path
    else:
        # Default prediction path: replace X with Y in input, or add _pred suffix
        if "_X.npy" in args.input_path:
            pred_path = args.input_path.replace("_X.npy", "_pred_Y.npy")
        else:
            raise ValueError("Label file not specified and cannot infer label file path from input filename")
    
    # Initialize model
    model = Predictor()
    model.init_kinetics_model(args.mech_path)
    model.load_model(args.modelname, args.epoch, args.model_root)
    
    # Execute prediction
    prediction = model.net_state2state(input_data)
    
    # Save prediction results
    np.save(pred_path, prediction)
    print(f"Prediction results saved to: {pred_path}")
    
    return prediction


def temporal_evolution(args):
    """
    Continuous prediction and comparison with Cantera
    
    Parameters:
        args: Command line arguments, including model name, epoch, mechanism file path, and gas conditions
    """
    print(f"Executing continuous prediction: Model={args.modelname}, Epoch={args.epoch}")
    
    # Parse gas conditions
    try:
        phi = float(args.phi)
        temperature = float(args.temperature)
        pressure = float(args.pressure)
        fuel = args.fuel
        reactor = args.reactor
    except ValueError as e:
        raise ValueError(f"Gas condition parameter format error: {e}")
    
    gas_condition = [phi, temperature, pressure, fuel, reactor]
    print(f"Gas conditions: Phi={phi}, T={temperature}K, P={pressure}atm, Fuel={fuel}, Reactor={reactor}")
    
    # Initialize model and execute prediction
    model = Predictor()
    model.init_kinetics_model(args.mech_path)
    
    # Check if using submodels
    if args.submodels:
        submodel_list = args.submodels.split(',')
        print(f"Using submodels: {submodel_list}")
        model.load_sub_models(submodel_list, args.epoch, args.model_root)
    else:
        model.load_model(args.modelname, args.epoch, args.model_root)
    
    # Execute continuous prediction
    model.evolution_predict(
        args.modelname,
        args.epoch,
        gas_condition,
        args.n_step,
        args.builtin_t,
        plot_all=args.plot_all,
        dpi=args.dpi
    )
    
    print("Temporal evolution prediction completed")


def export_model(args):
    """
    Export model to torch script format.
    python pred.py -f export --modelname [modelname] --epoch [epoch]
    """
    model = Predictor()
    output_dir = model.convert2torch_script(args.modelname, args.epoch, args.scriptname, args.model_root)


def visualize_prediction(args):
    """
    One-step prediction based on npy file and generate comparison plots
    
    Parameters:
        args: Command line arguments, including model name, epoch, mechanism file path, and input/label file paths
    """
    print(f"Executing visualization prediction: Model={args.modelname}, Epoch={args.epoch}")
    
    # Check if input file exists
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file does not exist: {args.input_path}")
    
    # Determine label file path
    if args.label_path:
        label_path = args.label_path
    else:
        # Default label path: replace X with Y
        if "_X.npy" in args.input_path:
            label_path = args.input_path.replace("_X.npy", "_pred_Y.npy")
        else:
            raise ValueError("Label file not specified and cannot infer label file path from input filename")
    
    # Check if label file exists
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file does not exist: {label_path}")
    
    # Determine data name
    if args.data_name:
        data_name = args.data_name
    else:
        # Default data name: extract from input filename
        data_name = os.path.basename(args.input_path).replace("_X.npy", "")
        
    model = Predictor()
    model.init_kinetics_model(args.mech_path)
    model.load_model(args.modelname, args.epoch, args.model_root)
    
    # Execute prediction and generate comparison plots
    model.one_step_predict(
        args.modelname,
        args.epoch,
        args.input_path,
        label_path,
        data_name,
        args.size_show,
        args.show_temperature,
        args.plot_dims,
        args.dpi
    )
    
    print("Visualization prediction completed")


def main():
    """
    Main entry point for the application.
    Usage:
    >>> python pred.py onestep_plot
    >>> python pred.py onestep_file
    >>> python pred.py evolution 
    Or you can override the configs through command line.
    >>> python pred.py evolution --modelname DRM19-test
    """
    # Create the main argument parser
    main_parser = argparse.ArgumentParser(description="DeePODE Prediction Tool")
    subparsers = main_parser.add_subparsers(title="Available Commands", dest="command")

    # Check if a command is provided
    if len(sys.argv) < 2:
        main_parser.print_help() 
        return
    
    command = sys.argv[1]


    
    # Execute the corresponding function based on the command
    if command == "onestep_plot":   ## scatter plot of one-step prediction (pred vs. label).
        """
        >>> python pred.py onestep_plot
        """
        # Configuration for one-step prediction plotting
        onestep_plot_config = {
            "model_root": "models",
            "modelname": "DRM19-test-gbct",  # Model name
            "epoch": 5000,                   # Training epochs
            "mech_path": "mechanism/DRM19.cti", # Mechanism file path
            "input_path": "dataset/DRM19/DRM19_0d_manifold_X.npy", # Input data file path
            "label_path": "dataset/DRM19/DRM19_0d_manifold_Y.npy", # Label data file path
            "data_name": "DRM19_0d_manifold", # Data name
            "size_show": 10000,              # Number of samples to display
            "show_temperature": [1000, 2500], # Temperature range to display
            "plot_dims": [0, 1, 2, 3, 4],    # Plot dimensions
            "dpi": 300,                      # Image DPI
        }
        onestep_plot_parser = subparsers.add_parser("onestep_plot", help="Execute visualization prediction")

        # Modify sys.argv for correct parsing
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        args = parse_keys(onestep_plot_config)
        visualize_prediction(args)




    
    elif command == "onestep_file": ## One-step prediction based on the .npy file
        """
        >>> python pred.py onestep_file
        """
       # One-step prediction and save file configuration
        onestep_file_config = {
            "model_root": "models",
            "modelname": "DRM19-test-gbct",  # Model name
            "epoch": 5000,                   # Training epochs
            "mech_path": "mechanism/DRM19.cti", # Mechanism file path
            "input_path": "dataset/DRM19/DRM19_0d_manifold_X.npy", # Input data file path
            "pred_path": "",                  # Prediction output path (optional)
        }
        onestep_file_parser = subparsers.add_parser("onestep_file", help="Execute one-step prediction and save results")

        sys.argv = [sys.argv[0]] + sys.argv[2:]
        args = parse_keys(onestep_file_config)
        one_step_prediction(args)
    




    elif command == "evolution":  ## Continuous temporal evolution trajectory plot.
        """
        >>> python pred.py evolution
        >>> python pred.py evolution --temperature 1650 --n_step 2000 
        """
       # Continuous prediction configuration
        evolution_config = {
            "model_root": "models",
            "modelname": "DRM19-test-gbct",  # Model name
            "epoch": 5000,                   # Training epochs
            "mech_path": "mechanism/DRM19.cti", # Mechanism file path
            "phi": 1.0,                      # Equivalence ratio
            "temperature": 1400,             # Temperature (K)
            "pressure": 1.0,                 # Pressure (atm)
            "fuel": "CH4",                   # Fuel type
            "reactor": "constP",             # Reactor type: constP (constant pressure), constV (constant volume)
            "n_step": 5000,                  # Simulation steps
            "builtin_t": 1e-8,               # Cantera maximum time step
            "plot_all": 1,                   # Whether to plot all features
            "submodels": "",                 # Submodel list, comma separated
            "dpi": 200,                      # Image DPI
        }
        evolution_parser = subparsers.add_parser("evolution", help="Execute continuous prediction")
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        args = parse_keys(evolution_config)

        temporal_evolution(args)
    



    elif command == "export": ## convert checkpoints to torch scripts.
        """
        >>> python pred.py export
        >>> python pred.py export --modelname "DRM19-test"
        """
        # Export model configuration
        export_config = {
            "model_root": "models",          # models root dir.
            "modelname": "DRM19-test-gbct",  # Model name
            "epoch": 5000,                   # Training epochs
            "scriptname": "",                # Script name (optional)
        }
        export_parser = subparsers.add_parser("export", help="Export model to torch script format")
        
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        args = parse_keys(export_config)
        export_model(args)
    
    



    elif command == "dryrun": ## load the model and make predictions on the test inputs.
        """
        >>> python pred.py dryrun
        >>> python pred.py --epoch 4000
        """
        config = {
            "model_root": "models",
            "modelname": "DRM19-test",       # Model name
            "epoch": 5000,                   # Training epochs
            "mech_path": "mechanism/DRM19.cti", # Mechanism file path
        }

        quick_parser = subparsers.add_parser("quick", help="Quick prediction with all-ones vector")
        sys.argv = [sys.argv[0]] + sys.argv[2:]

        args = parse_keys(config)
        quick_prediction(args)
    
    else:
        main_parser.print_help()


if __name__ == '__main__':
    main()