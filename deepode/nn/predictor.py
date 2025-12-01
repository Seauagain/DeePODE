# -*- coding: utf-8 -*-
"""
Utilize the trained neural network to finish downstream tasks.

Created on Wed Jul 27 22:30:20 2022
@author: Yuxiao Yi
"""

## system import 
import os, re
import json
import math
import logging, time
import cantera as ct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from distutils.spawn import find_executable
import argparse
matplotlib.use('AGG')

## custom import 
from .networks import *
from .data_processor import DataProcessor


class Predictor():
    """Neural network utilization class for downstream tasks."""
    
    def __init__(self) -> None:
        """Initialize the Predictor class."""
        self.use_sub_models = False  # use sub-models, default False
        self.data_processor = DataProcessor()
        self.model_root = ""

    def init_kinetics_model(self, mech_path):
        """Instantiate cantera.Solution object.
        
        Parameters
        ----------
        mech_path : str
            Mechanism input file, could be .yaml, .xml or .cti format.
        """
        self.gas = ct.Solution(mech_path)

    def build_model(self, args):
        """Create a neural network.
        
        Parameters
        ----------
        args : argparse.Namespace
            Model configuration parameters.
        """
        dnn_types = {
            'fc': FeedForwardNet,
            'resnet': ResNet,
            'mscale': MultiscaleNet
        }
        self.net = dnn_types[args.net_type](args)
        logging.info(f"The neural network is created. Network type: {args.net_type}")

    def load_model(self, modelname, epoch, device="cpu", model_root="models"):
        """Load a checkpoint, args and norm on cpu.
        
        Parameters
        ----------
        modelname : str
            Name of the model.
        epoch : int
            Epoch number to load.
        device : str, optional
            Device to load the model on. Default 'cpu'.
        """
        self.model_root = model_root
        model_path = os.path.join(model_root, f'{modelname}')
        ckpoint_path = os.path.join(model_path, 'checkpoint', f'model{epoch}.pt')
        setting_path = os.path.join(model_path, 'checkpoint', 'settings.json')
        norm_path = os.path.join(model_path, 'checkpoint', 'norm.json')
        
        self.args = self._json_to_namespace(setting_path)
        self.build_model(self.args)
        self.norm = self._json_to_namespace(norm_path)
        
        self.net.load_state_dict(torch.load(ckpoint_path, map_location='cpu'))
        logging.info(f"The {ckpoint_path} has been loaded")

    def load_sub_models(self, submodel_list, epoch_list=[]):
        """Load checkpoints, args and norms on cpu.
        
        Parameters
        ----------
        submodel_list : list
            List of model names to load.
        epoch_list : list or int, optional
            List of epochs to load for each model. If int, the same epoch is used for all models.
            If empty, default is 5000 for all models.
        """
        if isinstance(epoch_list, int):
            epoch_list = [epoch_list] * len(submodel_list)
        elif not epoch_list:
            epoch_list = [5000] * len(submodel_list)
        elif len(epoch_list) != len(submodel_list):
            raise ValueError("The length of model_list and epoch_list should be the same.")
            
        self.use_sub_models = True
        self.net_list = []
        self.args_list = []
        self.norm_list = []
        
        for idx, (modelname, epoch) in enumerate(zip(submodel_list, epoch_list)):
            self.load_model(modelname, epoch)
            self.net_list.append(self.net)
            self.args_list.append(self.args)
            self.norm_list.append(self.norm)
            
        self.switch_sub_models(0)
        

    def switch_sub_models(self, model_idx):
        """Switch sub-models according to the index of net_list.
        
        Parameters
        ----------
        model_idx : int
            Index of net_list to switch to.
        """
        self.net = self.net_list[model_idx]
        self.args = self.args_list[model_idx]
        self.norm = self.norm_list[model_idx]
        logging.info(f"Note: the current model is {submodel_list[0]}")

    @staticmethod
    def _json_to_namespace(json_path):
        """Load json and return a namespace object.
        
        Parameters
        ----------
        json_path : str
            The json file path.
        
        Returns
        -------
        args : argparse.Namespace
            A namespace object with attributes from the JSON file.
        """
        with open(json_path, 'r') as f:
            args_dict = json.load(f)
        return argparse.Namespace(**args_dict)

    def ct_state2state(self, state, delta_t, reactor, builtin_t):
        """Use Cantera to advance the state of the reactor network from the current time t towards t+Δt.
        
        Parameters
        ----------
        state : numpy.ndarray
            State vector organized as T,P(atm),Y.
        delta_t : float
            Evolution time step (sec).
        reactor : str
            The type of reactor, could be 'constP' or 'constV'.
        builtin_t : float
            Max time step for CVODE in Cantera.
        
        Returns
        ------- 
        state_out : numpy.ndarray
            The output state vector after Δt which is organized as T,P(atm),Y.
        """
        state = state.reshape(1, -1)
        reactor_types = {
            'constV': ct.IdealGasReactor,
            'constP': ct.IdealGasConstPressureReactor
        }
        
        self.gas.TPY = state[0, 0], state[0, 1] * ct.one_atm, state[0, 2:]
        r = reactor_types[reactor](self.gas)
        sim = ct.ReactorNet([r])
        sim.max_time_step = builtin_t
        sim.max_steps = 2e5  # max iteration steps for CVODES, default 2e4
        # sim.atol = 1e-24  # default 1e-15
        sim.rtol = 1e-15  # default 1e-9
        sim.advance(delta_t)

        state_out = np.hstack(self.gas.TPY)
        state_out[1] = self.gas.P / ct.one_atm
        return state_out.reshape(1, -1)

    def net_state2state(self, state):
        """Use DNN to advance the state of the reactor network from the current time t towards t+Δt.
        
        Parameters
        ----------
        state : numpy.ndarray
            State vector organized as T,P(atm),Y.
            
        Returns
        -------
        output_bct : numpy.ndarray
            The output state vector after Δt which is organized as T,P(atm),Y.
        """
        args = self.args
        norm = self.norm
        bct_lamda = args.power_transform
        delta_t = args.delta_t
        state_bct = state.reshape(-1, args.dim)  # TPY
    
        if not hasattr(args, "num_nonspecies"):
            self.num_nonspecies = 2

        ## Energy conservation for single state vector. 
        is_single_state = state_bct.shape[0] == 1
        if is_single_state:
            self.gas.TPY = state_bct[0, 0], state_bct[0, 1] * ct.one_atm, state_bct[0, 2:]
            enthalpy, pressure = self.gas.HP
        
        ## Box-Cox transformation for input states.
        state_bct[:, self.num_nonspecies:] = self.data_processor.box_cox( state_bct[:, self.num_nonspecies:], bct_lamda)

        ## normalization 
        state_normalized = (state_bct - norm.input_mean) / norm.input_std

        ## set special species
        if hasattr(args, "zero_input"):
            all_names = ["T", "P"] + self.gas.species_names
            all_names = list(map(str.upper, all_names))
            for sp in args.zero_input:
                if sp.upper() in all_names:
                    zero_idx = all_names.index(sp.upper())
                    state_normalized[:, zero_idx] = 0

        ## nmodel prediction 
        state_normalized = torch.from_numpy(state_normalized).float()
        output_normalized = self.net.forward(state_normalized)
        output_normalized = output_normalized.detach().cpu().numpy()

        ## inverse normalization
        output = output_normalized * norm.label_std + norm.label_mean

        ## use inverse Generalized BCT if specified
        if hasattr(args, "use_GBCT") and args.use_GBCT:
            output = self.data_processor.inverse_generalized_box_cox(output, args.lam)
            # output = self._inverse_generalized_bct(output, args.lam)
        
        ## add operation
        output_bct = output * delta_t + state_bct

        ## inverse Box-Cox transformation
        output_bct[:, self.num_nonspecies:] = self.data_processor.inverse_box_cox(output_bct[:, self.num_nonspecies:], bct_lamda)

        # energy conservation, use cantera to correct the temperature
        if is_single_state:
            print("\reneregy conservation correct for chemical data", end="")
            self.gas.HPY = enthalpy, pressure, output_bct[:, 2:]
            output_bct[:, 0] = self.gas.T

        return output_bct


    def init_chem_state(self, gas_condition, custom_composition=None):
        """Initialize the state vector [T, p(atm), Yi] for chemical reactor simulation.
        
        Parameters
        ----------
        gas_condition : list or tuple
            Initial conditions for zero dimensional ignition. Organized as [Phi,T,P(atm),fuel,reactor]. 
            Phi: equivalence ratio. T: temperature (K). P: pressure (atm). 
            fuel: fuel species name. reactor: reactor type.
        custom_composition : dict, optional
            Custom species composition as a dictionary {species_name: mole_fraction}.
            If provided, this overrides the equivalence ratio setting.
            
        Returns
        -------
        initial_state : numpy.ndarray
            The initial state vector organized as T,P(atm),Y with shape (1, n_dims).
        
        Raises
        ------
        AttributeError
            If gas model has not been initialized with init_kinetics_model.
        """
        # if not hasattr(self, 'gas'):
        #     raise AttributeError("Gas model not initialized. Call init_kinetics_model first.")
            
        # Phi, T, P, fuel, _ = gas_condition
        
        # # Set gas composition
        # if custom_composition is not None:
        #     self.gas.TPX = T, P * ct.one_atm, custom_composition
        # else:
        #     self.gas.set_equivalence_ratio(Phi, fuel, 'O2:1.0,N2:3.76')
        #     self.gas.TP = T, P * ct.one_atm
        
        # # Create initial state vector [T, P, Y1, Y2, ..., Yn]
        # initial_state = np.hstack([T, P, self.gas.Y])
        
        # return initial_state.reshape(1, -1)
        pass 


    def evolution_predict(self, modelname, epoch, gas_condition, n_step, builtin_t, plot_all=False, dpi=200):
        """Temporal evolution simulation computed by DNN and Cantera.
        
        Parameters
        ----------
        modelname : str
            The folder name of DNN model.
        epoch : int
            Epoch for loading checkpoint.  
        gas_condition : list or tuple
            Initial conditions for zero dimensional ignition. Organized as [Phi,T,P(atm),fuel,reactor]. 
            Phi: equivalence ratio. T: temperature (K). P: pressure (atm). 
            fuel: fuel species name. reactor: reactor type.
        n_step : int
            Simulation steps, time step = args.delta_t.
        builtin_t : float
            Max time step for CVODES in Cantera.
        plot_all : bool, optional
            Whether plot all the features (T,P,Yi), if True plot all else plot temperature. Default False.
        dpi : int, optional
            The dpi used to save figure. Default 200.
        """
        Phi, T, P, fuel, reactor = gas_condition
        self.gas.set_equivalence_ratio(Phi, fuel, 'O2:1.0,N2:3.76')
        delta_t = self.args.delta_t

        initial_state = np.c_[T, P, self.gas.Y.reshape(1, -1)]
        state_net = initial_state.copy()
        state_cantera = initial_state.copy()

        # Single model simulation
        if not self.use_sub_models:
            for i in range(n_step):
                # Calculate cantera output
                current_cantera = state_cantera[i, :].copy()
                next_cantera = self.ct_state2state(current_cantera, delta_t, reactor, builtin_t)
                state_cantera = np.r_[state_cantera, next_cantera]
                
                # Calculate net output
                current_net = state_net[i, :].copy()
                next_net = self.net_state2state(current_net)
                state_net = np.r_[state_net, next_net]
        # Multi-model simulation
        else:
            # Default sub-model
            model_idx = 0
            # Switch condition for sub-models
            for i in range(n_step):
                # Calculate cantera output
                current_cantera = state_cantera[i, :].copy()
                next_cantera = self.ct_state2state(current_cantera, delta_t, reactor, builtin_t)
                state_cantera = np.r_[state_cantera, next_cantera]
                
                # Check if we need to switch models
                if i > 2:
                    grad_T = np.abs((state_net[-1, 0] - state_net[-2, 0]) / self.args.delta_t)
                    current_T = state_net[-1, 0]

                    if model_idx == 1 and current_T > 2500 and grad_T < 5e5:
                        model_idx = 2
                        self.switch_sub_models(model_idx)

                    if model_idx == 0 and current_T > T and grad_T > 1e5:
                        model_idx = 1
                        self.switch_sub_models(model_idx)

                # Calculate net output
                current_net = state_net[i, :].copy()
                next_net = self.net_state2state(current_net)
                state_net = np.r_[state_net, next_net]

        # Save simulation results
        # np.save(f"{modelname}_epoch{epoch}_T{T}_P{P}_Phi{Phi}_cvode.npy", state_cantera)
        # np.save(f"{modelname}_epoch{epoch}_T{T}_P{P}_Phi{Phi}_net.npy", state_net)
        # logging.info("Simulation data saved")
        
        # Calculate and display temperature error
        # index = 0  # Temperature index
        # net_T = state_net[-1, index]
        # vode_T = state_cantera[-1, index]
        # err = np.abs(net_T - vode_T) / vode_T
        # logging.info(f"T rel err: {err:.5f}")

        # Calculate and display enthalpy error (assuming index 3 is enthalpy)
        # index = 3
        # net_H = state_net[-1, index]
        # vode_H = state_cantera[-1, index]
        # err = np.abs(net_H - vode_H) / vode_H
        # logging.info(f"H rel err: {err:.5f}")
        
        # Display simulation results
        plot_func = self.plot_all if plot_all else self.plot_temperature
        plot_func(modelname, epoch, gas_condition, state_net, state_cantera, dpi)
    

    def convert2torch_script(self, modelname, epoch, scriptname='', model_root="models"):
        r"""
        Convert a pytorch-format `.pt` file to torch script format for usage in libtorch (C++).
        
        Parameters
        -----------
        modelname : str
            The folder name of DNN model.
        epoch : int
            Epoch for loading checkpoint.
        scriptname : str
            The filename for torch script format `.pt`. 
    
        Returns
        -------
            The torch script format `.pt` will be saved along with settings.json and norm.json.
        """
        self.load_model(modelname, epoch)
        
        dim = self.args.input_dim if hasattr(self.args, "input_dim") else self.args.dim
        example = torch.ones(1, dim)
        # to avoid warnings:`optimize` is deprecated and has no effect.
        with torch.jit.optimized_execution(True): 
            traced_script_module = torch.jit.script(self.net, example)
        
        base_dir = os.path.join(model_root, modelname, "torch_scripts", modelname)
        os.makedirs(base_dir, exist_ok=True)
        
        if scriptname == '':
            script_filename = f"script_model{epoch}.pt"
        else:
            script_filename = f"{scriptname}.pt"
        
        script_path = os.path.join(base_dir, script_filename)
        traced_script_module.save(script_path)
        
        src_dir = os.path.join(model_root, modelname, "checkpoint")
        setting_src = os.path.join(src_dir, "settings.json")
        norm_src = os.path.join(src_dir, "norm.json")
        
        import shutil
        shutil.copy2(setting_src, os.path.join(base_dir, "settings.json"))
        shutil.copy2(norm_src, os.path.join(base_dir, "norm.json"))
        
        print(f"\nTorch script and configuration files saved in {base_dir}")
        return base_dir








    def plot_temperature(self, modelname, epoch, gas_condition, state_net, state_cantera, dpi):
        """Draw and save the ignition curves simulated by Cantera and DNN.
        
        Parameters
        ----------
        modelname : str
            The folder name of DNN model.
        epoch : int
            Epoch for loading checkpoint.  
        gas_condition : list or tuple
            Initial conditions for zero dimensional ignition. Organized as [Phi,T,P(atm),fuel,reactor]. 
            Phi: equivalence ratio. T: temperature (K). P: pressure (atm). 
            fuel: fuel species name. reactor: reactor type.
        state_net : numpy.ndarray
            The n-step continuous evolution simulated by DNN.
        state_cantera : numpy.ndarray
            The n-step continuous evolution simulated by Cantera.
        dpi : int
            The dpi used to save figure.
        """
        Phi, T, P, fuel, reactor = gas_condition
        num_steps = range(len(state_net))
        time_seq = np.array(num_steps) * self.args.delta_t * 1e3  # ms

        # Plot settings
        title_size = 25
        label_size = 16
        legend_size = 13
        tick_size = 11.5

        # Create temperature plot
        p1, = plt.plot(time_seq,
                       state_cantera[:, 0],
                       color="chocolate",
                       linewidth=1.5,
                       alpha=1)
        p2, = plt.plot(time_seq,
                       state_net[:, 0],
                       color="green",
                       linewidth=1.5,
                       linestyle='-.',
                       alpha=1)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.title(self._latex_style_name(0), fontsize=title_size)
        plt.xlabel("time (ms)", fontsize=label_size)
        plt.ylabel('Temperature (K)', fontsize=label_size)

        # Set y-axis limits
        plt.ylim(
            np.min(state_cantera[:, 0]) - 100,
            np.max(state_cantera[:, 0]) + 100)
        plt.legend([p1, p2], ["CVODE", 'DNN'],
                   loc="lower right",
                   fontsize=legend_size)

        # Save the figure
        pic_name = f'{modelname}_Phi={Phi}_T={T}_P={P}_epoch={epoch}.png'
        pic_folder = os.path.join(self.model_root, modelname, 'pic', f'pics{epoch}')
        os.makedirs(pic_folder, exist_ok=True)
        pic_path = os.path.join(pic_folder, pic_name)
        logging.info(f'Simulation picture saved in {pic_path}')
        plt.savefig(pic_path, dpi=dpi)
        plt.close()

    def plot_all(self, modelname, epoch, gas_condition, state_net,
                state_cantera, dpi):
        """Draw and save the evolution of temperature, pressure and species mass fraction.
        
        Parameters
        ----------
        modelname : str
            The folder name of DNN model.
        epoch : int
            Epoch for loading checkpoint.  
        gas_condition : list or tuple
            Initial conditions for zero dimensional ignition. Organized as [Phi,T,P(atm),fuel,reactor]. 
            Phi: equivalence ratio. T: temperature (K). P: pressure (atm). 
            fuel: fuel species name. reactor: reactor type.
        state_net : numpy.ndarray
            The n-step continuous evolution simulated by DNN.
        state_cantera : numpy.ndarray
            The n-step continuous evolution simulated by Cantera.
        dpi : int
            The dpi used to save figure.
        """
        Phi, T, P, _, _ = gas_condition
        num_steps = range(len(state_net))
        time_seq = np.array(num_steps) * self.args.delta_t * 1e3  # ms

        title_size = 20
        label_size = 16
        legend_size = 13
        tick_size = 14

        fig = plt.figure(figsize=(12.8, 9.6))
        plot_methods = {'semilogy': plt.semilogy, 'plot': plt.plot}
        for i in range(2 + self.gas.n_species):
            order_subplot = i % 9 + 1
            ax = fig.add_subplot(3, 3, order_subplot)
            method = 'plot' if i in [0, 1] else 'semilogy'
            plot_handle = plot_methods[method]
            p1, = plot_handle(time_seq,
                              state_cantera[:, i],
                              color="chocolate",
                              linewidth=2,
                              alpha=1)
            p2, = plot_handle(time_seq,
                              state_net[:, i],
                              color="green",
                              linewidth=2,
                              linestyle='-.',
                              alpha=1)

            ## todo: set label/ticks/legend
            if i == 1:
                plt.ylabel('Pressure (atm)', fontsize=label_size)
            elif i == 0:
                plt.ylim(
                    np.min(state_cantera[:, 0]) - 100,
                    np.max(state_cantera[:, 0]) + 100)
                plt.ylabel('Temperature (K)', fontsize=label_size)
            else:
                ## todo: adjust the display range of mass fraction
                degree_mid = math.floor(np.log10(state_cantera[-1, i] + 1e-30))
                degree_lower = degree_mid - 3
                degree_upper = min(0, degree_mid + 3)
                plt.ylim(10**degree_lower, 10**degree_upper)
                ticks = [
                    10**(i) for i in range(degree_lower, degree_upper + 1)
                ]
                plt.yticks(ticks, fontsize=tick_size)
                # ax.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
                plt.ylabel('mass fraction', fontsize=label_size)

            ##
            plt.legend([p1, p2], [
                "CVODE",
                'DNN',
            ],
                       fontsize=legend_size,
                       loc='lower right')
            plt.title(self._latex_style_name(i),
                      fontsize=title_size,
                      fontweight='bold')
            plt.xlabel("time (ms)", fontsize=label_size)
            plt.yticks(fontsize=tick_size)
            plt.xticks(fontsize=tick_size)

            ## condition for saving pictures
            pic_folder = os.path.join(self.model_root, modelname, 'pic', 'TemporalEvolution', f'pics{epoch}')
            os.makedirs(pic_folder, exist_ok=True)
            if order_subplot == 9:
                plt.tight_layout()
                # fig.suptitle('epoch={}'.format(epoch))
                pic_name = f'{modelname}_Phi={Phi}_T={T}_P={P}_epoch={epoch}_all{math.ceil(i / 9)}.png'
                pic_path = os.path.join(pic_folder, pic_name)
                plt.savefig(pic_path, dpi=dpi)
                print(f'simulation picture saved in {pic_path}')
                plt.close()
                fig = plt.figure(figsize=(12.8, 9.6))
            elif i == 1 + self.gas.n_species:
        
                plt.tight_layout()
                pic_name = f'{modelname}_Phi={Phi}_T={T}_P={P}_epoch={epoch}_all{math.ceil((i+1) / 9)}.png'

                pic_path = os.path.join(pic_folder, pic_name)
                plt.savefig(pic_path, dpi=dpi)
                print(f'simulation picture saved in {pic_path}')
                plt.close()


    def one_step_predict(self, modelname, epoch, input, label, data_name, size_show,
                       show_temperature=[1000, 3000], plot_dims='all', dpi=200):
        """Draw and save single-step prediction of DNN w.r.t a chemical test dataset.
        
        Parameters
        ----------
        modelname : str
            The folder name of DNN model.
        epoch : int
            Epoch for loading checkpoint. 
        input : str or numpy.ndarray
            Input dataset, could be file path or numpy.ndarray.
        label : str or numpy.ndarray
            Label dataset, could be file path or numpy.ndarray.
        data_name : str
            The data name suffix to save figure.
        size_show : int
            Number of data points to show.
        show_temperature : list, optional
            The range of temperature shown in the picture. Default [1000, 3000].
        plot_dims : str, list or tuple, optional
            List of dimensions expected to be plotted e.g. [1,4,5,7] or Range(20) or ['T','O','CH4','N2'].
            Default string 'all' means plotting all the dimensions.
        dpi : int, optional
            The dpi used to save figure. Default 200.
            
        Returns
        -------
        prediction : numpy.ndarray
            DNN model one-step prediction on the input dataset.
        """
        # Check input class type
        if isinstance(input, np.ndarray):
            pass
        elif isinstance(input, str):
            input = np.load(input)
            label = np.load(label)
        else:
            raise TypeError(
                f"Expected input to be <class 'str'> or <class 'np.ndarray'> but got {input.__class__}")
        
        # Shuffle dataset and randomly pick data
        permutation = np.random.permutation(input.shape[0])
        input = input[permutation]
        label = label[permutation]
        input = input[:size_show, :]
        label = label[:size_show, :]
        logging.info(f'One-step prediction takes data size {input.shape}')
        
        # Filter data by temperature range
        rows = np.where((input[:, 0] >= show_temperature[0]) & 
                        (input[:, 0] <= show_temperature[1]))[0]
        input = input[rows, :]
        label = label[rows, :]

        # Get DNN prediction
        prediction = self.net_state2state(input)

        # Check dimensions to display
        names = ['T', 'P'] + self.gas.species_names
        n_dims = self.gas.n_species + 2
        if plot_dims == 'all':
            plot_dims = range(n_dims)
            
        # Create plots for each dimension
        for dim in plot_dims:
            if isinstance(dim, str):
                if dim in ['T', 'P']:
                    dim = names.index(dim)
                else:
                    dim = self.gas.species_index(dim) + 2
            self._one_step_pred_plot(modelname, epoch, label, prediction, dim,
                                  data_name, show_temperature, dpi)
        logging.info(f"The display range of temperature is {show_temperature}K")
        
        return prediction

    def _one_step_pred_plot(self, modelname, epoch, label, prediction, dim,
                         data_name, show_temperature, dpi):
        """Built-in function to plot and save one-step prediction.
        
        Parameters
        ----------
        modelname : str
            The folder name of DNN model.
        epoch : int
            Epoch for loading checkpoint.
        label : numpy.ndarray
            Ground truth labels.
        prediction : numpy.ndarray
            Model predictions.
        dim : int
            Dimension index to plot.
        data_name : str
            The data name suffix to save figure.
        show_temperature : list
            The range of temperature shown in the picture.
        dpi : int
            The dpi used to save figure.
        """
        # Plot settings
        label_size = 14
        title_size = 14
        ticks_size = 14
        cbar_size = 12
        
        # Prepare axis labels
        species = self.gas.species_names
        axis_labels = ['Temperature (K)', 'Pressure (atm)'] + [sp for sp in species]
        
        # Create figure
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Plot identity line
        plt.plot(label[:, dim], label[:, dim], 'k-', linewidth=0.8, alpha=0.6)
        
        # Create scatter plot
        points = plt.scatter(
            label[:, dim],
            prediction[:, dim],
            c=label[:, 0],  # colored by temperature
            vmin=show_temperature[0],
            vmax=show_temperature[1],
            cmap='rainbow',
            s=1,
            marker='.',
            alpha=1)
            
        # Use log scale for species mass fractions
        if dim > 1:
            ax.set_xscale('log')
            ax.set_yscale('log')

        # Set labels and ticks
        plt.xticks(fontsize=ticks_size)
        plt.yticks(fontsize=ticks_size)
        plt.xlabel(f'Label {axis_labels[dim]}', fontsize=label_size)
        plt.ylabel(f'Predicted {axis_labels[dim]}', fontsize=label_size)
        plt.axis('square')
        
        # Add colorbar
        cbar = plt.colorbar(points)
        cbar.set_label("Temperature (K)", fontsize=title_size, fontweight='bold')
        cbar.ax.tick_params(labelsize=cbar_size)

        # Save figure
        pic_folder = os.path.join(self.model_root, modelname, 'pic', 'OneStepPred', data_name)
        os.makedirs(pic_folder, exist_ok=True)
        pic_path = os.path.join(
            pic_folder,
            f'{modelname}_epoch{epoch}_OneStepPred_on_{data_name}_dim={dim}.png'
        )
        plt.savefig(pic_path, dpi=dpi)
        plt.close()
        print(f'One-step prediction picture saved in {pic_path}')
        # logging.info(f'One-step prediction picture saved in {pic_path}')


    @staticmethod
    def _latexStyle(text):
        r"""Convert the given text to :math:`\LaTeX` bold style e.g. T-->$\\bf{T}$, CH4-->$\\bf{CH_{4}}$, C10H18O8-->$\\bf{C_{10}H_{18}O_{8}}$ """
        content = re.findall('\w\d+',
                             text)  # find letter+number e.g. C10 H2 N8 O12...
        content = list(set(content))  #unique
        for letter_number in content:
            letter = letter_number[0]
            number = letter_number[1:]
            text = re.sub(letter_number, f'{letter}_' + '{' + number + '}',
                          text)
        return '$\\bf{' + text + '}$'


    def _latex_style_name(self, index):
        r"""For a chemical dataset (TPY), the state names could be denoted as ['T','P',...species names]. Convert state_names[index] to 
        :math:`\LaTeX` bold style if there exists :math:`\LaTeX` enviroment.
        """
        state_names = ['T', 'P'] + self.gas.species_names
        name = self._latexStyle(state_names[index]) if find_executable(
            'latex') else state_names[index]
        return name

    def chem_evolution_preview(self,
                        mech_path,
                        gas_condition,
                        n_step,
                        delta_t,
                        dpi=200):
        r"""Zero-dimensional ignition simulated by Cantera.
        Parameters
        ----------
        gas_condition : list 
            [Phi, T, P, fuel, reactor]
        n_step : int
            Simulation steps
        """
        self.initGas(mech_path)
        mech_file = os.path.split(mech_path)[-1]
        mech_prefix = mech_file[:-5]  #mechanism file name
        # print(mech_prefix)
        print(f"mech_path: {mech_path}")
        print(f"species num: {self.gas.n_species}")
        print(f"reactions num: {self.gas.n_reactions}")

        Phi, T, P, fuel, reactor = gas_condition
        self.gas.set_equivalence_ratio(Phi, fuel, 'O2:1.0,N2:3.76')
        initial_state = np.c_[T, P, self.gas.Y.reshape(1, -1)]
        state_cantera = initial_state.copy()

        for i in range(n_step):
            ## todo: calculate cantera output
            current_cantera = state_cantera[i, :].copy()
            next_cantera = self.ctOneStep(current_cantera,
                                          delta_t,
                                          reactor,
                                          builtin_t=1e-8)
            state_cantera = np.r_[state_cantera, next_cantera]

        ## plot simulation curves
        time = np.array(range(n_step + 1)) * delta_t * 1e3
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('mass fraction')
        ax1.set_xlabel('time (ms)')
        ax1.set_xlim(-1e-3, time[-1])
        ax1.set_ylim(1e-10, 1)
        plot_list = []

        spces = [fuel, "O2", "H", "O"]
        colors = ["k", "purple", "orange", "blue"]
        for i, sp in enumerate(spces):
            index = self.gas.species_index(sp)
            p, = ax1.semilogy(time,
                              state_cantera[:, 2 + index],
                              color=colors[i])  #H
            plot_list.append(p)
        #plot temperature
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_xlabel('time (ms)')
        ax2.set_ylabel('Temperature (K)', color=color)
        p, = ax2.plot(time, state_cantera[:, 0], color=color, linewidth=1.5)
        plot_list.append(p)
        ax2.tick_params(axis='y', labelcolor=color)
        #legend
        plot_labels = spces + ["T"]
        plt.legend(plot_list, plot_labels, loc='lower right')
        plt.grid(alpha=0.3, axis='both', linestyle='-.')
        plt.title('Homogeneous Ignition %s/Air' % fuel)
        #save pic
        pic_name = f"zeroD_{mech_prefix}_{fuel}_Phi={Phi}_T={T}_P={P}.png"
        pic_home = os.path.join("picture", "ZeroD")
        pic_path = os.path.join(pic_home, pic_name)
        os.makedirs(pic_home, exist_ok=True)
        plt.savefig(pic_path, dpi=dpi)
        print(f"ignition preview picture saved in {pic_path}")
        plt.close()

    
    

    