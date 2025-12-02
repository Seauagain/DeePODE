# -*- coding: utf-8 -*-
"""
Train the neural networks.

Created on Wed Jul 27 22:30:20 2022
@author: Yuxiao Yi
"""

## system import 
import torch 
from torch import optim 
import torch.distributed as dist 
import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import time
import os, sys
import math, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging


## custom import 
from .networks import *
from .dataloader import create_dataloaders


class Trainer():
    """Neural network training class supporting single GPU and distributed training."""
    def __init__(self) -> None:
        pass
    
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
        self.count_parameters(args)
        logging.info(f"The neural network is created. Network type: {args.net_type}")
    
    def init_dataloaders(self, args):
        # Create dataloaders
        train_loader, valid_loader, norm_params = create_dataloaders(args)
        



    def count_parameters(self, args):
        """Count model parameters and FLOPs.
        
        Parameters
        ----------
        args : argparse.Namespace
            Model configuration parameters.
        """
        input_tensor = torch.ones(1, args.input_dim)
        # from fvcore.nn import FlopCountAnalysis
        # flops = FlopCountAnalysis(self.net, input_tensor)
        # args.flops = flops.total()
        args.total_params = sum(p.numel() for p in self.net.parameters())
        logging.info(f"Model total parameters: {args.total_params}")
    
    def load_ckpoint(self, modelname, epoch, model_root="models"):
        """Load a checkpoint on cpu.
        
        Parameters
        ----------
        modelname : str
            Name of the model.
        epoch : int
            Epoch number to load.
        """
        model_path = os.path.join(model_root, f'{modelname}')
        ckpoint_path = os.path.join(model_path, 'checkpoint', f'model{epoch}.pt')
        self.net.load_state_dict(torch.load(ckpoint_path, map_location='cpu'))
        logging.info(f"The {ckpoint_path} has been loaded")


    def setup_training(self, args):
        """Initialize the optimizer and loss function.
        
        Parameters
        ----------
        args : argparse.Namespace
            Training configuration parameters.
        """
        optimizers = {
            'SGD': optim.SGD(self.net.parameters(), lr=args.learnrate, momentum=0.9),
            'Adam': optim.Adam(self.net.parameters(), lr=args.learnrate)
        }
        
        loss_funs = {
            'MSE': nn.MSELoss(),
            'MAE': nn.L1Loss(),
            'L1': nn.L1Loss(),
            'CE': nn.CrossEntropyLoss(),
        }
        
        self.optimizer = optimizers[args.optim]
        self.loss_fun = loss_funs[args.lossfun]
        logging.info(f"Using optimizer: {args.optim}, loss function: {args.lossfun}")

    def run_training(self, args,init_epoch=1):
        """Main training entrance function.
        
        Parameters
        ----------
        args : argparse.Namespace
            Training configuration parameters.
        init_epoch : int, optional
            Starting epoch. Default 1.
        """
        # Choose training method based on configuration
        if torch.cuda.is_available() and args.use_ddp:
            self._runner_for_ddp(args, init_epoch)
        else:
            self.train_single(args, init_epoch)


    def train_single(self, args, init_epoch=1):
        """Train the model using single GPU or DataParallel.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data.
        valid_loader : torch.utils.data.DataLoader
            DataLoader for validation data.
        args : argparse.Namespace
            Training configuration parameters.
        norm_params : dict
            Normalization parameters.
        init_epoch : int, optional
            Starting epoch. Default 1.
        """
        # 
        train_loader, valid_loader, norm_params = create_dataloaders(args)

        # Move model to device
        self.net = self.net.to(args.device)
        logging.info(f"The model is on the device: {args.device}")
                
        # Initialize optimizer and loss function
        self.setup_training(args)
        
        # Save initial model state
        self.save_model(args, norm_params, epoch=0)
        
        # Initialize training variables
        train_loss = []
        valid_loss = []
        lr_current = args.learnrate
        batch_current = args.batch_size

        ## Hint: extract data blocks to speed up the epoch!
        ## Hint: use pinned memory to accelerate IO
        inputs_train, labels_train = self.contiguous_data(train_loader)
        inputs_valid, labels_valid = self.contiguous_data(valid_loader)       
        total_samples = inputs_train.size(0) 
        
        # Training loop
        for epoch in range(init_epoch, args.max_epoch + 1):
            epoch_start_time = time.time()
            
            # Learning rate and batch size adjustment
            if epoch % args.epoch_decay == 0 and epoch > init_epoch:
                batch_current = min(int(batch_current * args.batch_grow_rate), args.max_batch_size) if hasattr(args, 'max_batch_size') else int(batch_current * args.batch_grow_rate)
                lr_current = lr_current * args.lr_decay_rate
                self.optimizer = optim.Adam(self.net.parameters(), lr=lr_current)
                logging.info(f"Adjusted learning rate to {lr_current} and batch size to {batch_current}")
            
            # evaluate the full-batch loss on training and validation dataset
            self.net.eval()
            train_temp_loss=self.get_loss(inputs_train, labels_train, args.device, eval_batch_size=args.valid_batch_size)
            valid_temp_loss=self.get_loss(inputs_valid, labels_valid, args.device, eval_batch_size=args.valid_batch_size)
            train_loss.append(train_temp_loss)
            valid_loss.append(valid_temp_loss)
            # train_temp_loss = 1e-3 ## dryrun
            # valid_temp_loss = 1e-3

            # Training phase
            self.net.train()
            batch_count =  math.ceil(total_samples / batch_current)
            epoch_loss = 0
            for step in range(batch_count):
                index_start = step * batch_current
                index_end = min(index_start + batch_current, total_samples)
                inputs = inputs_data[index_start:index_end]
                labels = labels_data[index_start:index_end]
                if torch.cuda.is_available():
                    inputs = inputs.to(args.device, non_blocking=True)
                    labels = labels.to(args.device, non_blocking=True)
                # Forward pass
                outputs = self.net(inputs)
                loss = self.loss_fun(outputs, labels).mean()
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            epoch_time = time.time() - epoch_start_time
            logging.info(f'Epoch: {epoch:^5} | Train loss: {train_temp_loss:.5f} | Valid loss: {valid_temp_loss:.5f} | Time: {epoch_time:.2f}s')

            # Save checkpoints and loss plots
            if epoch % 10 == 0:
                self.save_loss(train_loss, valid_loss, args)
            
            if epoch % 100 == 0:
                self.save_model(args, norm_params, epoch)
        
        # Save final model and loss
        self.save_model(args, norm_params, args.max_epoch)
        self.save_loss(train_loss, valid_loss, args)
        logging.info("Training completed successfully")

    def _runner_for_ddp(self, args, init_epoch=1):
        """Train the model using DistributedDataParallel.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data.
        valid_loader : torch.utils.data.DataLoader
            DataLoader for validation data.
        args : argparse.Namespace
            Training configuration parameters.
        norm_params : dict
            Normalization parameters.
        init_epoch : int, optional
            Starting epoch. Default 1.
        """
        mp.spawn(
            self.train_ddp,
            args=(args, init_epoch),
            nprocs=args.world_size
        )
    
    @staticmethod
    def contiguous_data(dataloader):
        inputs, labels = dataloader[:]
        inputs = inputs.contiguous().pin_memory() 
        labels = labels.contiguous().pin_memory() 
        return inputs, labels
    
    def train_ddp(self, rank, args, init_epoch):
        """Runner function for DistributedDataParallel training.
        
        Parameters
        ----------
        rank : int
            Process rank.
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data.
        valid_loader : torch.utils.data.DataLoader
            DataLoader for validation data.
        args : argparse.Namespace
            Training configuration parameters.
        init_epoch : int
            Starting epoch.
        """
        # Setup distributed environment
        os.environ['MASTER_ADDR'] = args.master_host
        os.environ['MASTER_PORT'] = args.master_port
        
        dist.init_process_group('nccl', rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        args.device = f'cuda:{rank}'
        
        # Setup logging for this process
        from .utils import setup_logging
        setup_logging(args)

        # Create dataloaders after initializing process group
        train_loader, valid_loader, norm_params = create_dataloaders(args, distributed=True, rank=rank)
        # Move model to device and wrap with DDP
        self.net = self.net.to(args.device)
        self.net = DDP(self.net, device_ids=[rank], output_device=rank)
        logging.info(f"Process {rank}: Model initialized with DDP")
        
        # Initialize optimizer and loss function
        self.setup_training(args)
        
        # Save initial model state (only on rank 0)
        if rank == 0:
            self.save_model(args, norm_params, epoch=0)
        dist.barrier()
        
        # Initialize training variables
        train_loss = []
        valid_loss = []
        lr_current = args.learnrate
        batch_current = args.batch_size
 
        ## Hint: extract data blocks to speed up the epoch!
        ## Hint: use pinned memory to accelerate IO
        inputs_train, labels_train = self.contiguous_data(train_loader)
        inputs_valid, labels_valid = self.contiguous_data(valid_loader)        

        batch_current = args.batch_size
        total_samples = inputs_train.size(0)
     
        # Training loop
        for epoch in range(init_epoch, args.max_epoch + 1):
            epoch_start_time = time.time()
            
            # Learning rate and batch size schedule
            if epoch % args.epoch_decay == 0 and epoch > init_epoch:
                batch_current = min(int(batch_current * args.batch_grow_rate), args.max_batch_size) if hasattr(args, 'max_batch_size') else int(batch_current * args.batch_grow_rate)
                lr_current = lr_current * args.lr_decay_rate
                self.optimizer = optim.Adam(self.net.parameters(), lr=lr_current)
                logging.info(f"Process {rank}: Adjusted learning rate to {lr_current} and batch size to {batch_current}")
            
            # evaluate the full-batch loss on training and validation dataset
            self.net.eval()
            train_temp_loss=self.get_loss(inputs_train, labels_train, args.device, eval_batch_size=args.valid_batch_size)
            valid_temp_loss=self.get_loss(inputs_valid, labels_valid, args.device, eval_batch_size=args.valid_batch_size)
            train_loss.append(train_temp_loss)
            valid_loss.append(valid_temp_loss)
            # train_temp_loss = 1e-3 ## dryrun
            # valid_temp_loss = 1e-3

            # Training phase
            self.net.train()
            # shuffle_indices = torch.randperm(total_samples)
            batch_count =  math.ceil(total_samples / args.batch_size)
            epoch_loss = 0
            for step in range(batch_count):
                index_start = step * batch_current
                index_end = min(index_start + batch_current, total_samples)
                # batch_indices = shuffle_indices[index_start:index_end]    ## shuffle
                # batch_indices = [index_start:index_end]    ## shuffle
                inputs = inputs_train[index_start:index_end]
                labels = labels_train[index_start:index_end]
                if torch.cuda.is_available():
                    inputs = inputs.to(args.device, non_blocking=True)
                    labels = labels.to(args.device, non_blocking=True)
                # Forward
                outputs = self.net(inputs)
                loss = self.loss_fun(outputs, labels).mean()
                # Backward and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_time = time.time() - epoch_start_time
            logging.info(f'Epoch: {epoch:^5} | Rank: {rank:^2} | Train loss: {train_temp_loss:.5f} | Valid loss: {valid_temp_loss:.5f} | Time: {epoch_time:.2f}s')
            
            # Save checkpoints and loss plots (only on rank 0)
            if epoch % 10 == 0:
                if rank == 0:
                    self.save_loss(train_loss, valid_loss, args)
                dist.barrier()
            
            if epoch % 100 == 0:
                if rank == 0:
                    self.save_model(args, norm_params, epoch)
                dist.barrier()
        
        # Save final model and loss (only on rank 0)
        if rank == 0:
            self.save_model(args, norm_params, args.max_epoch)
            self.save_loss(train_loss, valid_loss, args)
        dist.barrier()
        
        # Clean up
        dist.destroy_process_group()
        logging.info(f"Process {rank}: Training completed successfully")
    
    def get_loss(self, inputs_data, labels_data, device, eval_batch_size=None):
        """Calculate the full loss on a given dataset.
        
        Parameters
        ----------
        inputs_data : torch.Tensor
            The input features tensor containing the full (or chunked) dataset.
        labels_data : torch.Tensor
            The target labels tensor corresponding to the inputs.
        device : str
            CPU or GPU device, e.g., 'cuda:0', 'cuda:1', or 'cpu'.
        eval_batch_size : int, optional
            Batch size used to calculate the loss, different from training batch size.
            Default 5000.
            
        Returns
        -------
        loss_value : float
            The average loss on the given dataset.
        """
        if eval_batch_size is None:
            eval_batch_size = 5000
        total_loss = 0
        total_samples = 0
        
        self.net.eval()
        # inputs_data, labels_data = dataloader[:]
        batch_count = math.ceil(len(inputs_data) / eval_batch_size)
    
        with torch.no_grad():
            for step in range(batch_count):
                index_start = step * eval_batch_size
                index_end = min(index_start + eval_batch_size, inputs_data.__len__())
                inputs = inputs_data[index_start:index_end]
                labels = labels_data[index_start:index_end]
                if torch.cuda.is_available():
                    try:
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                    except Exception as e:
                        logging.error(f"ERROR occurs in get_loss(): {e}")
                        sys.exit(0)
                outputs = self.net(inputs)
                loss = self.loss_fun(outputs, labels)
                total_loss += loss.item() * eval_batch_size
                total_samples += eval_batch_size
        
        return total_loss / total_samples if total_samples > 0 else float('inf')

    def save_model(self, args, norm_params, epoch):
        """Save the model checkpoint, hyperparameters, and normalization.
        
        Parameters
        ----------
        args : argparse.Namespace
            Hyper-parameters of the model.
        norm_params : dict
            Mean and standard deviation for normalization.
        epoch : int
            Current epoch from the beginning of the training.
        """
        model_path = args.model_path
        checkpoint_dir = os.path.join(model_path, 'checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        ckpoint_path = os.path.join(checkpoint_dir, f'model{epoch}.pt')
        setting_path = os.path.join(checkpoint_dir, 'settings.json')
        norm_path = os.path.join(checkpoint_dir, 'norm.json')
        
        # Get state dict from model (handle DP and DDP cases)
        state_dict = self.net.module.state_dict() if args.use_ddp else self.net.state_dict()
        
        # Save model weights
        torch.save(state_dict, ckpoint_path)
        logging.info(f"Model saved to {ckpoint_path}")
        
        # Save settings and normalization parameters
        self.save_json(setting_path, args.__dict__)
        self.save_json(norm_path, norm_params)
    
    @staticmethod
    def save_json(json_path, data_dict):
        """Save the dictionary to JSON file.
        
        Parameters
        ----------
        json_path : str
            Path to save the JSON file.
        data_dict : dict
            Dictionary to save.
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                serializable_dict[key] = value.tolist()
            else:
                serializable_dict[key] = value
        
        with open(json_path, "w") as f:
            json.dump(serializable_dict, f, ensure_ascii=False, indent=4, separators=(',', ':'))
    
    def save_loss(self, train_loss, valid_loss, args):
        """Save training and validation losses and plot loss curves.
        
        Parameters
        ----------
        train_loss : list
            Training loss history.
        valid_loss : list
            Validation loss history.
        args : argparse.Namespace
            Hyper-parameters.
        """
        loss_path = os.path.join(args.model_path, 'lossfile')
        os.makedirs(loss_path, exist_ok=True)
        
        train_file = os.path.join(loss_path, 'train_loss.npy')
        valid_file = os.path.join(loss_path, 'valid_loss.npy')
        
        np.save(train_file, np.array(train_loss))
        np.save(valid_file, np.array(valid_loss))
        
        self.plot_loss(args)
        logging.info(f"Loss data saved to {loss_path}")
    
    @staticmethod
    def plot_loss(args, axis='semilogy', dpi=200):
        """Draw and save training and validation loss curves.
        
        Parameters
        ----------
        args : argparse.Namespace
            Hyper-parameters.
        axis : str, optional
            Plot function, could be 'semilogy' or 'loglog'. Default 'semilogy'.
        dpi : int, optional
            The dpi used to save figure. Default 200.
        """
        loss_path = os.path.join(args.model_path, 'lossfile')
        losspic_path = os.path.join(loss_path, f'{args.modelname}_loss.png')
        train_file = os.path.join(loss_path, 'train_loss.npy')
        valid_file = os.path.join(loss_path, 'valid_loss.npy')
        
        train_loss = np.load(train_file)
        valid_loss = np.load(valid_file)
        
        n_iters_train = len(train_loss)
        
        n_iters_valid = len(valid_loss)
        
        plt.figure(figsize=(10, 6))
        
        plot_methods = {'semilogy': plt.semilogy, 'loglog': plt.loglog}
        plot_handle = plot_methods[axis]
        
        p1, = plot_handle(range(1, n_iters_train + 1),
                          train_loss,
                          color="chocolate",
                          linewidth=2,
                          alpha=1)
        p2, = plot_handle(range(1, n_iters_train + 1),
                          valid_loss,
                          color="forestgreen",
                          linestyle='-.',
                          linewidth=2,
                          alpha=1)
        
        # Set y-axis limits based on loss values
        if len(train_loss) > 0:
            min_loss = min(np.min(train_loss), np.min(valid_loss))
            order = math.floor(np.log10(min_loss)) if min_loss > 0 else -1
            plt.ylim(10**order, 1)
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(f'{args.lossfun} Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend([p1, p2], ["Training Loss", 'Validation Loss'],
                   loc="upper right", fontsize=10)
        plt.title(f"Training Progress - {args.modelname}", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(losspic_path, dpi=dpi)
        plt.close()