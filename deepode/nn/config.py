import argparse
import os 


def parser():
    parser = argparse.ArgumentParser(description='Baseline for DeePODE project')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    # ========================= Gas Configs ===========================
    parser.add_argument('--mech_path', type=str, help='chemical mechanism file dir')
    parser.add_argument('--zero_input', nargs='+', default=["Ar", "He"], type=str, help='species with zero values in the input dataset.')
    parser.add_argument('--zero_gradient', nargs='+', default=[ ], type=str, help='species with zero rate of change')

    # ========================= Data Configs ==========================
    parser.add_argument('--dataset_type', type=str, default="chemical", help='dataset type: base/chemical/nuclear')
    parser.add_argument('--input_path', type=str, help='input dataset dir')
    parser.add_argument('--label_path', type=str, help='label dataset dir')
    parser.add_argument('--shuffle', action='store_false', help='shuffle the training dataset') # default true
    parser.add_argument('--batch_size', default=1024, type=int, help='use for training duration per worker')
    parser.add_argument('--valid_batch_size', default=8192, type=int, help="use for validation duration per worker")
    parser.add_argument('--train_size', type=int, help='training dataset size')
    parser.add_argument('--valid_size', type=int, help='validation dataset size')
    parser.add_argument('--valid_interval', default=10, type=int, help='validation interval in epochs')
    parser.add_argument('--valid_ratio', default=0.1, type=float, help='split percentages of training data as validation')
    parser.add_argument('--prefetch', default=10, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=16, type=int, help="num_workers for dataloaders")
    parser.add_argument('--pin_memory', action='store_false', help="pin_memory for DataLoader") #default True
    parser.add_argument('--power_transform', default=0.1, type=float, help='power_transform for BCT')
    parser.add_argument('--delta_t', default=1e-6, type=float, help='dnn time step')
    parser.add_argument('-GBCT', '-gbct', '--use_GBCT', action='store_true', help='whether use DistributedDataParallel') #default=False
    parser.add_argument('--lam', default=0.5, type=float, help='power_transform for GBCT')

    # ========================= Framework Configs =====================
    parser.add_argument('--dim', default=-1, type=int, help='feature dimention of data')
    parser.add_argument('--input_dim', default=-1, type=int, help='input dimention of data')
    parser.add_argument('--label_dim', default=-1, type=int, help='label dimention of data')
    parser.add_argument('-l', '--layers', nargs='+', default=[1600, 800, 400], type=int, help='dnn hidden layers')
    parser.add_argument('--net_type', default='fc', type=str, help='dnn type')
    parser.add_argument('--actfun', default='gelu', type=str, help='activation function')

    # ========================= Training Configs =======================
    # parser.add_argument('--model_name', default="toy_model", type=str, help='the default model name')
    parser.add_argument('-T', '--TRange', nargs='+', default=[200, 3600], type=float, help='temperature range')
    parser.add_argument('--max_epoch', default=5000, type=int, help='max epochs in training')
    parser.add_argument('--epoch_decay', default=2500, type=int, help='epoch interval for lr decay')
    parser.add_argument('-lr', '--learnrate', default=1e-4, type=float)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='learning rate decay')
    parser.add_argument('--batch_grow_rate', default=128, type=int, help='every [epoch_decay] batch_size grows')
    parser.add_argument('--lossfun', default='L1', type=str, help='loss funtion: MSE,L1/MAE,CEl')
    parser.add_argument('--optim', default='Adam', type=str, help='optimizer for training')
    parser.add_argument('--model_root', default="models", type=str, help='the default root dir to store models')
    parser.add_argument('--modelname', type=str, help='the dnn model name')
    parser.add_argument('--model_path', type=str, help='the dnn model dir')
    parser.add_argument('--device', default='cuda:0', type=str, help='device setup')

    # ========================= Multi-GPU Configs =======================
    parser.add_argument('--total_available_gpus', default=0, type=int, help='gpu numbers of the machine')
    parser.add_argument('--gpu_device_name', default="", type=str, help='gpu type')
    parser.add_argument('--cuda_version', default="", type=str, help='cuda version')
    parser.add_argument('--gpu_memory', default=0, type=float, help='gpu memory')

    # parser.add_argument('-DP', '--use_DP', action='store_true', help='whether use torch.nn.DataParallel') #default=False
    # DDP
    parser.add_argument('-ddp', '--use_ddp', action='store_true', help='whether use DistributedDataParallel') #default=False
    # DDP CUDA available device
    parser.add_argument('-cuda', '--cuda_ids', default="-1", type=str, help='when using DistributedDataParallel (DDP)') #default=False
    parser.add_argument('--master_host', default='127.0.0.1', type=str, help='master ip address of DDP')
    parser.add_argument('--master_port', default='12355', type=str, help='master ip address of DDP')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank')
    parser.add_argument('-ws','--world_size', default=1, type=int, help='world_size for DDP')
    parser.add_argument('--backend', default='nccl', type=str, help='current process backend for DDP, gloo,nccl,mpi')

    # ========================= Other Configs =======================
    parser.add_argument('-note', '--description', default='test', type=str, help='description of the experiment(purpose/target/motivation)')

    return parser
