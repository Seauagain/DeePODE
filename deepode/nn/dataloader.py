"""
"""

import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from abc import ABC, abstractmethod
import cantera as ct 
import os 
import time
import torch.distributed as dist
from .data_processor import DataProcessor



def gen_fake(eps=0):
    """Generate a fake dataset with 1000 rows and 22 columns."""
    num = 1000
    # Initialize an array of zeros
    arr = np.zeros((num, 22))
    # Set the first column to 1000 + eps
    arr[:, 0] = 1000 + eps
    # Set the second column to a sequence from 1 to num
    arr[:, 1] = np.arange(1, num + 1) + eps
    # Fill the remaining columns with random floats in [0, 1)
    arr[:, 2:] = np.random.rand(num, 20)
    return arr






class BaseDataset:
    """Base"""
    def __init__(self, args):
        
        self.input_path = args.input_path
        self.label_path = args.label_path

        if self.input_path and self.label_path:
            self.inputs = np.load(self.input_path)
            self.labels = np.load(self.label_path)
            assert len(self.inputs) == len(self.labels), "输入和标签的样本数不一致"

        # self.inputs = gen_fake(eps=0)
        # self.labels = gen_fake(eps=1)

        # self.inputs_dim = self.inputs.shape[0]
        # self.labels_dim = self.labels.shape[0]

        # Only execute update_info when no child class
        if self.__class__ == BaseDataset:
            self.update_info(args)

            
    
    def __len__(self):
        """return the size of dataset"""
        return len(self.inputs)


    def __getitem__(self, idx):
        """获取指定索引的数据项"""
        input_data = self.inputs[idx]
        label_data = self.labels[idx]

        # 转换为 float32 类型
        input_data = torch.tensor(input_data, dtype=torch.float32)
        label_data = torch.tensor(label_data, dtype=torch.float32)
        return input_data, label_data
    
    def update_info(self, args):
        """ """
        args.dim = self.inputs.shape[1]
        args.input_dim = self.inputs.shape[1]
        args.label_dim = self.labels.shape[1]
        args.layers = [self.inputs.shape[1]] + args.layers + [self.labels.shape[1]]



class ChemicalDataset(BaseDataset):

    def __init__(self, args):
        super().__init__(args)

        self.data_colum_keys = self.set_column_keys(args.mech_path)
        self.num_nonspecies = 2
        self.data_processor = DataProcessor()
        self.norm_params = {}

        ## transformation pipeline 
        self.transform(args)
        ## update args
        self.update_info(args)


    def set_column_keys(self, mech_path):
        gas = ct.Solution(mech_path)
        all_keys = ["T", "P"] + gas.species_names
        all_keys = list(map(str.upper, all_keys))
        return all_keys
        
    def transform(self, args):

        ## Box-Cox Transformation on species concentration
        self.inputs[:, self.num_nonspecies:] = self.data_processor.box_cox(self.inputs[:, self.num_nonspecies:], args.power_transform)
        self.labels[:, self.num_nonspecies:] = self.data_processor.box_cox(self.labels[:, self.num_nonspecies:], args.power_transform)
        self.labels = (self.labels - self.inputs) / args.delta_t

        ## whether use gbct
        if args.use_GBCT:
            self.labels = self.data_processor.generalized_box_cox(self.labels, args.lam) 
        
        ## normalization
        self.inputs, input_norm = self.data_processor.normalization(self.inputs, scale_type="std", eps=0)
        self.labels, label_norm = self.data_processor.normalization(self.labels, scale_type="mean")
        

        ## set the special species concentration
        for sp in args.zero_input:
            if sp.upper() in self.data_colum_keys:
                zero_idx = self.data_colum_keys.index(sp.upper())
                self.inputs[:, zero_idx] = 0

        for sp in args.zero_gradient + args.zero_input:
            if sp.upper() in self.data_colum_keys:
                zero_idx = self.data_colum_keys.index(sp.upper())
                self.labels[:, zero_idx] = 0
        
        ## save normalization params
        self.norm_params = {
            "input_mean": input_norm["mean"],
            "input_std": input_norm["scale"],
            "label_mean": label_norm["mean"],
            "label_std": label_norm["scale"]
        }



    def clean_data(self, args):
        # 处理无效数据（负值和超出温度范围的数据）
        T_low, T_upper = args.TRange
        # 定义需要过滤的条件
        invalid_conditions = [
            (input[:, 0] == 0),      # 温度为零
            (label[:, 0] == 0),      # 标签温度为零  
            (input[:, 0] > T_upper),  # 温度超过上限
            (input[:, 0] < T_low),    # 温度低于下限
            (input < 0),             # 输入有负值
            (label < 0)              # 标签有负值
        ]
        # 合并所有异常条件的索引
        invalid_indices = np.unique(
            np.concatenate([np.where(cond)[0] for cond in invalid_conditions])
        )
        # 删除无效数据
        if invalid_indices.size > 0:
            self.inputs = np.delete(self.inputs, invalid_indices, axis=0)
            self.labels = np.delete(self.labels, invalid_indices, axis=0)
            logging.info(
                f'Removed {invalid_indices.size} invalid samples '
                f'(T range: [{T_low}, {T_upper}])'
            )


def create_dataloaders(args, dataset=None, distributed=False,rank=None):
    """创建训练和验证数据加载器
    
    Parameters
    ----------
    args : argparse.Namespace
        参数配置
    dataset : Dataset, optional
        数据集实例，如果为None则创建新的数据集
    distributed : bool
        是否为分布式训练
        
    Returns
    -------
    train_loader : DataLoader
        训练数据加载器
    valid_loader : DataLoader
        验证数据加载器
    norm_params : dict
        标准化参数
    """
    if distributed:
        assert rank is not None, "分布式训练时需要提供rank参数"
        
    if dataset is None:
        dataset_type = args.dataset_type.lower() if hasattr(args, 'dataset_type') else 'chemical'
    
        if dataset_type == 'chemical':
            dataset = ChemicalDataset(args)
        elif dataset_type == 'nuclear':
            # dataset = ChemicalDataset(args)
            pass
        elif dataset_type == "air":
            pass 
        elif dataset_type == "base": 
            dataset = BaseDataset(args)
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    # 获取标准化参数
    norm_params = dataset.norm_params if hasattr(dataset, 'norm_params') else dict()
    
    # 计算训练集和验证集的大小
    dataset_size = len(dataset)
    valid_size = int(dataset_size * args.valid_ratio)
    train_size = dataset_size - valid_size

    args.train_size = train_size
    args.valid_size = valid_size
    
    
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size], 
        generator=torch.Generator().manual_seed(args.seed if hasattr(args, 'seed') else 42)
    )
    
    
    if distributed:
        train_len = len(train_dataset)
        valid_len = len(valid_dataset)
        train_per_rank = train_len // args.world_size
        valid_per_rank = valid_len // args.world_size
        train_start = rank * train_per_rank
        train_end = train_start + train_per_rank if rank != args.world_size - 1 else train_len
        valid_start = rank * valid_per_rank
        valid_end = valid_start + valid_per_rank if rank != args.world_size - 1 else valid_len
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(train_start, train_end)))
        valid_dataset = torch.utils.data.Subset(valid_dataset, list(range(valid_start, valid_end)))
     
    return train_dataset, valid_dataset, norm_params



