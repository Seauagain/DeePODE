# -*- coding: utf-8 -*-
"""
数据处理类，包含预处理和逆变换功能。

Created on 2023
"""

import numpy as np
import torch
import cantera as ct
import os
import json


class DataProcessor:
    """
    数据处理类，提供数据预处理和逆变换功能。
    """
    
    def __init__(self, args=None):
        """
        初始化数据处理器。
        
        Parameters
        ----------
        args : argparse.Namespace or dict, optional
            配置参数，包含power_transform, delta_t等。
        """
        self.args = args
        self.norm = None
        self.gas = None
    
    @staticmethod
    def box_cox(data, bct_lambda):
        """bct"""
        return ( data**(bct_lambda) -1) / (bct_lambda)
    
    @staticmethod
    def inverse_box_cox(data, bct_lambda):
        """inverse bct"""
        return ( bct_lambda * data + 1 ) ** (1/bct_lambda)
    
    @staticmethod
    def generalized_box_cox(data, gbct_lambda):
        data_sign = np.sign(data)
        data_sign[data_sign==0] = 1 
        return (1/gbct_lambda) * data_sign * ( np.abs(data) )**gbct_lambda
    
    @staticmethod
    def inverse_generalized_box_cox(data, gbct_lambda):
        data_sign = np.sign(data)
        data_sign[data_sign==0] = 1 
        return data_sign * ( gbct_lambda * np.abs(data) )**(1 / gbct_lambda)

    @staticmethod
    def normalization(data, scale_type="std", eps=1e-20):
        mean = np.mean(data, axis=0)
        if scale_type == "std":
            std = np.std(data, axis=0, ddof=1) + eps
            normalized_data = (data - mean) / std
            scale_factor = std
        elif scale_type == "mean":
            mean_abs = np.mean(np.abs(data), axis=0) + eps
            normalized_data = (data - mean) / mean_abs
            scale_factor = mean_abs
        else:
            raise ValueError(f"not supported scaling type: {scale}")
            
        return normalized_data, {"mean": mean, "scale": scale_factor}
    
    @staticmethod
    def inverse_normalization(data):
        return data * self.norm.label_std + self.norm.label_mean
