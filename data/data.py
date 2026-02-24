#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-16 19:22:41
LastEditTime: 2021-01-19 20:55:10
@Description: file content
'''
from os.path import join
from torchvision.transforms import Compose, ToTensor
from .dataset import Data, Data_test, Data_eval, H5PanSharpeningDataset
from torchvision import transforms
import torch, numpy as np  #h5py,
import torch.utils.data as data

class SmartToTensor:
    """智能tensor转换，uint16数据按QB数据集处理"""

    def __call__(self, img):
        # 获取图像的基本信息
        img_array = np.array(img)

        # 确定数据类型和对应的最大值，转换为float32后进行归一化
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        elif img_array.dtype == np.uint16 or img_array.dtype == np.float32:
            # 自动探测位深，比死磕 2047 更安全
            current_max = img_array.max()
            img_array = img_array.astype(np.float32)  # 先转换为float32
            if current_max > 2047:
                # 可能是 12-bit 或没对齐的 16-bit
                img_array /= 65535.0
            else:
                # 针对 QB 的 11-bit 归一化
                img_array /= 2047.0
        else:
            # 其他数据类型，直接转换为float32
            img_array = img_array.astype(np.float32)

        # 维度转换 (HWC -> CHW)，处理不同通道数的图像(多通道 (H, W, C)或单通道 (H, W))
        if img_array.ndim == 3: # 多通道 (H, W, C)
            img_array = img_array.transpose(2, 0, 1) # HWC -> CHW
        elif img_array.ndim == 2: # 单通道 (H, W)
            img_array = img_array[np.newaxis, ...]# 添加通道维度 (C, H, W)

        return torch.from_numpy(img_array)

def transform():
    return Compose([
        SmartToTensor(),  # 智能tensor转换，uint16按QB数据集处理
    ])

def get_data(cfg, mode):
    data_path = cfg['data_dir_train']  # 或你也可以写 cfg['data']['data_dir_train']
    if data_path.endswith('.h5'):
        return H5PanSharpeningDataset(data_path, cfg)
    else:
        data_dir_ms = join(mode, cfg['source_ms'])
        data_dir_pan = join(mode, cfg['source_pan'])
        data_dir_mask = join(mode, "mask")
        return Data(data_dir_ms, data_dir_pan, cfg, transform=transform(), data_dir_mask=data_dir_mask)
    
# def get_data(cfg, mode):
#     data_dir_ms = join(mode, cfg['source_ms'])
#     data_dir_pan = join(mode, cfg['source_pan'])
#     data_dir_mask = join(mode,"mask")
#     cfg = cfg
#     return Data(data_dir_ms, data_dir_pan, cfg, transform=transform(),data_dir_mask=data_dir_mask)
    
def get_test_data(cfg, mode):
    # 优先使用 data_dir_test（如果存在），否则回退到 data_dir_eval
    data_path = cfg.get('data_dir_test') or cfg['data_dir_eval']
    if data_path.endswith('.h5'):
        return H5PanSharpeningDataset(data_path, cfg)
    else:
        data_dir_ms = join(mode, cfg['test']['source_ms'])
        data_dir_pan = join(mode, cfg['test']['source_pan'])
        data_dir_mask = join(mode, "mask")
        cfg = cfg
        return Data_test(data_dir_ms, data_dir_pan, cfg, transform=transform(),data_dir_mask=data_dir_mask)

    
def get_eval_data(cfg, data_dir, upscale_factor):
    data_path = cfg['data_dir_test']  # 或你也可以写 cfg['data']['data_dir_train']
    if data_path.endswith('.h5'):
        return H5PanSharpeningDataset(data_path, cfg)
    else:
        data_dir_ms = join(data_dir, cfg['test']['source_ms'])
        data_dir_pan = join(data_dir, cfg['test']['source_pan'])
        cfg = cfg
        return Data_eval(data_dir_ms, data_dir_pan, cfg, transform=transform())