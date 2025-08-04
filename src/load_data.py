from src.folderconstants import processed_data_folder
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset  # PyTorch数据加载工具
import pandas as pd
import  numpy as np
from src.utils import *
import torch.nn as nn
import torch
from time import time
from pprint import pprint

def load_dataset(dataset):
	folder = os.path.join(processed_data_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'A-3_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	# if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

# def load_dataset(dataset, prefixes):
# 	# 初始化全局数据变量
# 	global data_train, data_test, labels
#
# 	for prefix in prefixes:
# 		if dataset == 'SMD':
# 			file = f'machine-1-{prefix}_train.npy'
# 			test_file = f'machine-1-{prefix}_test.npy'
# 			label_file = f'machine-1-{prefix}_labels.npy'
# 		elif dataset == 'SMAP':
# 			file = f'P-1_machine-1-{prefix}_train.npy'
# 			test_file = f'P-1_machine-1-{prefix}_test.npy'
# 			label_file = f'P-1_machine-1-{prefix}_labels.npy'
# 		# 其他数据集的处理方式类似
#
# 		# 加载训练数据
# 		data_train_prefix = np.load(os.path.join(processed_data_folder, file))
# 		if 'train' in locals():
# 			data_train = np.concatenate([data_train, data_train_prefix], axis=0)
# 		else:
# 			data_train = data_train_prefix
#
# 		# 加载测试数据
# 		test_data_prefix = np.load(os.path.join(processed_data_folder, test_file))
# 		if 'test' in locals
# 			data_test = np.concatenate([data_test, test_data_prefix], axis=0)
# 		else:
# 			data_test = test_data_prefix
#
# 		# 加载标签
# 		labels_prefix = np.load(os.path.join(processed_data_folder, label_file))
# 		if 'labels' in locals():
# 			labels = np.concatenate([labels, labels_prefix], axis=0)
# 		else:
# 			labels = labels_prefix
#
# 	# 转换为张量并返回
# 	train_loader = DataLoader(torch.from_numpy(data_train), batch_size=data_train.shape[0])
# 	test_loader = DataLoader(torch.from_numpy(data_test), batch_size=data_test.shape[0])
# 	labels = torch.from_numpy(labels)
#
# 	return train_loader, test_loader, labels

def convert_to_windows(data, model,args):
    """
    将时序数据转换为滑动窗口格式（适配时序异常检测模型的输入要求）
    参数:
        data: 原始时序数据，形状为[时间步, 特征数]
        model: 模型实例，用于获取窗口大小（model.n_window）
    返回:
        windows: 转换后的窗口数据，形状为[窗口数, 窗口大小, 特征数]
    """
    windows = []
    w_size = model.n_window  # 窗口大小由模型定义,TranAD为10
    for i, g in enumerate(data): # (data为test.train)
		# i索引增加到w_size大小时满足
        if i >= w_size:
            w = data[i - w_size:i]  # 取[i-w_size, i)区间的数据作为一个窗口
		# 前10行不足一个窗口
        else:
            # 前w_size个窗口：用首元素重复(w_size-i)次，再拼接前i个元素
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        # 对TranAD或Attention模型，窗口保持原形状；其他模型展平窗口为一维特征
        windows.append(w if 'DTAAD' in args.model or 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    return torch.stack(windows)  # 堆叠所有窗口为张量

# 举例
# 1. 当 i < w_size（即 i=0,1,2,3,4，前 5 个时间步）
# 此时窗口大小不足 5，需要用 data[0]（第一个时间步的数据）填充前面的缺失部分，再拼接当前已有的时间步数据。
#
# i=0 时：
# w_size - i = 5 - 0 = 5 → 需要用 data[0] 重复 5 次
# data[0:i] = data[0:0] → 空张量（没有数据）
# 窗口 w = torch.cat([data[0].repeat(5, 1), data[0:0]])
# 结果：窗口包含 data[0] 重复 5 次，形状为 (5, 38)
# 内容：[data[0], data[0], data[0], data[0], data[0]]
# i=1 时：
# w_size - i = 5 - 1 = 4 → 用 data[0] 重复 4 次
# data[0:i] = data[0:1] → 包含 data[0]（1 行）
# 窗口 w = torch.cat([data[0].repeat(4, 1), data[0:1]])
# 结果：总长度 4 + 1 = 5，形状 (5, 38)
# 内容：[data[0], data[0], data[0], data[0], data[0]]（注意：i=1 时，实际只有 data[0] 一个有效时间步，其余用 data[0] 填充）

# 当满足窗口大小时
# i =5,取0-4的数据
# i =6,取1-5的数据
# ·········