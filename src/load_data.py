from src.folderconstants import processed_data_folder
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset  # PyTorch数据加载工具
import pandas as pd
import  numpy as np
from src.utils import *
def load_dataset(dataset):
	folder = os.path.join(processed_data_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
    # loader里面存了三个二维数组，machine-1-1_train.npy machine-1-1_test.npy machine-1-1_labels.npy
	if args.less: loader[0] = cut_array(0.2, loader[0])
    # batch_size一个批次是一台机器的28439次检测
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels