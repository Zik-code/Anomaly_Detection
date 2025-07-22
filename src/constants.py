from src.parser import *
from src.folderconstants import *

# Threshold parameters
lm_d = {
# 每个数据集对应一个包含两个元组的列表，每个元组是(level参数, 缩放系数)：
# level参数：用于SPOT算法的initialize方法，控制初始阈值的分位数水平（值越接近 1，初始阈值越严格）。
# 缩放系数：对SPOT计算的阈值进行最终缩放（例如1.04表示将阈值乘以 1.04，让异常判定更严格）。
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)], #
		# 'SWaT': [(0.993, 1), (0.993, 1)],
		# 'NAB': [(0.991, 1), (0.99, 1)],
		# 'SMAP': [(0.98, 1), (0.98, 1)],
		# 'MSL': [(0.97, 1), (0.999, 1.04)],
	}
lm = lm_d[args.dataset][1 if 'TranAD' or 'DTTAD' in args.model else 0]

# Hyperparameters
lr_d = {
		'SMD': 0.0001,
		# 'SWaT': 0.008,
		# 'SMAP': 0.001,
		# 'MSL': 0.002,
		# 'NAB': 0.009,
	}
lr = lr_d[args.dataset]

# Debugging
percentiles = {
		'SMD': (98, 2000),
		# 'SMAP': (97, 5000),
		# 'MSL': (97, 150),
		# 'WADI': (99, 1200),
		# 'NAB': (98, 2),
	}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9