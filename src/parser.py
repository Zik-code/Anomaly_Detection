import argparse

# 创建参数解析器对象描述程序
parser = argparse.ArgumentParser(description='Time-Series Data Anomaly Detection ')
# 添加命令行参数
parser.add_argument('--dataset',
					metavar='-d',
					type=str, # 命令行参数应当被转为的类型
					required=False,  # 命令行参数是否可忽略
					default='SMD',
                    help="dataset from ['SMD']")
parser.add_argument('--model',
					metavar='-m',
					type=str,
					required=False,
					default='TranAD',
                    help="model name")
parser.add_argument('--test',
					action='store_true',
					help="test the model")
parser.add_argument('--retrain',
					action='store_true',
					help="retrain the model")
parser.add_argument('--less',
					action='store_true',
					help="train using less data")

# 解析命令行参数
args = parser.parse_args()

# 使用SMD数据集、DAGMM模型，启用测试模式
# python main.py --dataset SMD --model DAGMM --test

# 使用合成数据集、默认模型，强制重新训练
# python main.py --retrain

# 使用SMD数据集、较少数据进行训练
# python main.py --dataset SMD --less