import os
import sys
import numpy as np



def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(
        os.path.join(dataset_folder, category, filename),
        dtype=np.float64,
        delimiter=','
    )
    print(f"SMD数据集 - {dataset} {category} 数据形状: {temp.shape}")
    output_path = os.path.join(processed_data_folder, "SMD")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{dataset}_{category}.npy")
    np.save(output_file, temp)
    print(f"已保存: {output_file}")
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        lines = f.readlines()
    for line in lines:
        pos_str, dims_str = line.split(':')[0], line.split(':')[1].split(',')
        start, end = int(pos_str.split('-')[0]), int(pos_str.split('-')[1])
        dims = [int(dim) - 1 for dim in dims_str]
        temp[start-1:end-1, dims] = 1
    output_path = os.path.join(processed_data_folder, "SMD")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{dataset}_{category}.npy")
    np.save(output_file, temp)
    print(f"已保存: {output_file}")

def load_data_smd():
    dataset_folder = "./data/SMD"
    os.makedirs(os.path.join(processed_data_folder, "SMD"), exist_ok=True)
    train_dir = os.path.join(dataset_folder, "train")
    file_list = os.listdir(train_dir)
    print(f"找到 {len(file_list)} 个训练数据文件")
    for filename in file_list:
        if filename.endswith('.txt'):
            dataset_name = filename.strip('.txt')
            load_and_save('train', filename, dataset_name, dataset_folder)
            test_shape = load_and_save('test', filename, dataset_name, dataset_folder)
            load_and_save2('labels', filename, dataset_name, dataset_folder, test_shape)

if __name__ == '__main__':
    print("开始处理SMD数据集...")
    load_data_smd()
    print(f"SMD数据集预处理完成，文件已保存到: {os.path.abspath(os.path.join(processed_data_folder, 'SMD'))}")