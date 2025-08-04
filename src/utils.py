import matplotlib.pyplot as plt  # 导入matplotlib绘图库，用于数据可视化
import os  # 导入os模块，用于文件和目录操作
import seaborn  # 导入seaborn库，用于绘制更美观的热图等统计图表


# 定义颜色类，用于在终端输出中添加颜色和样式（增强可读性）
class color:
    HEADER = '\033[95m'  # 紫色，用于标题
    BLUE = '\033[94m'  # 蓝色，用于普通信息
    GREEN = '\033[92m'  # 绿色，用于成功信息
    RED = '\033[93m'  # 黄色，用于警告信息
    FAIL = '\033[91m'  # 红色，用于错误信息
    ENDC = '\033[0m'  # 重置颜色和样式
    BOLD = '\033[1m'  # 加粗文本
    UNDERLINE = '\033[4m'  # 下划线文本


def plot_accuracies(accuracy_list, folder):
    """
    绘制训练过程中的平均损失和学习率曲线，并保存为PDF文件

    参数:
        accuracy_list: 包含训练损失和学习率的列表，每个元素为元组 (loss, lr)
        folder: 保存图表的文件夹路径（位于plots目录下）
    """
    # 创建保存图表的目录（若不存在）
    os.makedirs(f'plots/{folder}/', exist_ok=True)

    # 从列表中提取训练损失和学习率
    trainAcc = [i[0] for i in accuracy_list]  # 平均训练损失
    lrs = [i[1] for i in accuracy_list]  # 学习率

    # 绘制平均训练损失曲线（左侧Y轴）
    plt.xlabel('Epochs')  # X轴标签：训练轮次
    plt.ylabel('Average Training Loss')  # 左侧Y轴标签：平均训练损失
    plt.plot(
        range(len(trainAcc)),  # X轴数据：轮次索引
        trainAcc,
        label='Average Training Loss',  # 图例标签
        linewidth=1,  # 线宽
        linestyle='-',  # 实线
        marker='.'  # 点标记
    )

    # 创建右侧Y轴，用于绘制学习率曲线
    plt.twinx()
    plt.plot(
        range(len(lrs)),  # X轴数据：轮次索引
        lrs,
        label='Learning Rate',  # 图例标签
        color='r',  # 红色曲线
        linewidth=1,  # 线宽
        linestyle='--',  # 虚线
        marker='.'  # 点标记
    )

    # 保存图表到指定路径
    plt.savefig(f'plots/{folder}/training-graph.pdf')
    plt.clf()  # 清除当前图表，避免后续绘图干扰


def plot_attention(model, layers, folder):
    """
    绘制Transformer模型各层的注意力权重热图，并保存为图片

    参数:
        model: 训练好的Transformer模型（包含注意力权重数据）
        layers: 需要可视化的注意力层数
        folder: 保存图表的文件夹路径（位于plots目录下）
    """
    # 创建保存图表的目录（若不存在）
    os.makedirs(f'plots/{folder}/', exist_ok=True)

    # 遍历每一层注意力
    for layer in range(layers):
        # 创建1行2列的子图（用于对比局部和全局注意力）
        fig, (axs, axs1) = plt.subplots(1, 2, figsize=(10, 4))

        # 绘制局部注意力权重热图
        heatmap = seaborn.heatmap(
            model.transformer_encoder1.layers[layer].att[0].data.cpu(),  # 从模型中提取局部注意力数据（转至CPU）
            ax=axs  # 指定子图
        )
        heatmap.set_title("Local_attention", fontsize=10)  # 设置子图标题

        # 绘制全局注意力权重热图
        heatmap = seaborn.heatmap(
            model.transformer_encoder2.layers[layer].att[0].data.cpu(),  # 从模型中提取全局注意力数据（转至CPU）
            ax=axs1  # 指定子图
        )
        heatmap.set_title("Global_attention", fontsize=10)  # 设置子图标题

    # 保存热图到指定路径
    heatmap.get_figure().savefig(f'plots/{folder}/attention-score.png')
    plt.clf()  # 清除当前图表


def cut_array(percentage, arr):
    """
    按指定比例截取数组的中间部分（用于缩减数据集大小）

    参数:
        percentage: 保留数据的比例（0-1之间）
        arr: 输入的numpy数组（形状为 [N, D]，N为样本数，D为特征数）

    返回:
        截取后的数组（保留中间部分，长度为 N * percentage）
    """
    # 打印截取信息（加粗样式）
    print(f'{color.BOLD}Slicing dataset to {int(percentage * 100)}%{color.ENDC}')

    # 计算中间位置和截取窗口大小
    mid = round(arr.shape[0] / 2)  # 数组中间索引
    window = round(arr.shape[0] * percentage * 0.5)  # 从中间向两边扩展的窗口大小

    # 截取中间窗口内的数据并返回
    return arr[mid - window: mid + window, :]


def getresults2(df, result):
    """
    从评估结果DataFrame中计算关键指标（精确率、召回率、F1分数等）

    参数:
        df: 包含每个样本/批次评估结果的DataFrame，列包括 'FN', 'FP', 'TP', 'TN', 'precision', 'recall'
        result: （未使用）预留参数，可能用于扩展

    返回:
        results2: 包含汇总指标的字典，包括总FN/FP/TP/TN、平均精确率/召回率和F1分数
    """
    results2 = {}  # 存储结果的字典
    df1 = df.sum()  # 对所有行求和（计算总FN、FP、TP、TN）
    df2 = df.mean()  # 对所有行求平均（计算平均精确率、召回率）

    # 提取总假阴性、假阳性、真阳性、真阴性
    for a in ['FN', 'FP', 'TP', 'TN']:
        results2[a] = df1[a]

    # 提取平均精确率和召回率
    for a in ['precision', 'recall']:
        results2[a] = df2[a]

    # 计算F1分数（ harmonic mean of precision and recall）
    results2['f1*'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])

    return results2