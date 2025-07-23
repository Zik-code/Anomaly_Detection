# 导入必要的库
import matplotlib.pyplot as plt  # 绘图基础库
from matplotlib.backends.backend_pdf import PdfPages  # 用于生成PDF格式的图表
import statistics  # 统计相关工具（此处未直接使用，可能为预留）
import os, torch  # os用于文件操作，torch用于张量处理
import numpy as np  # 数值计算库

# 设置绘图样式
plt.style.use(['default'])  # 使用默认样式
plt.rcParams["text.usetex"] = False  # 禁用LaTeX渲染（加快绘图速度）
plt.rcParams['figure.figsize'] = 6, 2  # 设置默认图表尺寸（宽6，高2）

# 创建保存图表的文件夹
os.makedirs('plots', exist_ok=True)  # 若文件夹已存在则不报错


def smooth(y, box_pts=1):
    """
    对数据进行平滑处理（滑动平均），减少噪声干扰
    :param y: 原始数据（1D数组）
    :param box_pts: 滑动窗口大小（默认1，即不平滑）
    :return: 平滑后的数据
    """
    box = np.ones(box_pts) / box_pts  # 生成平均权重窗口（如窗口大小为3时，权重为[1/3,1/3,1/3]）
    y_smooth = np.convolve(y, box, mode='same')  # 卷积计算滑动平均，'same'保证输出与输入长度相同
    return y_smooth


def plotter(name, y_true, y_pred, ascore, labels):
    """
    生成并保存多维度时间序列的对比图表（真实值、预测值、异常标签、异常分数）
    :param name: 图表保存文件夹名称（格式为"模型名_数据集名"）
    :param y_true: 真实值张量（形状为[时间步, 特征维度]）
    :param y_pred: 模型预测值张量（形状同上）
    :param ascore: 异常分数张量（形状同上，值越高越可能是异常）
    :param labels: 真实异常标签张量（形状同上，1表示异常，0表示正常）
    """
    # 针对TranAD模型的特殊处理：真实值时间步向前滚动1位（因TranAD预测滞后1步）
    if 'TranAD' in name:
        y_true = torch.roll(y_true, 1, 0)  # 沿第0维度（时间维度）滚动，使真实值与预测值时间对齐

    # 创建保存当前模型+数据集图表的文件夹
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    # 创建PDF文件对象，用于保存所有图表
    pdf = PdfPages(f'plots/{name}/output.pdf')

    # 遍历每个特征维度，分别绘图
    for dim in range(y_true.shape[1]):
        # 提取当前维度的真实值、预测值、异常标签、异常分数
        y_t = y_true[:, dim]  # 真实值序列
        y_p = y_pred[:, dim]  # 预测值序列
        l = labels[:, dim]    # 异常标签序列
        a_s = ascore[:, dim]  # 异常分数序列

        # 创建包含两个子图的画布（上下布局，共享x轴）
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # 上方子图：绘制真实值与预测值对比，叠加异常标签
        ax1.set_ylabel('Value')  # y轴标签
        ax1.set_title(f'Dimension = {dim}')  # 标题标注当前特征维度
        # 绘制平滑后的真实值（线宽0.2，标签为'True'）
        ax1.plot(smooth(y_t), linewidth=0.2, label='True')
        # 绘制平滑后的预测值（线宽0.3，透明度0.6，标签为'Predicted'）
        ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
        # 创建右侧共享y轴的子图，用于绘制异常标签
        ax3 = ax1.twinx()
        # 绘制异常标签（虚线，线宽0.3，透明度0.5）
        ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
        # 对异常标签区域填充蓝色（透明度0.3），突出显示异常区间
        ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
        # 仅在第0维度添加图例（避免重复），位置在右上角
        if dim == 0:
            ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))

        # 下方子图：绘制异常分数
        ax2.plot(smooth(a_s), linewidth=0.2, color='g')  # 绿色线，线宽0.2
        ax2.set_xlabel('Timestamp')  # x轴标签（时间步）
        ax2.set_ylabel('Anomaly Score')  # y轴标签（异常分数）

        # 将当前图表保存到PDF
        pdf.savefig(fig)
        # 关闭当前图表，释放内存
        plt.close()

    # 关闭PDF文件，完成保存
    pdf.close()