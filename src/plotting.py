
# ------------------------------------------TranAD------------------------------------------

# # 导入必要的库
# import matplotlib.pyplot as plt  # 绘图基础库
# from matplotlib.backends.backend_pdf import PdfPages  # 用于生成PDF格式的图表
# import statistics  # 统计相关工具（此处未直接使用，可能为预留）
# import os, torch  # os用于文件操作，torch用于张量处理
# import numpy as np  # 数值计算库
#
# # 设置绘图样式
# plt.style.use(['default'])  # 使用默认样式
# plt.rcParams["text.usetex"] = False  # 禁用LaTeX渲染（加快绘图速度）
# plt.rcParams['figure.figsize'] = 6, 2  # 设置默认图表尺寸（宽6，高2）
#
# # 创建保存图表的文件夹
# os.makedirs('plots', exist_ok=True)  # 若文件夹已存在则不报错
#
#
# def smooth(y, box_pts=1):
#     """
#     对数据进行平滑处理（滑动平均），减少噪声干扰
#     :param y: 原始数据（1D数组）
#     :param box_pts: 滑动窗口大小（默认1，即不平滑）
#     :return: 平滑后的数据
#     """
#     box = np.ones(box_pts) / box_pts  # 生成平均权重窗口（如窗口大小为3时，权重为[1/3,1/3,1/3]）
#     y_smooth = np.convolve(y, box, mode='same')  # 卷积计算滑动平均，'same'保证输出与输入长度相同
#     return y_smooth
#
#
# def plotter(name, y_true, y_pred, ascore, labels):
#     """
#     生成并保存多维度时间序列的对比图表（真实值、预测值、异常标签、异常分数）
#     :param name: 图表保存文件夹名称（格式为"模型名_数据集名"）
#     :param y_true: 真实值张量（形状为[时间步, 特征维度]）
#     :param y_pred: 模型预测值张量（形状同上）
#     :param ascore: 异常分数张量（形状同上，值越高越可能是异常）
#     :param labels: 真实异常标签张量（形状同上，1表示异常，0表示正常）
#     """
#     # 针对TranAD模型的特殊处理：真实值时间步向前滚动1位（因TranAD预测滞后1步）
#     if 'TranAD' in name:
#         y_true = torch.roll(y_true, 1, 0)  # 沿第0维度（时间维度）滚动，使真实值与预测值时间对齐
#
#     # 创建保存当前模型+数据集图表的文件夹
#     os.makedirs(os.path.join('plots', name), exist_ok=True)
#     # 创建PDF文件对象，用于保存所有图表
#     pdf = PdfPages(f'plots/{name}/output.pdf')
#
#     # 遍历每个特征维度，分别绘图
#     for dim in range(y_true.shape[1]):
#         # 提取当前维度的真实值、预测值、异常标签、异常分数
#         y_t = y_true[:, dim]  # 真实值序列
#         y_p = y_pred[:, dim]  # 预测值序列
#         l = labels[:, dim]    # 异常标签序列
#         a_s = ascore[:, dim]  # 异常分数序列
#
#         # 创建包含两个子图的画布（上下布局，共享x轴）
#         fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#
#         # 上方子图：绘制真实值与预测值对比，叠加异常标签
#         ax1.set_ylabel('Value')  # y轴标签
#         ax1.set_title(f'Dimension = {dim}')  # 标题标注当前特征维度
#         # 绘制平滑后的真实值（线宽0.2，标签为'True'）
#         ax1.plot(smooth(y_t), linewidth=0.2, label='True')
#         # 绘制平滑后的预测值（线宽0.3，透明度0.6，标签为'Predicted'）
#         ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
#         # 创建右侧共享y轴的子图，用于绘制异常标签
#         ax3 = ax1.twinx()
#         # 绘制异常标签（虚线，线宽0.3，透明度0.5）
#         ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
#         # 对异常标签区域填充蓝色（透明度0.3），突出显示异常区间
#         ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
#         # 仅在第0维度添加图例（避免重复），位置在右上角
#         if dim == 0:
#             ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
#
#         # 下方子图：绘制异常分数
#         ax2.plot(smooth(a_s), linewidth=0.2, color='g')  # 绿色线，线宽0.2
#         ax2.set_xlabel('Timestamp')  # x轴标签（时间步）
#         ax2.set_ylabel('Anomaly Score')  # y轴标签（异常分数）
#
#         # 将当前图表保存到PDF
#         pdf.savefig(fig)
#         # 关闭当前图表，释放内存
#         plt.close()
#
#     # 关闭PDF文件，完成保存
#     pdf.close()


#-------------------------------------------DTAAD----------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os
import torch
import numpy as np

# 设置 matplotlib 绘图样式
plt.style.use(['default'])
# 禁用 LaTeX 渲染（避免依赖问题）
plt.rcParams["text.usetex"] = False
# 设置默认图片大小
plt.rcParams['figure.figsize'] = 6, 2

# 创建输出目录
os.makedirs('plots', exist_ok=True)

# 移动平均滤波实现平滑，减少绘图中噪声的干扰
def smooth(y, box_pts=1):
    """
    对信号进行平滑处理（移动平均）
    :param y: 输入信号
    :param box_pts: 窗口大小
    :return: 平滑后的信号
    """
    box = np.ones(box_pts) / box_pts  # 创建归一化的窗口
    y_smooth = np.convolve(y, box, mode='same')  # 卷积实现移动平均
    return y_smooth


def plotter(name, y_true, y_pred, ascore, labels):
    """
    生成异常检测结果可视化图表
    :param name: 模型名称（用于生成文件夹）
    :param y_true: 真实值 (时间序列)
    :param y_pred: 预测值
    :param ascore: 异常分数
    :param labels: 真实异常标签
    """
    # 对 TranAD/DTAAD 模型结果进行时间对齐（窗口预测需要回滚一位）
    if 'TranAD' in name or 'DTAAD' in name:
        y_true = torch.roll(y_true, 1, 0)

    # 创建模型专属目录
    os.makedirs(os.path.join('plots', name), exist_ok=True)

    # 创建 PDF 文件用于保存所有图表
    pdf = PdfPages(f'plots/{name}/output.pdf')

    # 遍历每个特征维度 y_pred: (28479,38) y_true: (28479,38)   ascore即每个位置的loss (28479,38)
    for dim in range(y_true.shape[1]):
        # 提取当前维度的数据
        y_t = y_true[:, dim]  # 真实值
        y_p = y_pred[:, dim]  # 预测值
        l = labels[:, dim]  # 真实异常标签
        a_s = ascore[:, dim]  # 异常分数

        # 创建子图（2行1列，共享X轴）
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # 配置第一个子图（真实值 vs 预测值）
        # 绘制平滑后的真实值（蓝色细线）和预测值（浅色实线），使用smooth函数减少噪声。
        # 通过ax3 = ax1.twinx()创建双 Y 轴，叠加绘制异常标签（虚线）并填充蓝色半透明区域，标记真实异常区间。
        # 仅在第 0 维度添加图例，显示 "True" 和 "Predicted"。
        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim}')
        ax1.plot(smooth(y_t), linewidth=0.2, label='True')  # 绘制真实值（平滑后）
        ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')  # 绘制预测值（平滑后）

        # 创建右侧Y轴（用于绘制异常标签）
        ax3 = ax1.twinx()
        # 绘制真实异常区间（蓝色填充）
        # ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3, label='True Anomaly')

        # 只在第一个维度显示图例
        # 并排放置 ncol=2
        # bbox_to_anchor=(0.6, 1.02)调整图例位置
        if dim == 0:
            ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))

        # 配置第二个子图（异常分数 vs 预测异常）
        ax2.plot(smooth(a_s), linewidth=0.2, color='g', label='Score')  # 绘制异常分数（平滑后）

        # 创建右侧Y轴（用于绘制预测异常）
        ax4 = ax2.twinx()
        # 绘制预测异常区间（红色填充）
        ax4.fill_between(np.arange(l.shape[0]), l, color='red', alpha=0.3, label='Predicted Anomaly')

        # 只在第一个维度显示图例
        if dim == 0:
            ax4.legend(bbox_to_anchor=(1, 1.02))

        # 配置X轴标签
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')

        # 隐藏Y轴刻度（专注于趋势展示）
        ax1.set_yticks([])
        ax2.set_yticks([])

        # 保存当前图表到PDF
        pdf.savefig(fig)
        plt.close()  # 关闭图表释放内存

    # 关闭PDF文件
    pdf.close()