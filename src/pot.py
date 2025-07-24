import numpy as np
from src.spot import SPOT  # 导入POT算法实现（用于异常阈值计算）
from src.constants import *  # 导入常量（如lm，用于阈值调整参数）
from sklearn.metrics import roc_auc_score  # 导入ROC AUC指标计算


def calc_point2point(predict, actual):
    """
    计算二分类基础指标（基于混淆矩阵）
    Args:
        predict (np.ndarray): 模型预测的异常标记数组（0表示正常，1表示异常）
        actual (np.ndarray): 真实异常标签数组（0表示正常，1表示异常）
    Returns:
        tuple: 包含F1分数、精确率、召回率、TP、TN、FP、FN、AUC的元组
    """
    # 计算混淆矩阵元素
    TP = np.sum(predict * actual)  # 真正例：预测异常且实际异常
    TN = np.sum((1 - predict) * (1 - actual))  # 真负例：预测正常且实际正常
    FP = np.sum(predict * (1 - actual))  # 假正例：预测异常但实际正常（误报）
    FN = np.sum((1 - predict) * actual)  # 假负例：预测正常但实际异常（漏报）

    # 计算精确率（预测为异常的样本中，真正异常的比例）
    precision = TP / (TP + FP + 0.00001)  # 加微小值避免除零错误
    # 计算召回率（实际异常的样本中，被正确预测的比例）
    recall = TP / (TP + FN + 0.00001)
    # 计算F1分数（精确率和召回率的调和平均）
    f1 = 2 * precision * recall / (precision + recall + 0.00001)

    # 计算ROC-AUC（衡量模型区分正常/异常的能力）
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0  # 处理只有一类标签时的异常情况

    return f1, precision, recall, TP, TN, FP, FN, roc_auc


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    调整异常预测结果，解决连续异常场景下的漏标问题
    （实际业务中异常通常连续出现，模型可能只标记部分点，需要修正）

    Args:
        score (np.ndarray): 模型输出的异常分数数组
        label (np.ndarray): 真实异常标签数组
        threshold (float): 异常判定阈值（分数超过此值视为异常）
        pred (np.ndarray or None): 已有的预测结果（若提供则忽略score和threshold）
        calc_latency (bool): 是否计算异常检测延迟（从异常开始到被检测到的步数）

    Returns:
        np.ndarray: 调整后的预测标签数组
        (可选) float: 平均检测延迟（仅当calc_latency=True时返回）
    """
    if len(score) != len(label):
        raise ValueError("异常分数与标签长度必须一致")

    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0  # 累计检测延迟
    anomaly_count = 0  # 真实异常段数量
    anomaly_state = False  # 标记是否处于异常段中

    # 生成初始预测结果（若未提供pred则用threshold判定）
    if pred is None:

        predict = score > threshold  # 分数超过阈值视为异常   predict:[False False False False False False··········]
    else:
        predict = pred  # 使用外部提供的预测结果
    # 将原始标签 label 转换为二值化的真实异常标记True,False ，
    # 原始label为[0.0.0.0.0.0.0.0.0.0.0···············]
    # 转化后actual为: [False False False False False False········· ]
    # 转化的原因：与predict匹配
    actual = label > 0.1  # 真实异常标签（处理可能的非0/1标签）

    # 遍历所有样本，修正连续异常段的预测结果
    for i in range(len(score)):
        # 情况1：当前是真实异常，且预测为异常，但尚未标记为异常段
        # if/elif 处理异常段状态的切换（进入 / 退出)
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True  # 进入异常段
            anomaly_count += 1  # 计数真实异常段
            # j的取值：从i开始，每次减1，直到j=1（因为range的结束值是开区间，0不包含在内）
            # 回溯修正：将当前异常段中前面漏标的点补标为异常
            for j in range(i, 0, -1):
                if not actual[j]:  # 遇到正常点则停止回溯
                    break
                else:
                    if not predict[j]:  # 补标漏检的异常点
                        predict[j] = True
                        latency += 1  # 累计延迟步数

        # 情况2：当前是正常样本，退出异常段
        elif not actual[i]:
            anomaly_state = False

        # 修正: 处于异常段中，无论predict[i]是否为异常，都标记为异常
        if anomaly_state:
            predict[i] = True

    # 计算平均检测延迟（总延迟/异常段数量），以异常段为单位统计延迟
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)  # 加微小值避免除零
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    计算序列数据的异常检测评估指标（处理连续异常场景）

    Args:
        score (np.ndarray): 异常分数数组
        label (np.ndarray): 真实标签数组
        threshold (float): 异常判定阈值
        calc_latency (bool): 是否计算检测延迟

    Returns:
        tuple: 包含评估指标的元组（含延迟，若calc_latency=True）
    """
    if calc_latency:
        # 调整预测结果并计算延迟
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        # 计算基础指标并附加延迟
        metrics = list(calc_point2point(predict, label))
        metrics.append(latency)
        return metrics
    else:
        # 仅调整预测结果并计算基础指标
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


# def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
#     """
#     网格搜索最优异常阈值（最大化F1分数）
#
#     Args:
#         score (np.ndarray): 异常分数数组
#         label (np.ndarray): 真实标签数组
#         start (float): 搜索起始阈值
#         end (float): 搜索结束阈值（若为None则只搜索start点）
#         step_num (int): 搜索步数（步长=（end-start)/step_num）
#         display_freq (int): 每隔多少步打印一次中间结果
#         verbose (bool): 是否打印搜索过程
#
#     Returns:
#         tuple: 最优指标（F1, 精确率, 召回率, ...）和对应的阈值
#     """
#     if step_num is None or end is None:
#         end = start
#         step_num = 1
#
#     search_step = step_num
#     search_range = end - start
#     search_lower_bound = start
#
#     if verbose:
#         print(f"搜索范围: [{search_lower_bound}, {search_lower_bound + search_range})")
#
#     threshold = search_lower_bound
#     best_metrics = (-1., -1., -1.)  # 初始化最优指标（F1, 精确率, 召回率）
#     best_threshold = 0.0  # 最优阈值
#
#     # 遍历所有候选阈值
#     for i in range(search_step):
#         threshold += search_range / float(search_step)  # 计算当前阈值
#         # 计算当前阈值下的指标（含延迟）
#         current_metrics = calc_seq(score, label, threshold, calc_latency=True)
#
#         # 更新最优结果（以F1为核心指标）
#         if current_metrics[0] > best_metrics[0]:
#             best_threshold = threshold
#             best_metrics = current_metrics
#
#         # 打印中间结果
#         if verbose and i % display_freq == 0:
#             print(
#                 f"当前阈值: {threshold:.6f}, 指标: {current_metrics}, 最优指标: {best_metrics}, 最优阈值: {best_threshold:.6f}")
#
#     print(f"最优结果: {best_metrics}, 最优阈值: {best_threshold}")
#     return best_metrics, best_threshold


def pot_eval(init_score, score, label, q=1e-5, level=0.02):
    """
    使用POT（Peak Over Threshold）算法计算异常阈值并评估
    POT算法原理：从正常样本中学习"正常分布"，将显著偏离该分布的样本判定为异常
    适用于无监督场景（仅需少量正常样本即可确定阈值）

    Args:
        init_score (np.ndarray): 训练集异常分数（用于学习正常分布）
        score (np.ndarray): 测试集异常分数（待评估的分数）
        label (np.ndarray): 测试集真实标签
        q (float): 风险水平（越小表示对异常的判定越严格）
        level (float): 初始阈值的概率水平

    Returns:
        dict: 包含各项评估指标的字典
        np.ndarray: 最终的异常预测标记数组
    """

    lms = lm[0]  # 从常量导入初始阈值参数（用于POT初始化）

    # 初始化POT模型（处理可能的初始化失败，逐步调整参数）
    #  result, pred = pot_eval(lt, l, ls)  # pred(28479,) pre
    while True:
        try:
            s = SPOT(q)  # 实例化POT模型（q为风险水平）
            s.fit(init_score, score)  # 预处理数据，转化为numpy.array,划分出小部分
            # 初始化阈值（level控制初始阈值的严格程度）
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except:
            # 若初始化失败，微调参数重试
            lms = lms * 0.999
        else:
            break  # 初始化成功，退出循环

    # 运行POT算法，得到异常阈值
    #dynamic=False ，检测过程中阈值保持不变,ret字典中存alarms:检测到的异常索引,threshoulds:阈值，总共时间步28479个阈值，仍以列表形式存在，目的是为每个数据点提供对应的阈值（dynamic=False,实际为同一固定值的重复），以适配逐点异常判断的逻辑，保持接口一致性。
    ret = s.run(dynamic=False)
    # 计算最终异常阈值（取POT输出阈值的均值并乘以调整系数）dynamic为False,阈值为同一个值，每个值是相同的，取均值实际就是每一个值
    pot_th = np.mean(ret['thresholds']) * lm[1]

    # 根据阈值将测试集的异常分数转换为异常标签，calc_latency=True为要计算异常延迟
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    # 计算基础评估指标
    p_t = calc_point2point(pred, label)

    # 返回评估结果字典和预测标记
    return {
        'f1': p_t[0],  # F1分数
        'precision': p_t[1],  # 精确率
        'recall': p_t[2],  # 召回率
        'TP': p_t[3],  # 真正例数量
        'TN': p_t[4],  # 真负例数量
        'FP': p_t[5],  # 假正例数量
        'FN': p_t[6],  # 假负例数量
        'ROC/AUC': p_t[7],  # ROC曲线下面积
        'threshold': pot_th,  # 最终使用的异常阈值
        'pot-latency': p_latency   # 检测延迟（可选）
    }, np.array(pred)