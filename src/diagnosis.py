import numpy as np
from sklearn.metrics import ndcg_score
from src.constants import lm  # 从自定义模块导入常量，具体用途需结合项目上下文


def hit_att(ascore, labels, ps=[100, 150]):
    """
    计算Hit@P%指标，评估模型在推荐或排序任务中的表现

    参数:
        ascore: 模型预测的分数数组，形状为[样本数, 项目数]
                每个元素表示对该项目的预测分数
        labels: 标签数组，形状与ascore相同
                元素为1表示该项目是相关的/正样本，0表示不相关
        ps: 百分比列表，用于计算不同比例下的Hit指标，默认[100, 150]

    返回:
        res: 字典，键为"Hit@P%"格式，值为对应比例下的平均Hit分数
    """
    res = {}  # 存储计算结果的字典

    # 遍历每个百分比参数
    for p in ps:
        hit_score = []  # 存储每个样本的Hit分数

        # 遍历每个样本
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]  # 获取当前样本的预测分数和标签

            # 对预测分数进行降序排序，取索引（得到项目的排序结果）
            # argsort默认升序，[::-1]反转为降序
            sorted_indices = np.argsort(a).tolist()[::-1]

            # 找出当前样本中所有正样本的索引（标签为1的位置）
            positive_indices = set(np.where(l == 1)[0])

            # 只有当存在正样本时才计算Hit分数
            if positive_indices:
                # 计算需要考虑的项目数量：正样本总数的p%（四舍五入）
                # 例如p=100表示考虑与正样本数量相同的项目
                # p=150表示考虑正样本数量1.5倍的项目
                size = round(p * len(positive_indices) / 100)

                # 取排序结果中前size个项目的索引
                top_indices = set(sorted_indices[:size])

                # 计算命中数量：排序结果前size个中包含的正样本数量
                intersect = top_indices.intersection(positive_indices)

                # 当前样本的Hit分数 = 命中数量 / 正样本总数
                hit = len(intersect) / len(positive_indices)
                hit_score.append(hit)

        # 计算所有样本的平均Hit分数，存入结果字典
        res[f'Hit@{p}%'] = np.mean(hit_score)

    return res


def ndcg(ascore, labels, ps=[100, 150]):
    """
    计算NDCG@P%指标，评估排序质量（考虑相关性的等级和排序位置）

    参数:
        ascore: 模型预测的分数数组，形状为[样本数, 项目数]
        labels: 标签数组，形状与ascore相同
                元素为1表示相关，0表示不相关
        ps: 百分比列表，用于计算不同比例下的NDCG指标，默认[100, 150]

    返回:
        res: 字典，键为"NDCG@P%"格式，值为对应比例下的平均NDCG分数
    """
    res = {}  # 存储计算结果的字典

    # 遍历每个百分比参数
    for p in ps:
        ndcg_scores = []  # 存储每个样本的NDCG分数

        # 遍历每个样本
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]  # 获取当前样本的预测分数和标签

            # 找出当前样本中所有正样本的索引
            positive_indices = list(np.where(l == 1)[0])

            # 只有当存在正样本时才计算NDCG分数
            if positive_indices:
                # 计算需要考虑的项目数量：正样本总数的p%（四舍五入）
                k_p = round(p * len(positive_indices) / 100)

                try:
                    # 计算NDCG分数
                    # ndcg_score要求输入形状为[1, 项目数]（批次维度）
                    score = ndcg_score(
                        l.reshape(1, -1),  # 真实标签
                        a.reshape(1, -1),  # 预测分数
                        k=k_p  # 只考虑前k_p个项目
                    )
                    ndcg_scores.append(score)
                except Exception as e:
                    # 发生异常时返回空字典（可根据实际需求修改异常处理方式）
                    return {}

        # 计算所有样本的平均NDCG分数，存入结果字典
        res[f'NDCG@{p}%'] = np.mean(ndcg_scores)

    return res