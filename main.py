import pickle
import os
import pandas as pd
from tqdm import tqdm  # 用于显示进度条
from src.models import *  # 导入自定义模型（如DAGMM、TranAD等异常检测模型）
from src.constants import *  # 导入常量（如颜色配置、路径等）
from src.pot import *  # 导入POT算法相关函数（用于异常阈值计算）
from src.utils import *  # 导入工具函数（如绘图、评估指标计算等）
#from src.diagnosis import *  #
import torch.nn as nn  # PyTorch神经网络模块
from time import time  # 用于计算时间
from pprint import pprint  # 用于格式化输出
from src.plotting import *
from src.load_data import *
from src.backprop import *


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    """
    保存模型参数、优化器状态、学习率调度器状态及训练记录
    参数:
        model: 训练好的模型
        optimizer: 优化器（如AdamW）
        scheduler: 学习率调度器
        epoch: 当前训练轮次
        accuracy_list: 训练过程中的损失和学习率记录
    """
    # 模型保存路径：checkpoints/模型名_数据集名/
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    # 保存字典：包含模型状态、优化器状态等关键信息
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  # 模型参数
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器参数
        'scheduler_state_dict': scheduler.state_dict(),  # 调度器参数
        'accuracy_list': accuracy_list  # 训练损失和学习率记录
    }, file_path)


def load_model(modelname, dims):
    """
    加载模型（若存在预训练模型则加载，否则初始化新模型）

    参数:
        modelname: 模型类名
        dims: 输入特征维度

    返回:
        model: 模型实例
        optimizer: 优化器实例
        scheduler: 学习率调度器实例
        epoch: 上次训练的轮次（-1表示新模型）
        accuracy_list: 训练记录（损失和学习率）
    """
    import src.models  # 局部导入，src.models模块较多，避免程序启动就加载模块，提高大型项目初始化速度
    # 动态获取模型类（根据模型名从src.models中获取）
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()  # 初始化模型，设置为double精度

    # 初始化优化器（AdamW，带权重衰减）
    optimizer = torch.optim.AdamW(
        model.parameters(), #需要优化的模型参数集合
        lr=model.lr,  # 学习率从模型属性获取
        weight_decay=1e-5  # 权重衰减，防止过拟合
    )
    # 学习率调度器,学习率每5轮衰减为原来的0.9倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

    # 检查预训练模型是否存在
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}加载预训练模型: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)  # 加载 checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 加载调度器参数
        epoch = checkpoint['epoch']  # 上次训练的轮次
        accuracy_list = checkpoint['accuracy_list']  # 训练记录
    else:
        print(f"{color.GREEN}创建新模型: {model.name}{color.ENDC}")
        epoch = -1  # 新模型，从轮次-1开始
        accuracy_list = []  # 空训练记录
    return model, optimizer, scheduler, epoch, accuracy_list




if __name__ == '__main__':
    # 加载数据集（训练集、测试集、异常标签）
    train_loader, test_loader, labels = load_dataset(args.dataset)

    # 加载模型（根据模型名和特征维度），labels.shape[1]为特征数，epoch, accuracy_list为预训练存储的数据和参数
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    # 读取完整的训练和测试数据（因为DataLoader的批大小是全量）
    trainD, testD = next(iter(train_loader)), next(iter(test_loader)) # 取出一个bacth_size数据
    trainO, testO = trainD, testD  # 保存原始数据（用于结果对齐）

    # 对需要窗口输入的模型，转换训练和测试数据为窗口格式，
    # 'TranAD'有多个模型，单独放置更灵活地匹配所有包含 TranAD 的模型名称，而无需枚举所有变体。
    if model.name in 'TranAD' or 'DTAAD'in model.name:
        trainD, testD = convert_to_windows(trainD, model,args), convert_to_windows(testD, model,args) # trainD:{28479,10,38}

        # **** 修改后的获取对应的反向传播策略
    backprop_strategy = BackpropFactory.get_strategy(model)
    # -------------------------- 训练阶段 --------------------------
    if not args.test:  # 训练
        print(f'{color.HEADER}在{args.dataset}数据集上训练{args.model}模型{color.ENDC}')
        num_epochs = 10  # 训练轮次
        e = epoch + 1  # 起始轮次
        start = time()  # 记录开始时间
        # 迭代训练轮次（用tqdm显示进度条）
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            # 调用backprop进行训练，返回损失和学习率
            # 新代码：通过策略实例调用forward方法
            lossT, lr = backprop_strategy.forward(
                epoch=e,
                data=trainD,
                dataO=trainO,
                optimizer=optimizer,
                scheduler=scheduler,
                training=True  # 训练模式
            )
            accuracy_list.append((lossT, lr))  # 记录训练过程
        # 打印训练时间

        print(color.BOLD + '训练时间: ' + "{:10.4f}".format(time() - start) + ' 秒' + color.ENDC)
        # 保存模型
        save_model(model, optimizer, scheduler, e, accuracy_list)
        # 绘制训练损失曲线
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    # -------------------------- 测试阶段 -------------------------
    torch.zero_grad = True  # 禁用梯度计算（节省内存）
    model.eval()
    print(f'{color.HEADER}在{args.dataset}数据集上测试{args.model}模型{color.ENDC}')
    # 调用backprop进行测试，返回损失和预测结果
    #loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False) # loss(28479,38) y_pred(28479,38)
    # 使用策略模式调用测试逻辑
    # loss:(28479,38)  y_pred(28479,38)
    loss, y_pred = backprop_strategy.forward(
        epoch=0,
        data=testD,
        dataO=testO,
        optimizer=optimizer,
        scheduler=scheduler,
        training=False
    )
    if not args.test:
        if 'TranAD' or 'DTAAD' in model.name: testO = torch.roll(testO, 1, 0)
        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

        ### Plot attention
    if not args.test:
        if 'DTAAD' in model.name:
            plot_attention(model, 1, f'{args.model}_{args.dataset}')

    # 评估指标计算
    df = pd.DataFrame()  # 存储每个特征的评估结果
    # 计算训练集损失（用于确定异常阈值）
    lossT, _ = backprop_strategy.forward(
        epoch=0,
        data=trainD,
        dataO=trainO,
        optimizer=optimizer,
        scheduler=scheduler,
        training=False
    )

    # 对每个特征单独评估
    for i in range(loss.shape[1]): # i 0~37
        lt = lossT[:, i]  # 训练集第i个特征的损失 lt(28479,)
        l = loss[:, i]  # 测试集第i个特征的损失 l(28479,)
        ls = labels[:, i]  # 第i个特征的异常标签 ls(28479,)
        # 用POT算法计算异常阈值并评估（返回精确率、召回率等指标）
        result, pred = pot_eval(lt, l, ls)  # pred(28479,) pred:[False,False,False·······]
        df = df._append(result, ignore_index=True)  # 记录每个特征的评估结果

    # 综合所有特征的损失进行评估
    # 训练集总损失（平均所有特征）
    lossTfinal = np.mean(lossT, axis=1)
    # 测试集总损失（平均所有特征）
    lossFinal = np.mean(loss, axis=1)
    # （只要有一个特征异常则标记为异常）
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    # 综合评估
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    # 添加命中分数和NDCG等指标
    # result.update(hit_att(loss, labels))
    # result.update(ndcg(loss, labels))

    # 打印评估结果
    print("各特征评估结果：")
    print(df)
    print("\n综合评估结果：")
    pprint(result)