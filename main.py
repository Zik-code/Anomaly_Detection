import pickle
import os
import pandas as pd
from tqdm import tqdm  # 用于显示进度条
from src.models import *  # 导入自定义模型（如DAGMM、TranAD等异常检测模型）
from src.constants import *  # 导入常量（如颜色配置、路径等）
from src.pot import *  # 导入POT算法相关函数（用于异常阈值计算）
from src.utils import *  # 导入工具函数（如绘图、评估指标计算等）
from src.diagnosis import *  # 导入诊断相关函数（可能用于结果分析）
import torch.nn as nn  # PyTorch神经网络模块
from time import time  # 用于计算时间
from pprint import pprint  # 用于格式化输出
from src.plotting import *
from src.load_data import *


def convert_to_windows(data, model):
    """
    将时序数据转换为滑动窗口格式（适配时序异常检测模型的输入要求）

    参数:
        data: 原始时序数据，形状为[时间步, 特征数]
        model: 模型实例，用于获取窗口大小（model.n_window）

    返回:
        windows: 转换后的窗口数据，形状为[窗口数, 窗口大小, 特征数]
    """
    windows = []
    w_size = model.n_window  # 窗口大小由模型定义
    for i, g in enumerate(data):
        # 若当前索引小于窗口大小，用首元素填充前面的缺失部分（避免窗口大小不足）
        if i >= w_size:
            w = data[i - w_size:i]  # 取[i-w_size, i)区间的数据作为一个窗口
        else:
            # 前w_size个窗口：用首元素重复(w_size-i)次，再拼接前i个元素
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        # 对TranAD或Attention模型，窗口保持原形状；其他模型展平窗口为一维特征
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    return torch.stack(windows)  # 堆叠所有窗口为张量




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


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    """
    反向传播函数：实现模型的训练或测试（根据training参数切换模式）

    参数:
        epoch: 当前轮次
        model: 模型实例
        data: 输入数据（可能是窗口化后的数据）
        dataO: 原始输入数据（用于计算损失时的特征对齐）
        optimizer: 优化器
        scheduler: 学习率调度器
        training: 是否为训练模式（True为训练，False为测试）

    返回:
        训练模式：返回总损失和当前学习率
        测试模式：返回损失数组和预测结果
    """
    # 基础损失函数：MSE（均方误差）
    # 训练时用mean reduction（平均损失），测试时用none（保留每个样本的损失）
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]  # 原始数据的特征数（用于结果对齐）

    if 'Attention' in model.name:
        l = nn.MSELoss(reduction='none')
        n = epoch + 1;
        w_size = model.n_window
        l1s = [];
        res = []
        if training:
            for d in data:
                ae, ats = model(d)
                # res.append(torch.mean(ats, axis=0).view(-1))
                l1 = l(ae, d)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            ae1s, y_pred = [], []
            for d in data:
                ae1 = model(d)
                y_pred.append(ae1[-1])
                ae1s.append(ae1)
            ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
            loss = torch.mean(l(ae1s, data), axis=1)
            return loss.detach().numpy(), y_pred.detach().numpy()

    # -------------------------- OmniAnomaly模型--------------------------
    elif 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []  # 记录MSE损失和KL散度
            for i, d in enumerate(data):
                # OmniAnomaly输出：预测结果、均值、对数方差、隐藏状态
                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)  # 重构损失
                # KL散度（变分自编码器的正则化项）
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD  # 总损失（MSE + beta*KLD）
                mses.append(torch.mean(MSE).item())
                klds.append(model.beta * torch.mean(KLD).item())
                # 反向传播
                optimizer.zero_grad()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']

        else:
            y_preds = []
            for i, d in enumerate(data):
                # 测试时保持隐藏状态连续（时序依赖） y_pred 单台机器一次预测的结果，即预测一行的结果。
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred) #  y_preds最后和测试集保持相同的维度
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)  # 计算所有样本的MSE
            return MSE.detach().numpy(), y_pred.detach().numpy()

    # -------------------------- USAD模型特殊处理 --------------------------
    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction='none')
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []  # 两个自编码器的损失记录

        if training:
            for d in data:
                # USAD输出：两个自编码器的重构结果、第二个自编码器对第一个的重构结果
                ae1s, ae2s, ae2ae1s = model(d)
                # 自编码器1的损失（带动态权重）
                l1 = (1 / n) * l(ae1s, d) + (1 - 1 / n) * l(ae2ae1s, d)
                # 自编码器2的损失（带动态权重）
                l2 = (1 / n) * l(ae2s, d) - (1 - 1 / n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)  # 总损失
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']

        else:
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data:
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1)
                ae2s.append(ae2)
                ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            # 提取与原始特征对应的预测结果
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            # 组合损失（0.1*ae1损失 + 0.9*ae2ae1损失）
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()

    # -------------------------- GAN类模型处理 --------------------------
    elif 'GAN' in model.name:
        l = nn.MSELoss(reduction='none')
        bcel = nn.BCELoss(reduction='mean')  # 二分类交叉熵损失（判别器用）
        msel = nn.MSELoss(reduction='mean')  # MSE损失（生成器用）
        # 标签平滑（真实样本标签0.9，伪造样本标签0.1，提高泛化性）
        real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])
        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        n = epoch + 1
        w_size = model.n_window
        mses, gls, dls = [], [], []  # 记录MSE、生成器损失、判别器损失

        if training:
            for d in data:
                # 训练判别器
                model.discriminator.zero_grad()  # 判别器梯度清零
                _, real, fake = model(d)  # 真实样本得分和伪造样本得分
                # 判别器损失：真实样本尽可能接近1，伪造样本尽可能接近0
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()  # 判别器反向传播
                model.generator.zero_grad()  # 生成器梯度清零
                optimizer.step()  # 更新判别器参数

                # 训练生成器
                z, _, fake = model(d)  # 生成器输出重构结果和伪造样本得分
                mse = msel(z, d)  # 生成器重构损失
                # 生成器损失：伪造样本尽可能接近1（欺骗判别器）
                gl = bcel(fake, real_label)
                tl = gl + mse  # 总损失（生成器）
                tl.backward()  # 生成器反向传播
                model.discriminator.zero_grad()  # 判别器梯度清零
                optimizer.step()  # 更新生成器参数

                # 记录损失
                mses.append(mse.item())
                gls.append(gl.item())
                dls.append(dl.item())
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls) + np.mean(dls), optimizer.param_groups[0]['lr']

        else:
            outputs = []
            for d in data:
                z, _, _ = model(d)  # 生成器输出重构结果
                outputs.append(z)
            outputs = torch.stack(outputs)
            # 提取与原始特征对应的预测结果
            y_pred = outputs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()

    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data);
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1;
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]

    # -------------------------- TranAD模型特殊处理 --------------------------
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        # 转换数据为DoubleTensor并创建数据集
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        # 训练时用模型定义的批大小，测试时用全量数据
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s = []  # 损失记录

        if training:
            for d, _ in dataloader:
                local_bs = d.shape[0]  # 本地批大小
                # 调整数据形状：[批大小, 窗口大小, 特征数] -> [窗口大小, 批大小, 特征数]
                window = d.permute(1, 0, 2)
                # 取窗口最后一个时刻的元素作为预测目标
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)  # TranAD模型输出预测结果
                # 损失计算：若输出为元组，用动态权重组合两个输出的损失
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple):
                    z = z[1]  # 取第二个输出作为主要预测结果
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)  # 总损失
                # 反向传播
                optimizer.zero_grad()
                loss.backward(retain_graph=True)  # 保留计算图（TranAD需要）
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']

        else:
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple):
                    z = z[1]  # 取第二个输出
            # 计算损失（仅窗口最后一个时刻）
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]

    # -------------------------- 通用模型处理（其他模型） --------------------------
    else:
        y_pred = model(data)  # 模型预测
        loss = l(y_pred, data)  # 计算MSE损失

        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']

        else:
            return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == '__main__':
    # 加载数据集（训练集、测试集、异常标签）
    train_loader, test_loader, labels = load_dataset(args.dataset)


    # 加载模型（根据模型名和特征维度），labels.shape[1]为特征数，epoch, accuracy_list为预训练存储的数据和参数
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    # 数据预处理：将原始数据转换为模型需要的窗口格式
    # 读取完整的训练和测试数据（因为DataLoader的批大小是全量）
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD  # 保存原始数据（用于结果对齐）

    # 对需要窗口输入的模型，转换训练和测试数据为窗口格式，
    # 'TranAD'有多个模型，单独放置更灵活地匹配所有包含 TranAD 的模型名称，而无需枚举所有变体。
    if model.name in ['GDN', 'MTAD_GAT','MAD_GAN'] or 'TranAD'in model.name:
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

    # -------------------------- 训练阶段 --------------------------
    if not args.test:  # 训练
        print(f'{color.HEADER}在{args.dataset}数据集上训练{args.model}模型{color.ENDC}')
        num_epochs = 10  # 训练轮次
        e = epoch + 1  # 起始轮次
        start = time()  # 记录开始时间
        # 迭代训练轮次（用tqdm显示进度条）
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            # 调用backprop进行训练，返回损失和学习率
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))  # 记录训练过程
        # 打印训练时间
        print(color.BOLD + '训练时间: ' + "{:10.4f}".format(time() - start) + ' 秒' + color.ENDC)
        # 保存模型
        save_model(model, optimizer, scheduler, e, accuracy_list)
        # 绘制训练损失曲线
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    # -------------------------- 测试阶段 --------------------------
    torch.zero_grad = True  # 禁用梯度计算（节省内存）
    model.eval()
    print(f'{color.HEADER}在{args.dataset}数据集上测试{args.model}模型{color.ENDC}')
    # 调用backprop进行测试，返回损失和预测结果
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    if not args.test:
        if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0)
        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

    # 评估指标计算
    df = pd.DataFrame()  # 存储每个特征的评估结果
    # 计算训练集损失（用于确定异常阈值）
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)

    # 对每个特征单独评估
    for i in range(loss.shape[1]):
        lt = lossT[:, i]  # 训练集第i个特征的损失
        l = loss[:, i]  # 测试集第i个特征的损失
        ls = labels[:, i]  # 第i个特征的异常标签
        # 用POT算法计算异常阈值并评估（返回精确率、召回率等指标）
        result, pred = pot_eval(lt, l, ls)
        df = df._append(result, ignore_index=True)  # 记录每个特征的评估结果

    # 综合所有特征的损失进行评估
    # 训练集总损失（平均所有特征）
    lossTfinal = np.mean(lossT, axis=1)
    # 测试集总损失（平均所有特征）
    lossFinal = np.mean(loss, axis=1)
    # 综合异常标签（只要有一个特征异常则标记为异常）
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    # 综合评估
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    # 添加命中分数和NDCG等指标
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))

    # 打印评估结果
    print(df)  # 每个特征的评估结果
    pprint(result)  # 综合评估结果
