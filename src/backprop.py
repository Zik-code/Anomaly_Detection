from tqdm import tqdm  # 用于显示进度条
from src.load_data import *


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

    # -------------------------- TranAD模型特殊处理 --------------------------
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        # 转换数据为DoubleTensor并创建数据集，data为trainD,train0
        data_x = torch.DoubleTensor(data) # {28479,10,38}
        dataset = TensorDataset(data_x, data_x)
        # batch_size为128，每次加载128个窗口的数据(machine-1-1的数据被分为大约2000多个窗口)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        # w-size:10
        w_size = model.n_window
        l1s = []  # 损失记录
        if training:
            for d, _ in dataloader: # d{128,10,38}
                local_bs = d.shape[0]  # 本地批大小 local_bs:128
                # 调整数据形状：[批大小, 窗口大小, 特征数] -> [窗口大小, 批大小, 特征数]
                # [窗口大小, 批大小, 特征数] -> [10,128,38]
                window = d.permute(1, 0, 2) # window:{10,128,38}
                # 取本窗口最后一个时刻的数据作为预测目标
                elem = window[-1, :, :].view(1, local_bs, feats) # elem:{1,128,38}
                z = model(window, elem)  # TranAD模型输出预测结果 z{tuple:2} x1,x2
                # 损失计算：若输出为元组，用动态权重组合两个输出的损失
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem) # l1:{1,128,38}
                if isinstance(z, tuple):
                    z = z[1]  # 取第二个输出作为主要预测结果 z:{1,128,38}
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
            for d, _ in dataloader: # d{28479,10,38}
                window = d.permute(1, 0, 2) # window(10,28479,38)
                elem = window[-1, :, :].view(1, bs, feats)  # elem(1,28479,38)
                z = model(window, elem)
                if isinstance(z, tuple):
                    z = z[1]  # 取第二个输出  z(1,28479,38)
            # 计算损失（仅窗口最后一个时刻）
            loss = l(z, elem)[0] #loss(1,28479,38)
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
