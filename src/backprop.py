import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC, abstractmethod


# -------------------------- 基类 --------------------------
class BackpropStrategy(ABC):
    """反向传播策略基类"""
    def __init__(self, model):
        self.model = model  # 模型实例
        self.feats = None  # 特征数（子类初始化）

    @abstractmethod
    def forward(self, epoch, data, dataO, optimizer, scheduler, training=True):
        """统一接口：训练或测试入口"""
        pass


# -------------------------- OmniAnomaly模型子类 --------------------------
class OmniAnomalyBackprop(BackpropStrategy):
    def forward(self, epoch, data, dataO, optimizer, scheduler, training=True):
        """
        反向传播函数：实现模型的训练或测试
        """
        l = nn.MSELoss(reduction='mean' if training else 'none')
        feats = dataO.shape[1]

        if training:
            mses, klds = [], []
            for i, d in enumerate(data):
                y_pred, mu, logvar, hidden = self.model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + self.model.beta * KLD
                mses.append(torch.mean(MSE).item())
                klds.append(self.model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            y_preds = []
            for i, d in enumerate(data):
                y_pred, _, _, hidden = self.model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().numpy(), y_pred.detach().numpy()


# -------------------------- TranAD模型子类 --------------------------
class TranADBackprop(BackpropStrategy):
    def forward(self, epoch, data, dataO, optimizer, scheduler, training=True):
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
        l = nn.MSELoss(reduction='none')
        feats = dataO.shape[1]  # 原始数据的特征数（用于结果对齐）

        # 转换数据为DoubleTensor并创建数据集，data为trainD,train0
        data_x = torch.DoubleTensor(data) # {28479,10,38}
        dataset = TensorDataset(data_x, data_x)
        # batch_size为128，每次加载128个窗口的数据(machine-1-1的数据被分为大约2000多个窗口)
        bs = self.model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        # w-size:10
        w_size = self.model.n_window
        l1s = []  # 损失记录
        if training:
            for d, _ in dataloader: # d{128,10,38}
                local_bs = d.shape[0]  # 本地批大小 local_bs:128
                # 调整数据形状：[批大小, 窗口大小, 特征数] -> [窗口大小, 批大小, 特征数]
                # [窗口大小, 批大小, 特征数] -> [10,128,38]
                window = d.permute(1, 0, 2) # window:{10,128,38}
                # 取本窗口最后一个时刻的数据作为预测目标
                elem = window[-1, :, :].view(1, local_bs, feats) # elem:{1,128,38}
                z = self.model(window, elem)  # TranAD模型输出预测结果 z{tuple:2} （x1=O₁, x2=Ô₂）
                # 损失计算：若输出为元组，用动态权重组合两个输出的损失
                # 进化损失计算：动态权重组合两阶段损失（文档中L₁=ϵ⁻ⁿ∥O₁-W∥₂+(1-ϵ⁻ⁿ)∥Ô₂-W∥₂，） z[0]-> O₁  z[1]-> Ô₂
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem) # l1:{1,128,38}
                if isinstance(z, tuple):
                    z = z[1]  # 取第二个输出作为主要预测结果 z:{1,128,38}
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            # 测试阶段：收集所有批次的预测结果和损失
            y_pred_list = []
            loss_list = []
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)  # 调整为[窗口大小, 批大小, 特征数]
                elem = window[-1, :, :].view(1, local_bs, feats)  # 目标值
                z = self.model(window, elem)  # 模型预测
                if isinstance(z, tuple):
                    z = z[1]  # 取第二个输出作为预测结果
                # 计算损失
                loss = l(z, elem)
                # 收集结果
                y_pred_list.append(z.detach())
                loss_list.append(loss.detach())
            # 拼接所有批次结果
            y_pred = torch.cat(y_pred_list, dim=1).squeeze(0)  # 移除多余维度
            loss = torch.cat(loss_list, dim=1).squeeze(0)
            return loss.numpy(), y_pred.numpy()

# -------------------------- DTAAD模型子类 --------------------------
class DTAADBackprop(BackpropStrategy):
    def forward(self, epoch, data, dataO, optimizer, scheduler, training=True):
        # 基础损失函数（保留原始逻辑）
        mse_loss = nn.MSELoss(reduction='mean' if training else 'none')
        feats = dataO.shape[1]  # 特征数
        _lambda = 0.8  # 双输出损失权重（与DTAAD的双分支对应）

        # 数据处理
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        batch_size = self.model.batch if training else len(data)  # 用model.batch而非batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=training)

        total_losses = []

        if training:
            self.model.train()
            for batch_data, _ in dataloader:
                # 1. 维度调整：[batch, window, feats] -> [batch, feats, window]（适配DTAAD的TCN输入）
                window = batch_data.permute(0, 2, 1).double()  # 关键：修正维度顺序

                # 2. 提取目标值（窗口最后一个时刻的特征）
                target = batch_data[:, -1, :].double()  # 形状：[batch, feats]

                # 3. 模型调用：只传入window（DTAAD的forward只接受1个参数）
                # 注意：DTAAD返回双输出(x1, x2)，形状为(Batch, 1, output_channel)
                x1, x2 = self.model(window)  # 正确调用：仅传入window

                # 4. 调整输出形状以匹配目标
                x1 = x1.squeeze(1)  # [batch, feats]
                x2 = x2.squeeze(1)  # [batch, feats]

                # 5. 计算双分支加权损失（与原始逻辑一致）
                loss = _lambda * mse_loss(x1, target) + (1 - _lambda) * mse_loss(x2, target)

                # 6. 反向传播
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_losses.append(loss.item())

            scheduler.step()
            avg_loss = np.mean(total_losses)
            tqdm.write(f'Epoch {epoch},\tTotal Loss = {avg_loss:.6f}')
            return avg_loss, optimizer.param_groups[0]['lr']

        else:
            self.model.eval()
            all_preds = []
            all_losses = []
            with torch.no_grad():
                for batch_data, _ in dataloader:
                    window = batch_data.permute(0, 2, 1).double()
                    target = batch_data[:, -1, :].double()

                    # 测试阶段模型调用：同样只传入window
                    x1, x2 = self.model(window)
                    x2 = x2.squeeze(1)  # 取第二个输出作为预测结果

                    # 计算损失
                    loss = mse_loss(x2, target)
                    all_preds.append(x2.cpu().numpy())
                    all_losses.append(loss.cpu().numpy())

            # 拼接所有批次结果
            y_pred = np.concatenate(all_preds, axis=0)
            loss = np.concatenate(all_losses, axis=0)
            return loss, y_pred

# -------------------------- 策略工厂类（用于匹配模型和策略） --------------------------
class BackpropFactory:
    @staticmethod
    def get_strategy(model):
        model_name = model.name
        if "OmniAnomaly" in model_name:
            return OmniAnomalyBackprop(model)
        elif "TranAD" in model_name:
            return TranADBackprop(model)
        elif "DTAAD" in model_name:
            return DTAADBackprop(model)
        else:
            raise ValueError(f"未找到模型 {model_name} 对应的反向传播策略")