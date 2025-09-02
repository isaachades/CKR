import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device,temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.contrast_mode = "all"
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.contrast_mode = contrast_mode
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        """
        生成一个掩码矩阵，用于屏蔽相关样本。

        参数:
        - N: int类型，矩阵的大小，假设为NxN矩阵。

        返回值:
        - mask: torch.Tensor类型，布尔矩阵，大小为NxN，其中对角线和对角线外的某些元素被设置为False，其余为True。
        """
        # 初始化一个全1的 NxN 矩阵
        mask = torch.ones((N, N))
        # 将对角线元素设置为0，即屏蔽对角线上的样本
        mask = mask.fill_diagonal_(0)
        # 在对角线两侧交替设置0，即屏蔽掉对角线外成对的样本
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        # 将掩码矩阵转换为布尔类型，方便后续使用
        mask = mask.bool()
        return mask

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        """
        前向传播函数用于计算对比学习损失。

        参数:
        - features: 输入特征，要求至少有3个维度，格式为[批量大小, 视图数量, ...]。
        - labels: 标签数组，可选，用于定义样本之间的关系。
        - mask: 自定义的样本关系掩码，可选。
        - target_labels: 目标标签列表，必须提供，用于指定要计算损失的类别。
        - reduction: 损失的减少方式，默认为'mean'，可选值包括'mean'和'none'。

        返回:
        - 计算得到的损失值，根据reduction参数进行汇总。
        """

        # 确保目标标签被正确提供
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"
        # features = F.normalize(features,dim = 1)
        # 根据输入特征决定运行设备
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # 处理输入特征维度不正确的情况
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # 处理labels和mask的冲突定义
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # 根据对比模式选择锚点特征
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 计算logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # 为了数值稳定性，减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 复制mask并处理自对比情况
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+ 1e-6 )

        # 计算正样本log-likelihood的均值
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+ 1e-6)

        # 计算损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # 根据目标标签调整损失
        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        # 根据指定的减少方式计算最终损失
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss





    def forward_feature(self, h_i, h_j):
        """
        前向传播特征计算函数。

        参数:
        - h_i: 输入的特征向量i
        - h_j: 输入的特征向量j

        返回值:
        - 计算得到的损失值
        """
        N = 2 * self.batch_size  # 计算总样本数
        h = torch.cat((h_i, h_j), dim=0)  # 将特征向量i和j连接起来

        # 计算相似度矩阵并进行归一化
        sim = torch.matmul(h, h.T) / self.temperature_f
        # 提取i到j和j到i的相似度作为正样本
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # 将正样本整理为N*1的矩阵
        # 通过mask筛选出负样本
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)  # 将负样本整理为N*m的矩阵，其中m为负样本数量

        labels = torch.zeros(N).to(positive_samples.device).long()  # 生成N个标签，全部标记为0
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # 将正负样本连接起来，形成logits
        loss = self.criterion(logits, labels)  # 计算损失
        loss /= N  # 对损失进行平均
        return loss

    def forward_label(self, q_i, q_j):
        """
        计算并返回前向标签损失。

        该函数主要用于通过计算两个分布（q_i 和 q_j）的交叉熵损失以及它们之间的相似度来生成一个损失值。
        其中，q_i 和 q_j 分别表示两种不同情况下的分布。

        参数:
        - q_i: Tensor, 表示第一种分布的查询矩阵。
        - q_j: Tensor, 表示第二种分布的查询矩阵。

        返回值:
        - loss: Tensor, 计算得到的损失值，综合考虑了分布的熵和分布间相似度的差异。
        """
        # 计算并归一化 q_i，然后计算其熵
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()

        # 计算并归一化 q_j，然后计算其熵
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()

        # 计算总熵
        entropy = ne_i + ne_j

        # 转置 q_i 和 q_j，准备计算相似度
        q_i = q_i.t()
        q_j = q_j.t()

        # 设置总类别数，并将 q_i 和 q_j 合并为一个大矩阵 q
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        # 计算 q 矩阵的相似度
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        # 提取 i 到 j 和 j 到 i 的相似度对
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        # 合并正向和反向相似度对，形成正样本集群
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # 生成负样本集群，通过屏蔽相关样本实现
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        # 准备标签和logits，用于计算损失
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        # 计算二元分类损失
        loss = self.criterion(logits, labels)
        # 平均损失，返回最终的损失值
        loss /= N
        return loss + entropy

    def forward_feature_InfoNCE(self, h_i, h_j, batch_size=128):
        self.batch_size = batch_size

        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_feature_PSCL(self, z1, z2, r=3.0):  #  r=3.0
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1 - mask1) * F.normalize(z1, dim=1) * np.sqrt(r)
        z2 = mask2 * z2 + (1 - mask2) * F.normalize(z2, dim=1) * np.sqrt(r)
        loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
        square_term = torch.matmul(z1, z2.T) ** 2
        loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                     z1.shape[0] / (z1.shape[0] - 1)

        return loss_part1 + loss_part2

    def forward_feature_RINCE(self, out_1, out_2, lam=0.001, q=0.5, temperature=0.5, batch_size=128):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
            lam, q, temperature
        """
        # # gather representations in case of distributed training
        # # out_1_dist: [batch_size * world_size, dim]
        # # out_2_dist: [batch_size * world_size, dim]
        # if torch.distributed.is_available() and torch.distributed.is_initialized():
        #     out_1_dist = SyncFunction.apply(out_1)
        #     out_2_dist = SyncFunction.apply(out_2)
        # else:
        self.batch_size = batch_size

        out_1_dist = out_1
        out_2_dist = out_2


        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        similarity = torch.exp(torch.mm(out, out_dist.t()) / temperature)
        # neg_mask = self.compute_neg_mask()
        N = 2 * self.batch_size
        neg_mask = self.mask_correlated_samples(N)
        neg = torch.sum(similarity * neg_mask.to(self.device), 1)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # InfoNCE loss
        # loss = -(torch.mean(torch.log(pos / (pos + neg))))

        # RINCE loss
        neg = ((lam*(pos + neg))**q) / q
        pos = -(pos**q) / q
        loss = pos.mean() + neg.mean()

        return loss

