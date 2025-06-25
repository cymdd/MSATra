from utils import *
from basemodel import *
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import torch
from scipy.stats import wasserstein_distance

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class GaussianNLLLoss(nn.Module):
    """https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    """
    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = 0.5*(torch.log(scale**2) + torch.abs(target - loc)**2 / scale**2)
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class DynamicWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var_pred = nn.Parameter(torch.zeros(1))  # 预测损失的不确定性
        self.log_var_align = nn.Parameter(torch.zeros(1)) # 对齐损失的不确定性
        self.log_var_div = nn.Parameter(torch.zeros(1))   # 多样性损失的不确定性

    def forward(self, pred_loss, align_loss, div_loss):

        # 计算各损失项的权重 (1 / (2 * sigma^2))
        exp_pred = torch.exp(-self.log_var_pred)
        exp_align = torch.exp(-self.log_var_align)
        exp_div = torch.exp(-self.log_var_div)
        weight_pred = 0.5 * exp_pred
        weight_align = 0.5 * exp_align
        weight_div = 0.5 * exp_div

        # 总损失公式
        total_loss = (
            weight_pred * pred_loss +
            weight_align * align_loss +
            weight_div * div_loss +
            math.log(exp_pred * exp_align * exp_div)  # 正则项
        )
        return total_loss

class MSATra(nn.Module):
    def __init__(self, args):
        super(MSATra, self).__init__()
        self.args = args
        self.scale_array = torch.tensor([1, 2, 4])
        self.batch_norm_gt = [{},{}]
        self.pre_obs = [{},{}]
        self.Scale_one_Encoder = Scale_Encoder(self.args,'s')
        self.Scale_Two_Encoder = Scale_Encoder(self.args, 'm')
        self.Scale_Four_Encoder = Scale_Encoder(self.args, 'm')
        self.Scale_Whole_Encoder = Scale_Encoder(self.args, 'l')

        self.Laplacian_Decoder = Laplacian_Decoder(self.args)
        self.AttentionModule = AttentionModule(self.args.hidden_size)
        self.Attention_TwoValue = Attention_TwoValue()
        self.Attention_AgScale = Attention_AgScale(self.args.hidden_size)
        self.Attention_AgScale2 = Attention_AgScale(self.args.hidden_size * 2)
        message_passing = []
        for i in range(self.args.message_layers):
            message_passing.append(Global_interaction(args))
        self.Global_interaction = nn.ModuleList(message_passing)

        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        self.dyw_loss = DynamicWeightedLoss()

    def forward(self, inputs_0, inputs_1, outbatch, iftest, ifvisualize=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if iftest:
            train_x_0, train_y_0, nei_list_batch_0, batch_split_0, batch_abs_gt_0, nei_num_batch_0 = self.getTrainX(
                inputs_0, device, 0)
            sour_Seq = self.get_ScaleSeq(train_x_0)

            sour_embedded = []
            for index in self.scale_array:
                index = index.item()
                if index == 1:
                    sour_embedded.append(self.Scale_one_Encoder(sour_Seq[index], 's'))
                elif index == 2:
                    sour_embedded.append(self.Scale_Two_Encoder(sour_Seq[index], 'm'))
                else:
                    sour_embedded.append(self.Scale_Four_Encoder(sour_Seq[index], 'm'))

            x_encoded_dense_0, hidden_state_unsplited_0, cn_0 =  self.Scale_Whole_Encoder.forward(train_x_0, 'l')

            hidden_state_global_0  = torch.zeros_like(hidden_state_unsplited_0, device=device)
            cn_global_0 = torch.zeros_like(cn_0, device=device)

            for index, batch in enumerate(batch_split_0):
                left_0, right_0 = batch[0], batch[1]
                element_states_0 = hidden_state_unsplited_0[left_0: right_0]  # [N, D]
                cn_state_0 = cn_0[left_0: right_0]  # [N, D]
                if element_states_0.shape[0] != 1:
                    hidden_state_global_0[left_0: right_0], cn_global_0[left_0: right_0] = self.Graph_conv_interaction(batch_abs_gt_0, left_0, right_0, element_states_0,
                                                                                          cn_state_0, nei_list_batch_0, index, device)
                else:
                    hidden_state_global_0[left_0: right_0] = element_states_0
                    cn_global_0[left_0: right_0] = cn_state_0

            sour_aggrate = self.scale_fusion(sour_embedded[:-1], hidden_state_global_0)
            sour_hidden = self.multiScale_sampling(sour_aggrate)

            mdn_out = self.Laplacian_Decoder.forward(x_encoded_dense_0, sour_hidden, cn_global_0)
            predict_loss, variety_loss, full_pre_tra = self.mdn_loss(train_y_0.permute(2, 0, 1), mdn_out, ifvisualize, batch_abs_gt_0, batch_split_0 ,outbatch)  # [K, H, N, 2]
            return full_pre_tra
        else:
            train_x_0, train_y_0, nei_list_batch_0, batch_split_0, batch_abs_gt_0, nei_num_batch_0 = self.getTrainX(
                inputs_0, device, 0)
            sour_Seq = self.get_ScaleSeq(train_x_0)
            train_x_1, train_y_1, nei_list_batch_1, batch_split_1, batch_abs_gt_1, nei_num_batch_1 = self.getTrainX(
                inputs_1, device, 1)
            tar_Seq = self.get_ScaleSeq(train_x_1)

            sour_embedded, tar_embedded = [],[]
            for index in self.scale_array:
                index = index.item()
                if index == 1:
                    sour_embedded.append(self.Scale_one_Encoder(sour_Seq[index], 's'))
                    tar_embedded.append(self.Scale_one_Encoder(tar_Seq[index], 's'))
                elif index == 2:
                    sour_embedded.append(self.Scale_Two_Encoder(sour_Seq[index], 'm'))
                    tar_embedded.append(self.Scale_Two_Encoder(tar_Seq[index], 'm'))
                else:
                    sour_embedded.append(self.Scale_Four_Encoder(sour_Seq[index], 'm'))
                    tar_embedded.append(self.Scale_Four_Encoder(tar_Seq[index], 'm'))

            x_encoded_dense_0, hidden_state_unsplited_0, cn_0 =  self.Scale_Whole_Encoder.forward(train_x_0, 'l')  # [N, D], [N, D]
            x_encoded_dense_1, hidden_state_unsplited_1, cn_1 =  self.Scale_Whole_Encoder.forward(train_x_1, 'l')  # [N, D], [N, D]

            sour_embedded.append(hidden_state_unsplited_0.unsqueeze(1))
            tar_embedded.append(hidden_state_unsplited_1.unsqueeze(1))

            align_loss = self.mmdAndwassersteinDis(sour_embedded, tar_embedded)

            hidden_state_global_0  = torch.zeros_like(hidden_state_unsplited_0, device=device)
            hidden_state_global_1  = torch.zeros_like(hidden_state_unsplited_1, device=device)
            cn_global_0 = torch.zeros_like(cn_0, device=device)
            cn_global_1= torch.zeros_like(cn_1, device=device)

            split_refle = balance_mapping(len(batch_split_0), len(batch_split_1))
            for index, batch in enumerate(split_refle):
                [b0,b1] = batch
                left_0, right_0 = batch_split_0[b0][0], batch_split_0[b0][1]
                left_1, right_1 = batch_split_1[b1][0], batch_split_1[b1][1]
                element_states_0 = hidden_state_unsplited_0[left_0: right_0]  # [N, D]
                element_states_1 = hidden_state_unsplited_1[left_1: right_1]  # [N, D]
                cn_state_0 = cn_0[left_0: right_0]  # [N, D]
                cn_state_1 = cn_1[left_1: right_1]  # [N, D]
                if element_states_0.shape[0] != 1:
                    hidden_state_global_0[left_0: right_0], cn_global_0[left_0: right_0] = self.Graph_conv_interaction(batch_abs_gt_0, left_0, right_0, element_states_0,
                                                                                              cn_state_0,nei_list_batch_0,b0, device)
                else:
                    hidden_state_global_0[left_0: right_0] = element_states_0
                    cn_global_0[left_0: right_0] = cn_state_0
                if element_states_1.shape[0] != 1:
                    hidden_state_global_1[left_1: right_1], cn_global_1[left_1: right_1] = self.Graph_conv_interaction(batch_abs_gt_1, left_1, right_1, element_states_1,
                                                                                              cn_state_1,nei_list_batch_1, b1, device)
                else:
                    hidden_state_global_1[left_1: right_1] = element_states_1
                    cn_global_1[left_1: right_1] = cn_state_1

            sour_aggrate = self.scale_fusion(sour_embedded[:-1], hidden_state_global_0)
            tar_aggrate = self.scale_fusion(tar_embedded[:-1], hidden_state_global_1)

            sour_hidden = self.multiScale_sampling(sour_aggrate)

            mdn_out = self.Laplacian_Decoder.forward(x_encoded_dense_0, sour_hidden, cn_global_0)
            predict_loss, variety_loss, full_pre_tra = self.mdn_loss(train_y_0.permute(2, 0, 1), mdn_out,ifvisualize)  # [K, H, N, 2]

            total_loss = self.dyw_loss(predict_loss, align_loss, variety_loss)
            print('predict_loss={:.5f} | align_loss={:.5f} | variety_loss={:.5f} | total_loss={:.5f}' \
                  .format(predict_loss,align_loss.item(),variety_loss, total_loss.item()))
            return total_loss, full_pre_tra

    def multiScale_sampling(self, sour_data):
        sour_updated = []
        for cur_sour in sour_data:
            # cur_tar = cur_tar.cpu().detach().numpy()
            #mean = np.mean(cur_tar, axis=0)  # axis=0 表示沿着第一个维度（n）计算均值
            # 计算协方差矩阵
            #cov_matrix = np.cov(cur_tar, rowvar=False).astype(np.float32)  # rowvar=False 表示每一行是一个观测，每一列是一个变量
            # 根据高斯分布参数采样出一个48维的向量
            sampled_vector = np.random.normal(loc=0, scale=1, size=self.args.hidden_size).astype(np.float32)
            sampled_vector_list = torch.from_numpy(sampled_vector).repeat(cur_sour.shape[0],1).to('cuda')
            uni_sample = torch.torch.rand(1).to('cuda')
            z_mix= cur_sour + uni_sample * sampled_vector_list
            z_out = torch.cat((cur_sour,z_mix),1)
            sour_updated.append(z_out)
        stack_up = torch.stack(sour_updated, dim=0)
        update_fusion = self.Attention_AgScale2(stack_up.reshape(stack_up.shape[1], stack_up.shape[0], stack_up.shape[2]))
        return update_fusion
    def scale_fusion(self, data, la_data):
        fusion_array = []
        for cur_scale in data:
            fusion_array.append(self.Attention_AgScale(cur_scale))
        fusion_array.append(la_data)
        return fusion_array

    def gaussian_rbf_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = (2 * kernel_mul * L2_distance.mean() + kernel_num) ** -0.5

        bandwidth = bandwidth * torch.ones(L2_distance.size()).cuda()
        kernel_exponent = (-L2_distance * bandwidth).exp()

        return kernel_exponent

    def mmdAndwassersteinDis(self, sour_data, tar_data, kernel_mul=2.0, kernel_num=5, fix_sigma=None,epsilon=0.1):
        mmd_loss = 0#汇总mmd损失
        wasserstein_dis = 0#汇总was距离
        for cur_sour,cur_tar in zip(sour_data,tar_data):
            cur_sour_ag = cur_sour.permute(1,0,2)
            cur_tar_ag = cur_tar.permute(1,0,2)
            seq_loss = 0
            seq_dis = 0 #seq_wasserstein_distance
            for x, y in zip(cur_sour_ag, cur_tar_ag):
                #------------------mdd-------------------
                source_n, target_n = x.size()[0], y.size()[0]
                kernels = self.gaussian_rbf_kernel(x, y, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                              fix_sigma=fix_sigma)
                XX = kernels[:source_n, :source_n]
                XY = kernels[:source_n, source_n:]
                YX = kernels[source_n:, :source_n]
                YY = kernels[source_n:, source_n:]

                XX_det = XX.mean()
                YY_det = YY.mean()
                YX_DET = YX.mean()
                XY_DET = XY.mean()

                cur_seq_mmd = XX_det + YY_det - YX_DET - XY_DET

                seq_loss += cur_seq_mmd
                #------------wasserstein------------------
                dist = torch.cdist(x, y, p=2)  # Compute pairwise Euclidean distance
                # Sinkhorn distance regularization
                K = torch.exp(-dist / epsilon)  # Kernel matrix
                K1 = K / K.sum(dim=1, keepdim=True)  # Normalize
                cur_seq_distance = torch.mean(torch.sum(K1 * dist, dim=1))
                seq_dis += cur_seq_distance
            mmd_loss += seq_loss
            wasserstein_dis += seq_dis
        align_loss = self.Attention_TwoValue(mmd_loss, wasserstein_dis)
        return align_loss

    def get_ScaleSeq(self, data):
        scale_array = self.scale_array
        results = {}
        for length in scale_array:
            num_full_blocks = len(data[0,0,:]) // length# 计算需要多少个完整的块和剩余的块
            remaining = len(data[0,0,:]) % length
            current_result = []# 创建一个空列表来存储当前长度的划分
            for i in range(num_full_blocks):# 添加完整的块
                current_result.append(data[:,:,i * length:(i + 1) * length])
            if remaining > 0:# 如果有剩余，添加剩余的元素
                borrowed_elements = data[:,:,-length:]
                current_result.append(borrowed_elements)# 将剩余的元素和借来的元素合并
            current_cat = torch.stack(current_result, dim=1)
            results[length.item()] = current_cat# 将当前长度的结果添加到总结果字典中
        return results

    def Graph_conv_interaction(self, batch_abs_gt, left, right, element_states, cn_state, nei_list_batch, b, device):
        corr = batch_abs_gt[self.args.obs_length - 1, left: right, :2].repeat(
            element_states.shape[0], 1, 1)  # [N, N, D]
        corr_index = corr.transpose(0, 1) - corr  # [N, N, D]
        speed, angle = get_speed_and_angle(batch_abs_gt[:,left: right,:],self.args.obs_length)
        speed_index = speed.transpose(0, 1) - speed#邻里相对速度
        angle_index = angle.transpose(0, 1) - angle#邻里相对角度
        nei_index = torch.tensor(nei_list_batch[b][self.args.obs_length - 1], device = device)  # [N, N]
        for i in range(self.args.message_layers):
            element_states, cn_state = self.Global_interaction[i](corr_index, speed_index, angle_index,
                                                                       nei_index, element_states, cn_state)
        return element_states, cn_state

    def getTrainX(self,inputs,device,setNum):
        batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split = inputs  # #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H], [B, 2]
        self.batch_norm_gt[setNum] = batch_norm_gt
        #得到时间步之间的运动偏移量，以观测轨迹最后一个时间步为原点
        train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length - 1, :,:]  # [H, N, 2]

        train_x = train_x.permute(1, 2, 0)  # [N, 2, H]
        train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0)  # [N, 2, H]
        self.pre_obs[setNum] = batch_norm_gt[1:self.args.obs_length]
        return train_x,train_y,nei_list_batch,batch_split,batch_abs_gt,nei_num_batch

    def mdn_loss(self, y, y_prime,ifvisualize, batch_abs_gt=None, batch_split=None, batch=None):
        batch_size=y.shape[1]
        y = y.permute(1, 0, 2)  #[N, H, 2]
        # [F, N, H, 2], [F, N, H, 2], [N, F]
        out_mu, out_sigma, out_pi = y_prime
        variety_loss = self.diversity_loss(out_mu.permute(1,0,2,3))
        y_hat = torch.cat((out_mu, out_sigma), dim=-1)
        reg_loss, cls_loss = 0, 0
        full_pre_tra = []
        l2_norm = (torch.norm(out_mu - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(batch_size)]
        reg_loss = reg_loss + self.reg_loss(y_hat_best, y)
        soft_target = F.softmax(-l2_norm / self.args.pred_length, dim=0).t().detach() # [N, F]
        cls_loss = cls_loss + self.cls_loss(out_pi, soft_target)
        predict_loss = reg_loss + cls_loss
        #best ADE
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs[0],sample_k), axis=0))
        # best FDE
        l2_norm_FDE = (torch.norm(out_mu[:,:,-1,:] - y[:,-1,:], p=2, dim=-1) )  # [F, N]
        best_mode = l2_norm_FDE.argmin(dim=0)
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs[0],sample_k), axis=0))
        return predict_loss, variety_loss.item(), full_pre_tra

    def diversity_loss(self, trajectories):
        batch_num, num_predictions, time_steps, _ = trajectories.shape
        # 初始化最小距离为一个很大的数
        min_distance = torch.tensor(float('inf')).to(trajectories.device)
        # 遍历每对轨迹
        for i in range(num_predictions):
            for j in range(i + 1, num_predictions):
                # 计算每对轨迹在每个时间步的欧几里得距离
                distance = torch.sum(torch.norm(trajectories[:, i, :, :] - trajectories[:, j, :, :], p=2, dim=-1),
                                     dim=-1)
                # 更新最小距离
                min_distance = torch.min(min_distance, distance)

        # 多样性损失是负的最小距离的平均值
        variety_loss = -torch.mean(min_distance)
        return variety_loss
    
