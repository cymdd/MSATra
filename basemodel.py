import torch
import torch.nn as nn
import torch.nn.functional as F
from laplace_decoder import MLPDecoder,GRUDecoder

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            # print("LSTM------",m.named_parameters())
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)  # initializing the lstm bias with zeros
        else:
            print(m, "************")


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        device = x.device
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight.to(device) * x + self.bias.to(device)

        # return self.weight.to('cpu') * x.to('cpu') + self.bias.to('cpu')


class MLP_gate(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP_gate, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

#聚合mmd和wasdis
class Attention_TwoValue(nn.Module):
    def __init__(self):
        super(Attention_TwoValue, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x1, x2):
        # 将两个标量值合并为一个向量
        combined = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0)), dim=0)
        # 计算注意力权重，使用sigmoid函数确保权重在0和1之间
        attention_weight = torch.sigmoid(self.linear(combined))
        # 使用注意力权重来融合两个值
        output = attention_weight * x1 + (1 - attention_weight) * x2
        return output
#聚合尺度
class Attention_AgScale(nn.Module):
    def __init__(self, feature_dim):
        super(Attention_AgScale, self).__init__()
        self.feature_dim = feature_dim
        # 全连接层，用于将特征维度映射到1
        self.weight_fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        s_shape = x.shape[1]
        # 计算每个向量的注意力分数
        attention_scores = self.weight_fc(x.view(-1, self.feature_dim))  # 形状变为(420, 1)
        attention_scores = attention_scores.view(-1, s_shape)  # 形状变为(60, 7)
        # 归一化注意力分数
        attention_scores = F.softmax(attention_scores, dim=-1)  # 形状保持(60, 7)
        # 将注意力分数扩展到原始特征维度
        attention_scores = attention_scores.unsqueeze(-1)  # 形状变为(60, 7, 1)
        # 应用注意力分数
        attended_x = x * attention_scores.expand_as(x)  # 形状变为(60, 7, 48)
        # 对第二个维度进行求和，以合并向量
        attended_x = attended_x.sum(dim=1)  # 形状变为(60, 48)
        return attended_x

#轨迹向量加权求和
class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.W = nn.Linear(feature_dim, 1)  # 权重计算的线性层

    def forward(self, x):
        scores = self.W(x)  # 计算注意力分数
        alpha = F.softmax(scores, dim=0)  # 应用softmax函数获取权重
        return (x * alpha).sum(dim=0)  # 加权求和


class Scale_Encoder(nn.Module):
    def __init__(self, args, scale):
        super(Scale_Encoder, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        if scale == 's':
            self.conv1d=nn.Conv1d(2, self.hidden_size, kernel_size=1, stride=1, padding=0)
        elif scale == 'm':
            self.conv1d = nn.Conv1d(2, self.hidden_size, kernel_size=2, stride=1, padding=1)
        elif scale == 'l':
            self.conv1d = nn.Conv1d(2, self.hidden_size, kernel_size=3, stride=1, padding=1)
        self.mlp1 = MLP(self.hidden_size)
        if scale != 's':
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.args.x_encoder_head, \
                                                       dim_feedforward=self.hidden_size, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.args.x_encoder_layers)
            self.mlp = MLP(self.hidden_size)
            self.lstm = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                bias=True,
                                batch_first=True,
                                dropout=0,
                                bidirectional=False)
        initialize_weights(self.conv1d.modules())
    def forward(self, x, scale):
        if scale != 'l':
            length = x.shape[1]
            x = x.reshape(-1, 2, x.shape[3])
        self.x_dense=self.conv1d(x).permute(0, 2, 1) #[N, H, dim]
        self.x_dense=self.mlp1(self.x_dense) + self.x_dense #[N, H, dim]
        if scale == 's':
            return self.x_dense.view(-1, length, self.hidden_size)
        self.x_dense_in = self.transformer_encoder(self.x_dense) + self.x_dense  # [N, H, D]
        output, (hn, cn) = self.lstm(self.x_dense_in)
        self.x_state, cn = hn.squeeze(0), cn.squeeze(0)  # # [N, D]
        if scale == 'm':
            return self.x_state.view(-1, length, self.hidden_size)
        xstateMlp = torch.clone(self.x_state)
        self.x_encoded = self.mlp(xstateMlp) + self.x_state  # [N, D]
        return self.x_encoded, self.x_state, cn

class Global_interaction(nn.Module):
    def __init__(self, args):
        super(Global_interaction, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        # Motion gate
        self.ngate = MLP_gate(self.hidden_size * 5, self.hidden_size)  # sigmoid
        # Relative spatial embedding layer
        self.relativeLayer_r = MLP(2, self.hidden_size)
        self.relativeLayer_sa = MLP(1, self.hidden_size)
        # Attention for motion and interaction
        self.WAr = MLP(self.hidden_size * 3, 1)
        self.weight = MLP(self.hidden_size)
        # Multi-Head Cross-Attention for interaction between relative speed, angle, position
        self.multihead_attention = MultiHeadCrossAttention(input_dim=self.hidden_size * 5, output_dim=self.hidden_size * 3,
                                                           num_heads=self.args.num_heads)

    def forward(self, corr_index,speed_index, angle_index, nei_index, hidden_state, cn):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self_h = hidden_state
        self.N = corr_index.shape[0]
        self.D = self.hidden_size
        nei_inputs = self_h.repeat(self.N, 1)  # [N, N, D]
        nei_index_t = nei_index.view(self.N * self.N)  # [N*N]
        corr_t = corr_index.contiguous().view((self.N * self.N, -1))  # [N*N, D]
        speed_t = speed_index.contiguous().view((self.N * self.N, -1))
        angle_t = angle_index.contiguous().view((self.N * self.N, -1))

        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return hidden_state, cn

        # Compute relative spatial embeddings (relative positions)
        r_t = self.relativeLayer_r(corr_t[nei_index_t > 0])  # [N*N, D]
        s_t = self.relativeLayer_sa(speed_t[nei_index_t > 0])
        a_t = self.relativeLayer_sa(angle_t[nei_index_t > 0])
        # Preparing inputs for cross-attention
        inputs_part = nei_inputs[nei_index_t > 0].float()
        hi_t = nei_inputs.view((self.N, self.N, self.hidden_size)).permute(1, 0, 2).contiguous().view(-1,
                                                                                                      self.hidden_size)  # [N*N, D]
        # Concatenate relative position, speed, angle, and hidden state for cross-attention
        combined_features = torch.cat((r_t, s_t, a_t, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]),1)

        # Motion Gate mechanism
        nGate = self.ngate(combined_features).float()  # [N*N, D]

        # Attention mechanism
        Pos_t = torch.full((self.N * self.N, 1), 0, device=device).view(-1).float()
        tt = self.multihead_attention(combined_features,corr_t[nei_index_t > 0],
                                      speed_t[nei_index_t > 0],angle_t[nei_index_t > 0])
        Pos_t[nei_index_t > 0] = tt
        Pos = Pos_t.view((self.N, self.N))
        Pos[Pos == 0] = -10000
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)

        # Message Passing
        H = torch.full((self.N * self.N, self.D), 0, device=device).float()
        H[nei_index_t > 0] = inputs_part * nGate
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.D, 1).transpose(0, 1)
        H = H.view(self.N, self.N, -1)  # [N, N, D]
        H_sum = self.weight(torch.sum(H, 1))  # [N, D]

        # Update hidden states
        C = H_sum + cn  # [N, D]
        H = hidden_state + F.tanh(C)  # [N, D]
        return H, C


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_dim = output_dim // num_heads

        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        self.query_layer = nn.Linear(input_dim, output_dim)
        self.key_layer = nn.Linear(input_dim, output_dim)
        self.value_layer = nn.Linear(input_dim, output_dim)

        # Structure-aware bias
        self.spatial_bias_layer = nn.Sequential(
            nn.Linear(2, self.head_dim),  # (relative_x, relative_y)
            nn.ReLU(),
            nn.Linear(self.head_dim, 1)
        )

        # Motion gate
        self.motion_gate_layer = nn.Sequential(
            nn.Linear(2, self.head_dim),  # (velocity_x, velocity_y, angle)
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.Sigmoid()
        )

        self.fc_out = nn.Linear(output_dim, 1)  # 1 attention value per target

    def forward(self, x, rel_pos, rel_vel, rel_angle):
        """
        x: (N, input_dim) - input features for N targets
        rel_pos: (N, 2) - relative position (dx, dy)
        rel_vel: (N, 2) - relative velocity (vx, vy)
        rel_angle: (N, 1) - relative angle
        interaction_mask: (N,) - 1 for active neighbor, 0 for masked (optional)
        """
        N = x.size(0)

        # Linear projections and reshape to (N, num_heads, head_dim)
        Q = self.query_layer(x).view(N, self.num_heads, self.head_dim)
        K = self.key_layer(x).view(N, self.num_heads, self.head_dim)
        V = self.value_layer(x).view(N, self.num_heads, self.head_dim)

        # Add singleton dimension for broadcasting
        Q = Q.unsqueeze(1)  # (N, 1, num_heads, head_dim)
        K = K.unsqueeze(0)  # (1, N, num_heads, head_dim)
        V = V.unsqueeze(0)  # (1, N, num_heads, head_dim)

        # Step 1: Scaled dot-product attention
        scores = (Q * K).sum(-1) / (self.head_dim ** 0.5)  # (N, N, num_heads)

        # Step 2: Add structure-aware bias
        # rel_feat = torch.cat([rel_pos, rel_angle], dim=-1)  # (N, 2)
        spatial_bias = self.spatial_bias_layer(rel_pos).view(N, 1, 1)
        scores = scores + spatial_bias  # Broadcasted to (N, N, num_heads)

        # # Step 3: Apply interaction mask (if provided)
        # if interaction_mask is not None:
        #     mask = interaction_mask.unsqueeze(-1).expand(-1, -1, self.num_heads)  # (N, N, num_heads)
        #     scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 4: Attention weights
        attention = torch.softmax(scores, dim=1)  # softmax over neighbor dim (N)

        # Step 5: Motion gate
        motion_feat = torch.cat([rel_vel, rel_angle], dim=-1)  # (N, 2)
        motion_gate = self.motion_gate_layer(motion_feat).view(1, N, 1)
        attention = attention * motion_gate  # apply motion gating

        # Step 6: Compute attended value
        attention = attention.unsqueeze(-1)
        out = torch.sum(attention * V, dim=1)  # (N, num_heads, head_dim)

        # Step 7: Merge heads and project
        out = out.contiguous().view(N, self.output_dim)
        out = self.fc_out(out)  # (N, 1)
        attention_values = out.squeeze(-1)  # (N,)

        return attention_values

class Laplacian_Decoder(nn.Module):

    def __init__(self,args):
        super(Laplacian_Decoder, self).__init__()
        self.args = args
        if args.mlp_decoder:
            self._decoder = MLPDecoder(args)
        else:
            self._decoder = GRUDecoder(args)

    def forward(self,x_encode, hidden_state, cn):
        mdn_out = self._decoder(x_encode, hidden_state, cn)
        loc, scale, pi = mdn_out  # [F, N, H, 2], [F, N, H, 2], [N, F]
        return (loc, scale, pi)