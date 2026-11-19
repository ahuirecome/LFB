import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, activation, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.in_layer = nn.Linear(in_dim, self.hidden_dim)

        self.mid_layers = nn.ModuleList()
        for k in range(self.num_layers - 2):
            self.mid_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.out_layer = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, x):
        x = self.activation(self.in_layer(x))
        for k in range(len(self.mid_layers)):
            x = self.activation(self.mid_layers[k](x) + x)
        x = self.out_layer(x)
        return x

    def sample(self, x):
        """
        重参数化采样 z ~ N(mu, sigma2)
        """
        mu, sigma2 = self.forward(x)
        sigma = torch.sqrt(sigma2)
        dist = Normal(loc=mu, scale=sigma)
        z = dist.rsample()   # rsample 支持梯度
        return z#, mu, sigma2

    def get_setting(self):
        out = f"dim({self.hidden_dim})"
        if self.activation == torch.cos:
            out = out + "_cos"
        elif self.activation == torch.tanh:
            out = out + "_tanh"
        elif type(self.activation) == nn.ReLU:
            out = out + "_relu"
        elif self.activation == F.relu:
            out = out + "_relu"
        elif type(self.activation) == nn.LeakyReLU:
            out = out + "_leakyrelu"
        return out

    def get_flat_params(self):
        return torch.cat([p.view(-1).detach() for p in self.parameters()]).cpu().numpy()

    def set_flat_params(self, flat_params):
        ptr = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(torch.from_numpy(flat_params[ptr:ptr + size]).view_as(p))
            ptr += size


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianHeadNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 两个头：分别输出 mu 和 raw_sigma2
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma2_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [B, in_dim]
        return: mu, sigma2
        mu: [B,1]
        sigma2: [B,1] > 0
        """
        h = self.backbone(x)
        mu = self.mu_head(h)
        raw_sigma2 = self.sigma2_head(h)
        # softplus 保证方差严格大于0，避免数值问题
        sigma2 = F.softplus(raw_sigma2) + 1e-6
        return mu, sigma2

    def sample(self, x):
        """
        重参数化采样 z ~ N(mu, sigma2)
        """
        mu, sigma2 = self.forward(x)
        sigma = torch.sqrt(sigma2)
        dist = Normal(loc=mu, scale=sigma)
        z = dist.rsample()   # rsample 支持梯度
        return z, mu, sigma2

    def sample_and_cdf_transform(self, x):
        """
        【你前面关心的】z 经过标准正态CDF(Phi)映射到 (0,1)
        u = Phi(z)，仅当 z~N(0,1) 时 u~U(0,1)
        这里 z 是 N(mu, sigma2)，所以只是变换，不是天然均匀！
        """
        z, mu, sigma2 = self.sample(x)
        sigma = torch.sqrt(sigma2)
        # 标准化 z 再送入标准正态CDF
        z_std = (z - mu) / sigma
        u = torch.special.ndtr(z_std)  # ndtr = Phi
        return u, z, mu, sigma2


if __name__ == "__main__":
    # 测试
    in_d = 10
    net = GaussianHeadNet(in_dim=in_d, hidden_dim=64)
    batch_size = 32
    x = torch.randn(batch_size, in_d)

    z, mu, sigma2 = net.sample(x)
    u, _, _, _ = net.sample_and_cdf_transform(x)

    print("mu shape:", mu.shape)
    print("sigma2 shape:", sigma2.shape)
    print("sampled z shape:", z.shape)
    print("u (after Phi) shape:", u.shape)
    print("u range: [%.4f, %.4f]" % (u.min().item(), u.max().item()))
