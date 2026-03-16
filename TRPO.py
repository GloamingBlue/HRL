import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
import copy
import matplotlib as plt
from rl_utils import train_on_policy_agent, moving_average, TransitionDict
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


class PolictNet(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolictNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class TRPO:
    def __init__(
            self, state_dim: int, hidden_dim: int, action_dim: int, 
            lmbda: float, kl_constraint: float, alpha: float, critic_lr: float, gamma: float, device: torch.device
    ) -> None:
        # 策略网络参数不需要优化器更新
        self.actor = PolictNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE指数加权平均参数
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.alpha = alpha  # 线性搜索长度超参数
        self.device = device

    def take_action(self, state: np.ndarray) -> int:
        state = torch.tensor(np.asarray(state)[None, :], dtype=torch.float, device=self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def hessian_matrix_vector_product(
            self, states: np.ndarray, old_action_dist: torch.distributions.Categorical, vector: torch.Tensor
    ) -> torch.Tensor:
        """计算黑塞矩阵和一个向量的乘积"""
        new_action_dist = torch.distributions.Categorical(self.actor(states))  # 计算新策略的分布
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dist, new_action_dist))  # 计算旧策略对于新策略的平均kl距离
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)  # 计算kl距离相对于策略的一阶导数，保留计算图，使得一阶导可导，方便计算黑塞矩阵
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())  # 计算一阶导和向量的乘积 相对于策略的二阶导，即黑塞矩阵乘以该向量
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad: torch.Tensor, states: np.ndarray, old_action_dists: torch.distributions.Categorical) -> torch.Tensor:
        """使用共轭梯度法求解方程"""
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dat(p, Hp)
            
