import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
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
    

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))  # (-2, 2)，和倒立摆的动作空间对应上了
        std = F.softplus(self.fc_std(x))  # 使用softplus将x转换为大于0
        return mu, std  # 高斯分布的均值和标准差


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

def compute_advantage(gamma: float, lmbda: float, td_delta: torch.Tensor) -> torch.Tensor:
    """广义优势估计GAE"""
    td_delta = td_delta.detach().numpy()
    advantage_list: list[float] = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.asarray(advantage_list), dtype=torch.float)


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
        """计算黑塞矩阵和向量vector的乘积"""
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
        x = torch.zeros_like(grad)  # 初始化为0
        r = grad.clone()  # 初始化为g
        p = grad.clone()  # 初始化为g
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)  # H × p
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha *Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x  # H^-1 × g
            
    def compute_surrogate_obj(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            advantage: torch.Tensor,
            old_log_probs: torch.Tensor,
            actor: torch.nn.Module
    ) -> torch.Tensor:
        """计算策略目标(代理目标)"""
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)  # 重要性采样
        return torch.mean(ratio * advantage)  # 重要性采样(策略项) * 优势项

    def line_search(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            advantage: torch.Tensor,
            old_log_probs: torch.Tensor,
            old_action_dists: torch.distributions.Categorical,
            max_vec: torch.Tensor
    ) -> torch.Tensor:
        """线性搜索"""
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())  # 把 actor 的所有参数拉平成一个长向量
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)  # 旧策略代回计算代理目标，相当于E[A]，接近0
        for i in range(15):
            coef = self.alpha**i  # 搜索步长
            new_para = old_para + coef * max_vec  # 新策略
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(new_actor(states))  # 新策略根据states采样动作
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))  # 计算旧策略和新策略的kl散度
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)  # 新策略的代理目标
            if new_obj > old_obj and kl_div < self.kl_constraint:  # 当新策略的代理目标大于E[A]，并且kl散度不超过阈值，即新策略的最大值
                return new_para
        return old_para  # 如果搜索完所有情况都不满足条件，返回旧策略，代表策略不更新
    
    def policy_learn(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            old_action_dists: torch.distributions.Categorical,
            old_log_probs: torch.Tensor,
            advantage: torch.Tensor
    ) -> None:
        """更新策略函数"""
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)  # 代理目标
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())  # 一阶梯度g
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()  # 展开每一层参数对应的梯度张量，然后拼接为一维向量，并且从计算图中分离，不做反向传播
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)  # 二阶梯度H的逆乘以一阶梯度，即TRPO的策略梯度方向x
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)  # H × x
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))  # 根号下二倍的kl距离除以x^{T}Hx+1e-8,即策略梯度方向前面的部分
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists, descent_direction * max_coef)  # 线性搜索合适的步长，用来计算新的策略参数
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())  # 把展开后的一维参数，按原来的层结构装回去

    def update(self, transition_dict: TransitionDict) -> None:
        states = torch.tensor(np.asarray(transition_dict['states']), dtype=torch.float, device=self.device)
        actions = torch.tensor(transition_dict['actions'], device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float, device=self.device).view(-1, 1)
        next_states = torch.tensor(np.asarray(transition_dict['next_states']), dtype=torch.float, device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float, device=self.device).view(-1, 1)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)  # 实现策略函数的更新


class TRPOContinuous:
    """ 处理连续动作的TRPO算法 """
    def __init__(
                self, state_dim: int, hidden_dim: int, action_dim: int, 
                lmbda: float, kl_constraint: float, alpha: float, critic_lr: float, gamma: float, device: torch.device
        ) -> None:
        # 策略网络参数不需要优化器更新
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.device = device

    def take_action(self, state: np.ndarray) -> list[float]:
        state = torch.tensor(np.asarray(state)[None, :], dtype=torch.float, device=self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]

    def hessian_matrix_vector_product(
            self,
            states: torch.Tensor,
            old_action_dists: torch.distributions.Normal,
            vector: torch.Tensor,
            damping: float=0.1
    ) -> torch.Tensor:
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector + damping * vector  # 加阻尼项0.1Iv，让矩阵更稳定

    def conjugate_gradient(
            self,
            grad: torch.Tensor,
            states: torch.Tensor,
            old_action_dists: torch.distributions.Normal
    ) -> torch.Tensor:
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            advantage: torch.Tensor,
            old_log_probs: torch.Tensor,
            actor: torch.nn.Module
    ) -> torch.Tensor:
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            advantage: torch.Tensor,
            old_log_probs: torch.Tensor,
            old_action_dists: torch.distributions.Normal,
            max_vec: torch.Tensor
    ) -> torch.Tensor:
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        for i in range(15):
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            old_action_dists: torch.distributions.Normal,
            old_log_probs: torch.Tensor,
            advantage: torch.Tensor
    ) -> None:
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists, descent_direction * max_coef)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

    def update(self, transition_dict: TransitionDict) -> None:
        states = torch.tensor(np.asarray(transition_dict['states']), dtype=torch.float, device=self.device)
        actions = torch.tensor(transition_dict['actions'], device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float, device=self.device).view(-1, 1)
        next_states = torch.tensor(np.asarray(transition_dict['next_states']), dtype=torch.float, device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float, device=self.device).view(-1, 1)

        rewards = (rewards + 8.0) / 8.0  # 对奖励进行修改,方便训练,改为0到1
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = old_action_dists.log_prob(actions)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)


def runTRPO() -> None:
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    critic_lr = 1e-2
    kl_constraint = 0.0005
    alpha = 0.5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    np.random.seed(42)
    env.reset(seed=42)
    env.action_space.seed(42)
    torch.manual_seed(42)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = TRPO(state_dim, hidden_dim, action_dim, lmbda, kl_constraint, alpha, critic_lr, gamma, device)
    return_list = train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'TRPO on {env_name}')
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'TRPO on {env_name}')
    plt.show()

def runTRPOContinuous() -> None:
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    critic_lr = 1e-2
    kl_constraint = 0.00005
    alpha = 0.5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    np.random.seed(42)
    env.reset(seed=42)
    env.action_space.seed(42)
    torch.manual_seed(42)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TRPOContinuous(state_dim, hidden_dim, action_dim, lmbda, kl_constraint, alpha, critic_lr, gamma, device)
    return_list = train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'TRPOContinuous on {env_name}')
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'TRPOContinuous on {env_name}')
    plt.show()


if __name__ == '__main__':
    runTRPO()
    runTRPOContinuous()
