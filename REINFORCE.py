import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rl_utils import train_on_policy_agent, moving_average, TransitionDict
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    

class REINFORCE:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, learning_rate: float, gamma: float, device: torch.device) -> None:
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state: np.ndarray) -> int:  # 根据动作的概率分布随机采样
        state = torch.tensor(np.asarray(state)[None, :], dtype=torch.float, device=self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict: TransitionDict) -> None:
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor(np.asarray(state_list[i])[None, :], dtype=torch.float, device=self.device)
            action = torch.tensor([action_list[i]], device=self.device).view(-1, 1)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward  # 蒙特卡洛采样计算时刻t的动作价值回报
            loss = -log_prob * G  # 时刻t的loss
            loss.backward()  # 回传时刻t损失的梯度，最终会累加每一个时刻的损失的梯度
        self.optimizer.step()  # 累加得到的损失的梯度一起进行梯度更新，和算法流程对应


if __name__ == '__main__':
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    np.random.seed(0)
    env.reset(seed=0)
    env.action_space.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)
    return_list = train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'REINFORCE on {env_name}')
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'REINFORCE on {env_name}')
    plt.show()
