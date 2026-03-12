import gymnasium as gym
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rl_utils import train_on_policy_agent, moving_average, TransitionDict
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim:int) -> None:
        super(PolicyNet, self).__init__()
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
    

class ActorCritic:
    def __init__(
            self, state_dim: int, hidden_dim: int, action_dim: int, 
            actor_lr: float, critic_lr: float, gamma: float, device: torch.device
    ) -> None:
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  # 策略网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 状态价值网络
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state: np.ndarray) -> int:
        state = torch.tensor(np.asarray(state)[None, :], dtype=torch.float, device=self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict: TransitionDict) -> None:
        states = torch.tensor(np.asarray(transition_dict['states']), dtype=torch.float, device=self.device)
        actions = torch.tensor(transition_dict['actions'], device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float, device=self.device).view(-1, 1)
        next_states = torch.tensor(np.asarray(transition_dict['next_states']), dtype=torch.float, device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  # 时序差分目标
        td_delta = td_target - self.critic(states)  # 时序差分误差，实际上是优势函数，大于0代表动作比预期好，反之则比预期差
        log_probs = torch.log(self.actor(states).gather(1, actions))  # 获取策略网络输出的对应动作的对数概率
        actor_loss = torch.mean(-log_probs * td_delta.detach())  # detach优势函数，不让actor的梯度更新也更新critic网络
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))  # 这里detach是将目标值当常数标签来回归，即半梯度，保证训练稳定
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    np.random.seed(0)
    env.reset(seed=0)
    env.action_space.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
    return_list = train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'Actor-Critic on {env_name}')
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'Actor-Critic on {env_name}')
    plt.show()
    env.close()

    render_env = gym.make(env_name, render_mode='human')
    state, _ = render_env.reset()
    done = False
    episode_return = 0.0
    while not done:
        state_tensor = torch.tensor(np.asarray(state)[None, :], dtype=torch.float, device=device)
        action = agent.actor(state_tensor).argmax(dim=1).item()
        next_state, reward, terminated, truncated, _ = render_env.step(action)
        time.sleep(0.02)
        state = next_state
        episode_return += reward
        done = terminated or truncated
    print(f"Final greedy episode return: {episode_return}")
    render_env.close()
