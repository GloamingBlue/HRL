import random
import time
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from rl_utils import ReplayBuffer, train_off_policy_agent, moving_average, TransitionDict
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


class Qnet(torch.nn.Module):
    """只有一层隐藏层的Q网络"""
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class VAnet(torch.nn.Module):
    """状态价值函数和优势函数的复合网络"""
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # 平均代替最大化，虽然不满足贝尔曼最优方程，但实际应用更加稳定
        return Q


class DQN:
    """DQN算法"""
    def __init__(
            self, state_dim: int, hidden_dim: int, action_dim: int, 
            learning_rate: float, gamma: float, epsilon: float, target_update: int, 
            device: torch.device, dqn_type: str='VanillaDQN'
    ) -> None:
        self.action_dim = action_dim
        if dqn_type == 'DuelingDQN':  # Dueling DQN采取不一样的网络框架
            self.q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 记录更新次数
        self.device = device
        self.dqn_type = dqn_type
        
    def take_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.asarray(state)[None, :], dtype=torch.float, device=self.device)
            action = self.q_net(state).argmax().item()  # 取最大值所在下标
        return action
    
    def max_q_value(self, state: np.ndarray) -> float:
        state = torch.tensor(np.asarray(state)[None, :], dtype=torch.float, device=self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict: TransitionDict) -> None:
        states = torch.tensor(np.asarray(transition_dict['states']), dtype=torch.float, device=self.device)
        actions = torch.tensor(transition_dict['actions'], device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float, device=self.device).view(-1, 1)
        next_states = torch.tensor(np.asarray(transition_dict['next_states']), dtype=torch.float, device=self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:  # 常规的DQN
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


class ConvolutionalQnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim: int, in_channels: int=4) -> None:
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc4(x))
        return self.head(x)


def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)

def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    action_continuous = dis_to_con(action, env, agent.action_dim)
                    next_state, reward, terminated, truncated, _ = env.step([action_continuous])
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list

def runDQN() -> None:
    """运行DQN和卷积DQN"""
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env_name = 'CartPole-v1'
    random.seed(42)
    np.random.seed(42)
    env.reset(seed=42)
    env.action_space.seed(42)
    torch.manual_seed(42)

    env = gym.make(env_name)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    return_list = train_off_policy_agent(
        env, agent, num_episodes, replay_buffer, minimal_size, batch_size
    )
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    render_env = gym.make(env_name, render_mode='human')
    state, _ = render_env.reset()
    done = False
    episode_return = 0.0
    while not done:
        state_tensor = torch.tensor(np.asarray(state)[None, :], dtype=torch.float, device=device)
        action = agent.q_net(state_tensor).argmax().item()
        next_state, reward, terminated, truncated, _ = render_env.step(action)
        time.sleep(0.02)
        state = next_state
        episode_return += reward
        done = terminated or truncated
    print(f"Final greedy episode return (state): {episode_return}")

    pixel_lr = 1e-4
    pixel_epsilon = 0.1
    env = gym.make(env_name, render_mode='rgb_array')
    env = gym.wrappers.AddRenderObservation(env, render_only=True)
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: np.asarray(
            Image.fromarray(obs).convert('L').resize((84, 84), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        ),
        gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8),
    )
    env = gym.wrappers.FrameStackObservation(env, 4)
    replay_buffer = ReplayBuffer(buffer_size)
    action_dim = env.action_space.n
    pixel_agent = DQN(1, hidden_dim, action_dim, pixel_lr, gamma, pixel_epsilon, target_update, device)
    pixel_agent.q_net = ConvolutionalQnet(action_dim, in_channels=4).to(device)
    pixel_agent.target_q_net = ConvolutionalQnet(action_dim, in_channels=4).to(device)
    pixel_agent.target_q_net.load_state_dict(pixel_agent.q_net.state_dict())
    pixel_agent.optimizer = torch.optim.AdamW(pixel_agent.q_net.parameters(), lr=pixel_lr)

    return_list = train_off_policy_agent(
        env, pixel_agent, num_episodes, replay_buffer, minimal_size, batch_size
    )
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Pixel DQN on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Pixel DQN on {}'.format(env_name))
    plt.show()

    render_env = gym.make(env_name, render_mode='rgb_array')
    render_env = gym.wrappers.AddRenderObservation(render_env, render_only=True)
    render_env = gym.wrappers.TransformObservation(
        render_env,
        lambda obs: np.asarray(
            Image.fromarray(obs).convert('L').resize((84, 84), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        ),
        gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8),
    )
    render_env = gym.wrappers.FrameStackObservation(render_env, 4)
    state, _ = render_env.reset()
    done = False
    episode_return = 0.0
    while not done:
        state_tensor = torch.tensor(np.asarray(state)[None, :], dtype=torch.float, device=device)
        action = pixel_agent.q_net(state_tensor).argmax().item()
        next_state, reward, terminated, truncated, _ = render_env.step(action)
        time.sleep(0.02)
        state = next_state
        episode_return += reward
        done = terminated or truncated
    print(f"Final greedy episode return (pixel): {episode_return}")

    render_env.close()
    env.close()

def runDoubleDQN() -> None:
    # DQN
    lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 11  # 将连续动作分成11个离散动作

    random.seed(42)
    np.random.seed(42)
    env.reset(seed=42)
    env.action_space.seed(42)
    torch.manual_seed(42)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size)

    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    # DoubleDQN
    random.seed(42)
    np.random.seed(42)
    env.reset(seed=42)
    env.action_space.seed(42)
    torch.manual_seed(42)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, 'DoubleDQN')
    return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size)

    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Double DQN on {}'.format(env_name))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('Double DQN on {}'.format(env_name))
    plt.show()

def runDuelingDQN() -> None:
    # DQN
    lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 11  # 将连续动作分成11个离散动作

    random.seed(42)
    np.random.seed(42)
    env.reset(seed=42)
    env.action_space.seed(42)
    torch.manual_seed(42)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, 'DuelingDQN')
    return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size)

    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dueling DQN on {}'.format(env_name))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('Dueling DQN on {}'.format(env_name))
    plt.show()


if __name__ == '__main__':
    runDQN()
    runDoubleDQN()
    runDuelingDQN()
