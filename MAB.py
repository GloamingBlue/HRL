import numpy as np
from matplotlib import pyplot as plt
from typing import List


class BernoulliBandit:
    """伯努利多臂老虎机, K为拉杆个数, 可视为无状态的强化学习(stateless reinforcement learning)"""
    def __init__(self, K: int) -> None:
        self.probs = np.random.uniform(size=K)  # probs是获奖概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k: int) -> int:
        """按照拉下拉杆k的获得奖励的概率奖励"""
        return 1 if np.random.rand() < self.probs[k] else 0
    

class Solver:
    """多臂老虎机求解类基本框架"""
    def __init__(self, bandit: BernoulliBandit) -> None:
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累计懊悔
        self.actions = []  # 记录每一步的动作
        self.regrets = []  # 记录每一步的累计懊悔
        self.R = 0  # 累计奖励

    def update_regret(self, k: int) -> None:
        """计算累计懊悔并保存"""
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self) -> None:
        """返回当前动作选择哪一根拉杆, 由每个具体的策略实现, 常见的有1.ϵ-贪心算法 2.上置信界算法 3.汤普森采样算法"""
        raise NotImplementedError
    
    def run(self, num_steps: int) -> None:
        """运行指定次数的策略"""
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    """ϵ-贪心算法"""
    def __init__(self, bandit: BernoulliBandit, epsilon: float=0.01, init_prob: float=1.0) -> None:  # init_prob设置为1.0很关键，或者0.5，因为需要满足初始值要等于 “奖励的先验期望”且避免初始决策偏置
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)  # 初始化所有拉杆的期望奖励估值

    def run_one_step(self) -> int:
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 以ϵ概率随机拉杆
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.R += r
        self.estimates[k] += (r - self.estimates[k]) / (self.counts[k] + 1)
        return k


class DecayingEpsilonGreedy(Solver):
    """epsilon值随时间衰减的ϵ-贪心算法"""
    def __init__(self, bandit: BernoulliBandit, init_prob: float=1.0) -> None:  # ϵ随时间反比例衰减
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self) -> int:
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.R += r
        self.estimates[k] += (r - self.estimates[k]) / (self.counts[k] + 1)
        return k


class UCB(Solver):
    """UCB算法,上置信界算法"""
    def __init__(self, bandit: BernoulliBandit, coef: float, init_prob: float=1.0) -> None:  # 设置p为随时间反比例衰减
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self) -> int:
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / 2 * (self.counts + 1))
        k = np.argmax(ucb)  # 选择上置信界最大的拉杆
        r = self.bandit.step(k)
        self.R += r
        self.estimates[k] += (r - self.estimates[k]) / (self.counts[k] + 1)
        return k


class ThompsonSampling(Solver):
    """汤普森采样算法"""
    def __init__(self, bandit: BernoulliBandit) -> None:  # 使用Beta分布建模
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 表示每根拉杆奖励为0的次数

    def run_one_step(self) -> int:
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)
        self.R += r
        self._a[k] += r  # 更新次数,r=1时增加_a[k]次数,r=0时增加_b[k]次数
        self._b[k] += 1 - r
        return k


def plot_results(solvers: List[Solver], solver_names: List[str]) -> None:
    """绘制累计懊悔随时间变化的图像, solvers为策略算法列表, solver_names为策略算法名称列表"""
    for i, solver in enumerate(solvers):
        plt.plot(range(len(solver.regrets)), solver.regrets, label=solver_names[i])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title(f'{solvers[0].bandit.K}-armed bandit')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print(f"随机生成了一个{K}臂伯努利老虎机")
    print(f"获奖概率最大的拉杆为{bandit_10_arm.best_idx}号, 获奖概率为{bandit_10_arm.best_prob}")

    S = 5000
    epsilons = [1e-4, 0.01, 0.1, 0.15, 0.2]
    epsilon_names = [f'epsilon={e}' for e in epsilons]
    epsilon_greedy_solver_list = [EpsilonGreedy(bandit=bandit_10_arm, epsilon=e) for e in epsilons]

    epsilon_names.append('decaying epsilon')
    epsilon_greedy_solver_list.append(DecayingEpsilonGreedy(bandit=bandit_10_arm))
    epsilon_names.append('UCB')
    epsilon_greedy_solver_list.append(UCB(bandit=bandit_10_arm, coef=0.1))  # coef越小效果越好
    epsilon_names.append('thompson sampling')
    epsilon_greedy_solver_list.append(ThompsonSampling(bandit=bandit_10_arm))

    for i, solver in enumerate(epsilon_greedy_solver_list):
        solver.run(S)
        if i < len(epsilon_names) - 3:
            print(f'{epsilons[i]} ϵ-贪心算法的累计懊悔为:{solver.regret}')
            print(f'{epsilons[i]} {S}步的累计奖励为{solver.R}')
        elif i == len(epsilon_names) - 3:
            print(f'Decaying ϵ-贪心算法的累计懊悔为:{solver.regret}')
            print(f'Decaying ϵ-贪心算法{S}步的累计奖励为{solver.R}')
        elif i == len(epsilon_names) - 2:
            print(f'UCB算法的累计懊悔为:{solver.regret}')
            print(f'UCB算法{S}步的累计奖励为{solver.R}')
        else:
            print(f'汤普森采样算法的累计懊悔为:{solver.regret}')
            print(f'汤普森采样算法{S}步的累计奖励为{solver.R}')
    plot_results(epsilon_greedy_solver_list, epsilon_names)
