import numpy as np
import MDP
from sympy import *

class RL2:
    def __init__(self, mdp, sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs:
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def sampleSoftmaxPolicy(self, policyParams, state):
        '''从随机策略中采样单个动作的程序，通过以下概率公式采样
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        本函数将被reinforce()调用来选取动作

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs:
        action -- sampled action

        提示：计算出概率后，可以用np.random.choice()，来进行采样
        '''

        # 获取该状态下所有动作的参数
        state_params = policyParams[:, state]

        # 计算每个动作的softmax概率
        exp_values = np.exp(state_params)  # 每个动作的指数值
        probs = exp_values / np.sum(exp_values)  # 归一化，得到概率分布

        # 使用np.random.choice按照概率分布进行采样
        action = np.random.choice(len(probs), p=probs)

        return action



    def epsilonGreedyBandit(self, nIterations):
        '''Epsilon greedy 算法 for bandits (假设没有折扣因子).
        Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        reward_list = []

        actionCounts = np.zeros(self.mdp.nActions)
        for iteration in range(1, nIterations + 1):
            # 计算epsilon
            epsilon = 1 / iteration
            
            # 探索（以epsilon概率选择一个随机动作）或者利用（选择当前最优的动作）
            if np.random.rand() < epsilon:
                # 随机选择一个动作（探索）
                action = np.random.choice(self.mdp.nActions)
            else:
                # 选择平均奖励最高的动作（利用）
                action = np.argmax(empiricalMeans)
            
            # 获取该动作的奖励
            reward = self.mdp.R[action]

            if np.random.rand() < reward:
                reward_list.append(1)
                reward = 1
            else:
                reward_list.append(0)
                reward = 0
            
            # 更新动作选择次数
            actionCounts[action] += 1
            
            # 更新每个动作的平均奖励
            empiricalMeans[action] += (reward - empiricalMeans[action]) / actionCounts[action]
        reward_list = np.array(reward_list)
        return empiricalMeans, reward_list


    def thompsonSamplingBandit(self, prior, nIterations, k=1):
        '''Thompson sampling 算法 for Bernoulli bandits (假设没有折扣因子)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards


        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)

        提示：根据beta分布的参数，可以采用np.random.beta()进行采样
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        # 初始化Beta分布的参数（每个动作的alpha和beta）
        alpha = prior[:, 0].copy()
        beta = prior[:, 1].copy()

        reward_list = []

        for iteration in range(nIterations):
            # Thompson Sampling: 对每个动作从Beta分布中进行k次采样
            sampled_rewards = []
            for i in range(len(alpha)):
                sampled_rewards.append(np.random.beta(alpha[i], beta[i], k))
            sampled_rewards = np.concatenate(sampled_rewards, axis=0)
            # 选择平均奖励最大的动作
            action = np.argmax(sampled_rewards)
            
            # 获取该动作的奖励
            reward = self.mdp.R[action]

            # 更新Beta分布的参数
            if np.random.rand() < reward:
                alpha[action] += 1
                reward_list.append(1)
            else:
                beta[action] += 1
                reward_list.append(0)
            
            # 更新每个动作的平均奖励
            empiricalMeans[action] += (reward - empiricalMeans[action]) / (iteration + 1)
        reward_list = np.array(reward_list)
        return empiricalMeans, reward_list


    def UCBbandit(self, nIterations):
        '''Upper confidence bound 算法 for bandits (假设没有折扣因子)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        # 初始化每个动作被选择的次数
        actionCounts = np.zeros(self.mdp.nActions)
        
        reward_list = []

        # 拉动每个动作一次，确保每个动作都至少被选择一次
        for action in range(self.mdp.nActions):
            reward = self.mdp.R[action]
            if np.random.rand() < reward:
                reward_list.append(1)
                reward = 1
            else:
                reward_list.append(0)
                reward = 0
            actionCounts[action] += 1
            empiricalMeans[action] = reward  # 初始的平均奖励就是第一次的奖励

        # 进行后续的nIterations次动作选择
        for iteration in range(self.mdp.nActions, nIterations):
            # 计算UCB值
            ucb_values = empiricalMeans + np.sqrt(2 * np.log(iteration) / actionCounts)
            
            # 选择UCB值最大的动作
            action = np.argmax(ucb_values)
            
            # 获取该动作的奖励
            reward = self.mdp.R[action]
            if np.random.rand() < reward:
                reward_list.append(1)
                reward = 1
            else:
                reward_list.append(0)
                reward = 0

            # 更新选择次数
            actionCounts[action] += 1
            
            # 更新该动作的平均奖励
            empiricalMeans[action] += (reward - empiricalMeans[action]) / actionCounts[action]
        reward_list = np.array(reward_list)
        return empiricalMeans, reward_list

    def reinforce(self, s0, initialPolicyParams, nEpisodes, nSteps):
        '''reinforce 算法，学习到一个随机策略，建模为：
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        上面的sampleSoftmaxPolicy()实现该方法，通过调用sampleSoftmaxPolicy(policyParams,state)来选择动作
        并且同学们需要根据上课讲述的REINFORCE算法，计算梯度，根据更新公式，完成策略参数的更新。
        其中，超参数：折扣因子gamma=0.95，学习率alpha=0.01

        Inputs:
        s0 -- 初始状态
        initialPolicyParams -- 初始策略的参数 (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs:
        policyParams -- 最终策略的参数 (array of |A|x|S| entries)
        rewardList --用于记录每个episodes的累计折扣奖励 (array of |nEpisodes| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policyParams = initialPolicyParams
        rewardList = []
        gamma = 0.95  # 折扣因子
        alpha = 0.01  # 学习率

        # 进行nEpisodes次训练
        for episode in range(nEpisodes):
            state = s0
            trajectory = []
            totalReward = 0
            
            # 生成一个回合的轨迹
            for step in range(nSteps):
                action = self.sampleSoftmaxPolicy(policyParams, state)
                reward, nextState = self.sampleRewardAndNextState(state, action)
                trajectory.append((state, action, reward))
                totalReward += reward * (gamma ** step)
                state = nextState
            
            rewardList.append(totalReward)
            
            # 计算并应用梯度更新
            for t, (state, action, _) in enumerate(trajectory):
                # 计算从时间步 t 开始的折扣奖励 G_t
                G_t = sum([traj[2] * (gamma ** i) for i, traj in enumerate(trajectory[t:])])
                
                # 计算当前状态下的策略概率
                state_params = policyParams[:, state]
                exp_scores = np.exp(state_params)
                probs = exp_scores / np.sum(exp_scores)
                
                # 构建梯度向量 ∇θ log π(a|s)
                grad_log_pi = -probs.copy()
                grad_log_pi[action] += 1  # 因为 d log pi(a|s) / d theta_a = 1 - pi(a|s)
                
                # 更新所有动作在当前状态下的策略参数
                policyParams[:, state] += alpha * grad_log_pi * G_t
        
        return policyParams, rewardList
