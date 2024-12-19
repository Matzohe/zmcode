import math

import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
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

        reward = self.sampleReward(self.mdp.R[action,state]) #按照当前R值为均值根据高斯密度函数得到随机奖励
        cumProb = np.cumsum(self.mdp.T[action,state,:]) # 输出当前state对应的action的下一个state的累积概率
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0, alpha=0.5):
        '''
        qLearning算法，需要将Epsilon exploration和 Boltzmann exploration 相结合。
        以epsilon的概率随机取一个动作，否则采用 Boltzmann exploration取动作。
        当epsilon和temperature都为0时，将不进行探索。

        Inputs:
        s0 -- 初始状态
        initialQ -- 初始化Q函数 (|A|x|S| array)
        nEpisodes -- 回合（episodes）的数量 (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- 每个回合的步数(steps)
        epsilon -- 随机选取一个动作的概率
        temperature -- 调节 Boltzmann exploration 的参数

        Outputs: 
        Q -- 最终的 Q函数 (|A|x|S| array)
        policy -- 最终的策略
        rewardList -- 每个episode的累计奖励（|nEpisodes| array）
        '''

        Q = initialQ
        action_number = Q.shape[0]
        rewardList = []
        for _i in range(nEpisodes):
            # 每格episode从s0开始，探索轨迹
            current_state = s0
            ep_reward = 0
            for _j in range(nSteps):
                # 进行nSteps次探索，每次探索有epsilon的概率随机选一个动作，有1-epsilon的概率采用 Boltzmann exploration

                if np.random.rand() < epsilon:
                    action = np.random.randint(action_number)
                elif temperature > 0:
                    # 当temperature大于0时，进行玻尔兹曼探索
                    action_prob = np.exp(Q[:, current_state] / temperature) / np.sum(np.exp(Q[:, current_state] / temperature))
                    # 根据softmax函数得到动作概率，选择当前状态下的action
                    action = np.random.choice(action_number, p=action_prob)
                else:
                    # 当两者都为0时，直接进行贪心搜索
                    action = np.argmax(Q[:, current_state])
                
                # 产生下一步的状态，以及执行当前操作的回报
                reward, next_state = self.sampleRewardAndNextState(state=current_state, action=action)
                ep_reward += reward
                # 根据Q-Learning公式更新Q函数
                Q[action, current_state] = Q[action, current_state] + alpha * (reward + self.mdp.discount * np.max(Q[:, next_state]) - Q[action, current_state])
                
                current_state = next_state
            rewardList.append(ep_reward)
        # 使用Q函数得到策略，选取每个状态Q值最大的动作
        policy = np.argmax(Q, axis=0)

        return [Q,policy,rewardList]