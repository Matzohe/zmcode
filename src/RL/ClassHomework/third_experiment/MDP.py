import numpy as np

class MDP:
    '''一个简单的MDP类，它包含如下成员'''

    def __init__(self,T,R,discount):
        '''构建MDP类

        输入:
        T -- 转移函数: |A| x |S| x |S'| array
        R -- 奖励函数: |A| x |S| array
        discount -- 折扣因子: scalar in [0,1)

        构造函数检验输入是否有效，并在MDP对象中构建相应变量'''

        assert T.ndim == 3, "转移函数无效，应该有3个维度"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "无效的转换函数：它具有维度 " + repr(T.shape) + ", 但它应该是(nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "无效的转移函数：某些转移概率不等于1"
        self.T = T
        assert R.ndim == 2, "奖励功能无效：应该有2个维度"
        assert R.shape == (self.nActions,self.nStates), "奖励函数无效：它具有维度 " + repr(R.shape) + ", 但它应该是 (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "折扣系数无效：它应该在[0,1]中"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''值迭代法
        V <-- max_a R^a + gamma T^a V

        输入:
        initialV -- 初始的值函数: 大小为|S|的array
        nIterations -- 迭代次数的限制：标量 (默认值: infinity)
        tolerance -- ||V^n-V^n+1||_inf的阈值: 标量 (默认值: 0.01)

        Outputs: 
        V -- 值函数: 大小为|S|的array
        iterId -- 执行的迭代次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        iterId = 0
        epsilon = 0
        V = initialV
        while True:
            Vprev = V
            V = np.max([self.R[a] + self.discount * self.T[a] @ Vprev for a in range(len(self.R))], axis=0)
            epsilon = np.max(np.abs(V-Vprev))
            iterId += 1
            if iterId >= nIterations or epsilon <= tolerance:
                break

        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''从值函数中提取具体策略的程序
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- 值函数: 大小为|S|的array

        Output:
        policy -- 策略: 大小为|S|的array'''
        policy = np.argmax([self.R[a] + self.discount * self.T[a] @ V.T for a in range(len(self.R))], axis=0)
        return policy 

    def evaluatePolicy(self,policy):
        '''通过求解线性方程组来评估政策
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- 策略: 大小为|S|的array

        Ouput:
        V -- 值函数: 大小为|S|的array'''
        policy = policy.astype(np.int16)
        new_policy = [[policy[i], i] for i in range(len(policy))]
        new_R = np.array([self.R[x, y] for x, y in new_policy])
        new_T = np.array([self.T[x, y] for x, y in new_policy])
        V = np.linalg.inv(np.eye(self.nStates) - self.discount * new_T) @ new_R
        
        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''策略迭代程序:  在策略评估(solve V^pi = R^pi + gamma T^pi V^pi) 和
        策略改进 (pi <-- argmax_a R^a + gamma T^a V^pi)之间多次迭代

        Inputs:
        initialPolicy -- 初始策略: 大小为|S|的array
        nIterations -- 迭代数量的限制: 标量 (默认值: inf)

        Outputs: 
        policy -- 策略: 大小为|S|的array
        V -- 值函数: 大小为|S|的array
        iterId --策略迭代执行的次数 : 标量'''


        V = np.zeros(self.nStates)
        policy = initialPolicy
        iterId = 0
        
        while True:
            old_policy = policy
            V = self.evaluatePolicy(policy)

            policy = self.extractPolicy(V)
            iterId += 1
            if np.all(policy == old_policy) or iterId >= nIterations:
                break

        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''部分的策略评估:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- 策略: 大小为|S|的array
        initialV -- 初始的值函数: 大小为|S|的array
        nIterations -- 迭代数量的限制: 标量 (默认值: infinity)
        tolerance --  ||V^n-V^n+1||_inf的阈值: 标量 (默认值: 0.01)

        Outputs: 
        V -- 值函数: 大小为|S|的array
        iterId -- 迭代执行的次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        V = initialV
        iterId = 0
        epsilon = 0
        while True:
            Vprev = V
            new_policy = [[policy[i], i] for i in range(len(policy))]
            new_R = np.array([self.R[x, y] for x, y in new_policy])
            new_T = np.array([self.T[x, y] for x, y in new_policy])

            V = new_R + self.discount * (new_T @ V.T).reshape(-1)
            epsilon = np.max(np.abs(V-Vprev))
            iterId += 1
            if iterId >= nIterations or epsilon <= tolerance:
                break

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''修改的策略迭代程序: 在部分策略评估 (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        和策略改进(pi <-- argmax_a R^a + gamma T^a V^pi)之间多次迭代

        Inputs:
        initialPolicy -- 初始策略: 大小为|S|的array
        initialV -- 初始的值函数: 大小为|S|的array
        nEvalIterations -- 每次部分策略评估时迭代次数的限制: 标量 (默认值: 5)
        nIterations -- 修改的策略迭代中迭代次数的限制: 标量 (默认值: inf)
        tolerance -- ||V^n-V^n+1||_inf的阈值: scalar (默认值: 0.01)

        Outputs: 
        policy -- 策略: 大小为|S|的array
        V --值函数: 大小为|S|的array
        iterId -- 修改后策略迭代执行的迭代次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        _iterId = 0
        epsilon = 0
        policy = initialPolicy
        _V = initialV

        while True:
            old_value = _V
            [_V, iterId, _] = self.evaluatePolicyPartially(policy,_V,nIterations=nEvalIterations,tolerance=tolerance)
            policy = self.extractPolicy(_V)
            _iterId += iterId
            epsilon = np.max(np.abs(_V-old_value))
            if epsilon <= tolerance:
                break
            

        return [policy,_V,_iterId,epsilon]
        