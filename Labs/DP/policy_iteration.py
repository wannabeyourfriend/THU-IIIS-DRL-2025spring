from env_cliff_walking import *
class PolicyIteration:
    """
    策略迭代算法
    """
    def __init__(self, env, theta, gamma):
        self.env = env
        # Initialize the V^{\pi}(s) = 0
        self.v = [0] * self.env.ncol * self.env.nrow
        # Initialize the origin policy as random policy
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)]
        # The threshold for the stop condition
        self.theta = theta
        # The discount factor
        self.gamma = gamma
        # callback function
        self.callback = None
        self.id = 'policy_iteration'
        
    def set_callback(self, callback):
        self.callback = callback
        
    def policy_evaluation(self, cnt):
        """
        策略评估算法
        """
        count = 1 # keep a record for the number of iterations
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                        # 本章环境比较特殊,奖励和下一个状态有关,所以需要和状态转移概率相乘
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if self.callback:
                self.callback(self.v, self.pi)
            if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代
            count += 1
        print(f"策略评估进行{count}轮后完成, 策略迭代到{cnt}轮")
        
    def policy_improvement(self):
        """
        策略改进算法
        """
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi
    
    def policy_iteration(self):
        """
        策略迭代算法
        """
        cnt = 0
        while True:
            cnt += 1
            self.policy_evaluation(cnt)
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if new_pi == old_pi: break