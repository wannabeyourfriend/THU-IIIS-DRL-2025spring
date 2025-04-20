# RL

> Notes taker: Alex
>
> Contact：`wang-zx23@mails.tsinghua.edu.cn`
>
> Reference: `Mingsheng Long ML lecture9-10 Huazhe Xu DRL Lecture 1-2`

[TOC]

## 0 Intro

- 强化学习广泛应用于序列决策问题中，是机器学习的一个子领域，与监督学习、无监督学习都有交集。对于每一个**time step** $ t$，智能体接受一个观察$O_t$,得到一个标量监督信号**scalar cumulative reward** $R_t$,实施一个动作**action** $A_t$,环境**environment**随时间迭代。

- 强化学习中的**history**视为$O_t, R_t, A_t$的序列，**state**是**history**的函数，$S_t = f(H_t)$

- 奖励$R_t$是一个标量信号， 强化学习的基本假设是**Reward Hypothesis**, 即一个目标可以被最大化积累奖励所刻画。
- Agent会对世界模型的观察可以分为完全观察和部分观察，完全观察对应的是$O= S^a = S^e$这时对应的是MDP，马尔科夫决策过程；然而实际上在实际的任务中，智能体自身的状态是环境状态的一个子集，观察只能是部分的$S^a \neq S^b$，这时对应的是POMDP。

## 1 MDP

### 1.1 Basics

> RL的基本要素：Objective，State，Action，Reward

对于一个follow RL算法的agent，用策略函数$\pi$来刻画agent的行为，agent采取的策略可以分为$a = \pi(s)$决定性策略和随机性策略$\pi(a | s) = \mathbb{P}(A_t = a | S_t = s)$;使用价值函数来评估一个状态/策略的好坏：一般约定使用$V_{\pi}(s)$和$Q_{\pi}(s,a)$。RL中的**model**分为对环境的建模和对奖励函数的建模，通过转移概率和期望的形式反映未来状态和奖励与当前状态、动作以及历史的关系。一般约定$\mathcal{P}^{a}_{ss'} = \mathbb{P}[S_{t + 1} = s' | S_t =s, A_t = a] \quad \mathcal{R}^a_s = \mathbb{E}[R_{t + 1} | S_t = s, A_t = a] $。考虑到奖励信号常是延迟的，我们需要对预期奖励进行建模。

eg:在机器人控制领域，Objective是使得robot move forward，state是joints的angle和position，action是joints上的torques，reward是专门设计出用于激励机器人不跌倒、向前移动的奖励

### 1.2 Markov Process

- **Markov Process Definition:**$P(S_{t+ 1} | S_t) = P(S_{t+ 1} | S_1, \cdots, S_t)$,则称$S_t$是Markov的。一个MP由$<S, P>$的tuple构成。S是所有可能状态的集合，P是状态转移矩阵。

- **Markov Reward Process:**在以上的MP中，加入对于reward function$\mathcal{R}_s = \mathbb{E}[R_{t + 1}| S_t = s]$和用于计算累计奖励的折扣因子$\gamma \in [0, 1]$,在累计奖励计算中，可以使用总累计回报$G_t = \sum_{k = 0}^{\infty} \gamma^kR_{t + k + 1}$，折扣因子体现了考虑到未来的不确定性对于决策的影响，折扣因子越大，体现的是比较far-sighted的评估，而近似于0的则比较myopic.瞬时奖励$G_t$是r.v.
- **Value function:**从状态s出发，MRP的价值函数$v(s)$刻画的是期望回报。$V(s) = \mathbb{E}[G_t | S_t = s]$.
- **Bellman Equation:**可以将$G_t$分为当前的奖励和下一步的奖励,有$V(s) = \mathbb{E}[R_{t + 1} + \gamma V(S_{t + 1}) | S_t = s] = \mathcal{R}_s + \gamma \sum_{s' \in \mathcal{S} }\mathcal{P}_{ss'} V(s')$

proof：
$$
V(s) = \mathbb{E}[G_t | S_t = s] \\
= \mathbb{E}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \ldots | S_t = s] \\
 = \mathbb{E}[R_t + \gamma (R_{t+1} + \gamma R_{t+2} + \ldots) | S_t = s] \\
 = \mathbb{E}[R_t + \gamma G_{t+1} | S_t = s] \\
 = \mathbb{E}[R_t + \gamma V(S_{t+1}) | S_t = s] \\
$$
这里解释最后一个（笔者认为并不显然的）等式：

利用重期望法则（条件期望的重期望法则）：$E[X|Z=z] = E[E[X | Y] | Z =z]$同时我们还有$V(s) = \mathbb{E}[G_t | S_t = s]$,即$V(S_{t + 1}) = \mathbb{E}[G_{t + 1} |  S_{t + 1}]$那么有$E[V(S_{t + 1}) = \mathbb{E}[G_{t + 1} |  S_{t + 1}] | S_t = s] = \mathbb{E}[G_{t + 1} | S_t = s]$.

写成矩阵代数的形式：$\mathbf{v} = \mathbf{R} + \gamma \mathbf{P}\mathbf{v}$ 这个线性方程其实是有解的。

### 1.3 Markov Decision Process

- **Definition**：常用上标$\mathcal{P}_{ss'}^a  = \mathbb{P}[S_{t + 1} = s' | S_t = s, A_t =a] \quad \mathcal{R}^a_s = \mathbb{E}[R_{t+ 1}| S_t =s, A_t = a] $来表示动作$a$.而agent follow的policy可以定义为在状态s上动作a的distribution，即$\pi(a | s) = \mathbb{P}[A_t =a | S_t = s]$,Markov Decision Process中的agent的policy只取决于当前的状态，而与历史状态无关，同时其形式与时间也是无关的（平稳性，时间上的参数共享）：$A_t ~ \pi (·|S_t), \forall t > 0$

- **State Value function:**(状态价值函数)  $  V^{\pi}(s) =  \mathbb{E}^{\pi}[G_t | S_t = s]$

- **Action Value function:**(动作价值函数) $Q^{\pi}(s,a) = \mathbb{E}^{\pi}[G_t | S_t = s, A_t = a]$

- **Bellman Equation:**以下三组等式可以相互推导，都是贝尔曼期望方程。本质上其实是全概率公式的展开。

$$
V^{\pi}(s) = \mathbb{E}_{\pi}[R_{t+1} + \gamma V^{\pi}(S_{t+1}) | S_t = s] \\
Q^{\pi}(s, a) = \mathbb{E}_{\pi}[R_{t+1} + \gamma Q^{\pi}(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
$$

$$
V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a | s) \left( R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V^{\pi}(s') \right) \\
Q^{\pi}(s, a) = R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a' | s') Q^{\pi}(s', a')
$$

$$
V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a | s) Q^{\pi}(s, a) \\
Q^{\pi}(s, a) = R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V^{\pi}(s') \\
$$

- **Optimal Policy:** 一般用$V^{\pi}(s) \forall s$在policy上建立偏序关系，$\pi \geq \pi' \text{if } V^{\pi}(s) \geq V^{\pi'}(s) \forall s$，那么在MDP中，可以保证最优策略$\pi^{*}$的存在性，且在最优策略下，每一个状态s可以获得最大的状态价值函数，每一个状态动作对也可以获得最大的动作价值函数。如果知道了最优的动作价值函数$Q^{*}(s,a')$我们可以直接获得最优策略（由动态规划的最优子结构的存在性保证）$\pi^{*}=\mathbb{1}[\text{argmax}_{a \in \mathcal{A}}Q^{*}(s',a)]$但是找到$Q^{*}$是难的。

- **Bellman Optical Equation:**

  价值迭代使用的就是贝尔曼最优方程。

$$
V^*(s) = \max_{a \in \mathcal{A}} \{ r(s, a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s, a) V^*(s')\\
Q^*(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} p(s'|s, a) \max_{a' \in \mathcal{A}} Q^*(s', a')\\
$$

## 2 Model-based Learning

动态规划具有最优子结构。多步的贪心可以收敛到全局极值点。策略迭代 = 策略评估+策略提升，其最优性由**策略提升定理**来保证：贪心的改进,$\pi \to \pi ^{*}$,使用$\pi'(s) = \text{argmax}_{a \in \mathcal{A}}Q^{\pi }(s, a)$可以保证$V^{\pi}(s) \leq V^{\pi^*}(s) $，详细证明见后。

这里总结两个算法：

### Policy Iteration

- ![img](https://pic1.zhimg.com/v2-e7e49525a3eeb7a76355f5685f410d22_r.jpg)

### Value Iteration

- ![img](https://pic1.zhimg.com/v2-e7e49525a3eeb7a76355f5685f410d22_r.jpg)

## 3 Model-free learning

### MC-Value Iteration

![image-20250314084852638](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250314084852638.png)

通过蒙特卡洛，我们采样的实际上是真实的分布中的多条完整的trajectory（回报序列），用以估计当前的价值函数，蒙特卡洛算法是对真实的价值函数的无偏估计但是由于每次采样的是不同的时间步，回报的波动会影响每个状态值的更新，方差较大

### TD

时序差分算法相当于向未来看了$n$步，采用这$n$步作为更新价值的奖励信号，当$n \rightarrow \infty$ 实际上呢就得到了蒙特卡洛算法。$TD(0)$只往前看了一步，是有偏的估计，但方差比较小。

#### TD(0)

![image-20250314091416955](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250314091416955.png)

![image-20250314093401144](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250314093401144.png)

#### TD($\lambda$)

为结合MC和TD两种算法的优势，同时提升$TD(n)$算法的**efficiency**，我们引入资格迹($Eligibility \quad Trace$)的概念，用来度量某个状态对于奖励信号的提示强度(因为奖励不是及时的，我们需要衡量对**trajectory**中的奖励信号进行一个猜测)。可以理解为对状态发生频率以及最终的奖励发生前的状态两种**heurisitics**进行一个合理的tradeoff，我们让$E_0 (s) = 0 \quad E_t(s) = \lambda \gamma E_{t - 1}(s) + \mathbb{1}(S_t) = s$ . 基于此可以得到$TD(\lambda)$的算法：

![image-20250314095115269](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250314095115269.png)

### SARSA

我们希望对动作价值函数$Q(S,A)$给出一个好的估计，从而产生基于TD的控制算法。

#### SARSA

![image-20250314100059183](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250314100059183.png)

#### SARSA($\lambda$)

![image-20250314100253697](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250314100253697.png)

### Q-learning

我们希望采用探索与利用平衡的策略来收集数据，但使用贪婪的方式进行控制，于是我们可以得到如下的离线学习算法，即著名的Q-learning算法。

![image-20250314131435038](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250314131435038.png)

