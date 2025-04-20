# Cliff-walking Lab
>This is a complemantion of the cliff walking lab in RL.
### Policy Iteration
#### Run
```bash
python main.py --method policy
```
#### Demo
![](https://github.com/wannabeyourfriend/RL-DP-Cliff-walking/blob/main/result/policy_iteration.gif)
#### Result
![](https://github.com/wannabeyourfriend/RL-DP-Cliff-walking/blob/main/result/result.png)
###
```bash
python main.py --method value
```
#### Demo
![](https://github.com/wannabeyourfriend/RL-DP-Cliff-walking/blob/main/result/value_iteration.gif)


### Analysis
![](https://github.com/wannabeyourfriend/RL-DP-Cliff-walking/blob/main/result/results-contrast.png)

智能体在开始的阶段会采取相对保守的策略，尽可能远离悬崖。随着智能体的探索，它会逐渐发现悬崖并采取更激进但总体收益更大的策略，沿着悬崖的边缘移动。

### Acknowledgement
```
https://github.com/boyu-ai/Hands-on-RL
```
