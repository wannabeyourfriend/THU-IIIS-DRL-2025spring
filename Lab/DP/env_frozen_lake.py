from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
import gym
import pygame
import numpy as np

# 初始化pygame
pygame.init()

# 创建环境
env = gym.make("FrozenLake-v1", render_mode="rgb_array")  # 创建环境
env.unwrapped
state, _ = env.reset()

# 获取渲染的图像
rgb_array = env.render()

# 设置pygame窗口
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("FrozenLake 环境")

# 将numpy数组转换为pygame表面
def array_to_surface(arr):
    arr = np.swapaxes(arr, 0, 1)
    return pygame.surfarray.make_surface(arr)

# 显示环境
surface = array_to_surface(rgb_array)
# 放大图像以便更好地查看
scaled_surface = pygame.transform.scale(surface, (screen_width, screen_height))
screen.blit(scaled_surface, (0, 0))
pygame.display.flip()

# 保持窗口打开直到用户关闭
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()

# 原有的代码继续执行
holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)


# 这个动作意义是Gym库针对冰湖环境事先规定好的
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
# print_agent(agent, action_meaning, [5, 7, 11, 12], [15])