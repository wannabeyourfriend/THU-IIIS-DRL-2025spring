import pygame
import sys
import numpy as np
from env_cliff_walking import CliffWalkingEnv
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
import argparse

class CliffWalkingViz:
    def __init__(self, env, algorithm):
        pygame.init()
        self.env = env
        self.algorithm = algorithm
        self.cell_size = 100
        self.width = env.ncol * self.cell_size
        self.height = env.nrow * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("悬崖漫步策略迭代可视化")
        
        # 设置颜色
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # 设置字体
        self.font = pygame.font.Font(None, 24)
        
        # 显示动作
        self.arrows = ['^', 'v', '<', '>']  # 上下左右四个动作的箭头表示
        self.arrow_font = pygame.font.Font(None, 36)  # 箭头使用更大的字号
        
    def draw_grid(self):
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        
    def draw_cliff(self):
        for j in range(1, self.env.ncol - 1):
            rect = pygame.Rect(j * self.cell_size, 
                             (self.env.nrow - 1) * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.RED, rect)
            
    def draw_start_goal(self):
        # 起点
        start_rect = pygame.Rect(0, (self.env.nrow - 1) * self.cell_size,
                               self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.GREEN, start_rect)
        
        # 终点
        goal_rect = pygame.Rect((self.env.ncol - 1) * self.cell_size,
                              (self.env.nrow - 1) * self.cell_size,
                              self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.GREEN, goal_rect)
        
        # 添加颜色映射相关的属性
        self.min_value = -100  # 最小值（悬崖奖励）
        self.max_value = 0     # 最大值（终点奖励）
        self.BLUE = (0, 0, 255)
        
    def get_color_for_value(self, value):
        # 将价值映射到颜色深浅
        normalized = (value - self.min_value) / (self.max_value - self.min_value + 0.01)
        normalized = max(0, min(1, normalized))  # 确保在 0-1 之间
        return (
            int(255 * (1 - normalized)),  # R
            int(255 * (1 - normalized)),  # G
            255                           # B
        )
        
    def draw_values(self, values):
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                state = i * self.env.ncol + j
                value = values[state]
                
                # 绘制背景色
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                 self.cell_size, self.cell_size)
                color = self.get_color_for_value(value)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
                
                # 绘制数值
                text = self.font.render(f"{value:.2f}", True, self.BLACK)
                text_rect = text.get_rect(center=(j * self.cell_size + self.cell_size // 2,
                                                i * self.cell_size + self.cell_size // 2))
                self.screen.blit(text, text_rect)
    
    def update_display(self, values, policy):
        self.min_value = min(values)
        self.max_value = max(values)
        
        self.screen.fill(self.WHITE)
        self.draw_values(values)
        self.draw_policy(policy)  # 添加策略箭头的绘制
        self.draw_cliff()
        self.draw_start_goal()
        pygame.display.flip()
        if self.algorithm.id == 'value_iteration':
            pygame.time.wait(200)  # 等待一段时间以显示价值迭代的效果
        
        elif self.algorithm.id == 'policy_iteration':
            pygame.time.wait(200)
            
        else:
            raise ValueError('algorithm must be policy_iteration or value_iteration')
        
    def run(self):
        # 设置回调函数
        self.algorithm.set_callback(self.update_display)
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # 开始迭代
                        if isinstance(self.algorithm, PolicyIteration):
                            self.algorithm.policy_iteration()
                        else:
                            self.algorithm.value_iteration()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # 初始显示
            self.update_display(self.algorithm.v, self.algorithm.pi)
            
        pygame.quit()

    def draw_policy(self, policy):
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                state = i * self.env.ncol + j
                # 跳过悬崖位置
                if i == self.env.nrow - 1 and j > 0 and j < self.env.ncol - 1:
                    continue
                    
                # 处理策略
                if policy[state] is None:
                    continue
                    
                max_prob = max(policy[state])
                best_actions = [a for a, p in enumerate(policy[state]) if p == max_prob]
                
                # 为每个最优动作绘制箭头
                for action in best_actions:
                    arrow = self.arrows[action]
                    text = self.arrow_font.render(arrow, True, self.BLACK)
                    
                    # 根据动作数量调整箭头位置
                    if len(best_actions) == 1:
                        # 单个动作时箭头居中
                        text_rect = text.get_rect(center=(
                            j * self.cell_size + self.cell_size // 2,
                            i * self.cell_size + self.cell_size * 0.75
                        ))
                    else:
                        # 多个动作时箭头分散
                        offset = 20 * (action - 1.5)  # 在水平方向上偏移
                        text_rect = text.get_rect(center=(
                            j * self.cell_size + self.cell_size // 2 + offset,
                            i * self.cell_size + self.cell_size * 0.75
                        ))
                    
                    self.screen.blit(text, text_rect)

def test_policy_iteration():
    env = CliffWalkingEnv()
    policy_iteration = PolicyIteration(env, theta=0.001, gamma=0.9)
    viz = CliffWalkingViz(env, policy_iteration)
    viz.run()
    
def test_value_iteration():
    env = CliffWalkingEnv()
    value_iteration = ValueIteration(env, theta=0.001, gamma=0.9)
    viz = CliffWalkingViz(env, value_iteration)
    viz.run()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Cliff Walking DP algorithm')
    parser.add_argument('--method', type=str, choices=['policy', 'value'], default='policy_iteration', help='choose policy_iteration or value_iteration')
    args = parser.parse_args()
    if args.method == 'policy':
        test_policy_iteration()
    elif args.method == 'value':
        test_value_iteration()
    else:
        raise ValueError('method must be policy or value')