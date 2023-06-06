# -*- coding:utf-8 -*-
# @Time: 2023/5/29 0:06
# @Author: Noodles
# @File: Agame.py
# @Software: PyCharm


import pygame

# 游戏参数
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SCREEN_SIZE = [400, 400]
BAR_SIZE = [50, 5]
BALL_SIZE = [15, 15]
# 游戏移动判定
MOVE_STAY = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]


class Game(object):
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Simple Game')
        self.score = 0
        self.ball_pos_x = SCREEN_SIZE[0] // 2 - BALL_SIZE[0] / 2
        self.ball_pos_y = SCREEN_SIZE[1] // 2 - BALL_SIZE[1] / 2
        self.combo = 0
        self.ball_dir_x = -1  # -1 = left 1 = right
        self.ball_dir_y = -1  # -1 = up   1 = down
        self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])

        self.bar_pos_x = SCREEN_SIZE[0] // 2 - BAR_SIZE[0] // 2
        self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1] - BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])

    # action是MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
    # ai控制棒子左右移动；返回游戏界面像素数和对应的奖励。(像素->奖励->强化棒子往奖励高的方向移动)
    def step(self, action):
        action = list(action)

        if action == MOVE_LEFT:
            self.bar_pos_x = self.bar_pos_x - 2
        elif action == MOVE_RIGHT:
            self.bar_pos_x = self.bar_pos_x + 2
        else:
            pass

        if self.bar_pos_x < 0:
            self.bar_pos_x = 0
        if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
            self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]

        self.screen.fill(BLACK)
        self.bar_pos.left = self.bar_pos_x
        pygame.draw.rect(self.screen, WHITE, self.bar_pos)

        self.ball_pos.left += self.ball_dir_x * 2
        self.ball_pos.bottom += self.ball_dir_y * 3
        pygame.draw.rect(self.screen, WHITE, self.ball_pos)

        if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1] + 1):
            self.ball_dir_y = self.ball_dir_y * -1
        if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
            self.ball_dir_x = self.ball_dir_x * -1

        reward = 0
        if self.bar_pos.top <= self.ball_pos.bottom and (
                self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
            # reward = 1  # 击中奖励
            self.score += 1
        elif self.bar_pos.top <= self.ball_pos.bottom and (
                self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
            # reward = -1  # 没击中惩罚
            self.score += -1

        # 如果球的x坐标在条的范围内
        if self.bar_pos.left <= self.ball_pos.left and self.bar_pos.right >= self.ball_pos.right and \
                self.bar_pos.top > self.ball_pos.bottom:
            reward = 0.05
        elif ((self.ball_pos.left < self.bar_pos.left <= self.ball_pos.right) or
                (self.ball_pos.left <= self.bar_pos.right < self.ball_pos.right)):
            reward = 0.01
        elif ((self.bar_pos.left-self.ball_pos.right >= 1) or
                (self.ball_pos.left - self.bar_pos.right >= 1)):
            reward = -0.001

        # # 在游戏界面显示得分
        font = pygame.font.Font(None, 20)  # 初始化默认字体
        score_text = font.render("Score: " + str(self.score), True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # 获得游戏界面像素
        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        # 返回游戏界面像素和对应的奖励
        return screen_image, reward
