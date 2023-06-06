import tensorflow as tf
import numpy as np
from Agame import Game
import cv2
import pygame
from pygame.locals import *
import sys
from collections import deque
import random

# 参数
ACTIONS = 3
GAMMA = 0.05
OBSERVE = 50000
EXPLORE = 200000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 1.0
REPLAY_MEMORY = 5000
BATCH = 64
FRAME_PER_ACTION = 1
tf = tf.compat.v1
tf.disable_eager_execution()


def weights_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2D(x, w, stride):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def create_network():
    w_conv1 = weights_variable([6, 6, 4, 32])  # 原来为卷积核8x8x4---但套用下面公式后反向推出卷积核大小应为6x6，所以改为6x6
    b_conv1 = bias_variable([32])

    w_conv2 = weights_variable([4, 4, 32, 64])  # 卷积核4x4x32
    b_conv2 = bias_variable([64])

    w_conv3 = weights_variable([3, 3, 64, 64])  # 卷积核3x3x64
    b_conv3 = bias_variable([64])

    w_fc1 = weights_variable([1600, 512])  # 1600为下面公式推导的最终结果
    b_fc1 = weights_variable([512])

    w_fc2 = weights_variable([512, ACTIONS])
    b_fc2 = weights_variable([ACTIONS])

    s = tf.placeholder('float', [None, 80, 80, 4])
    # 下面公式n为输入像素，p为padding：SAME为填充：值为1，VAlID为不填充：值为0，f为卷积核大小：8x8就是8，s为步长strides
    # 卷积公式(n+2p-f)/s+1
    h_conv1 = tf.nn.relu(conv2D(s, w_conv1, 4) + b_conv1)  # 代入公式(80+2x1-8)/4+1=19.5约20：20x20x32
    h_pool1 = max_pool_2x2(h_conv1)  # 池化：2x2步长2为原来一半：10x10x32

    h_conv2 = tf.nn.relu(conv2D(h_pool1, w_conv2, 2) + b_conv2)  # 再一次卷积(10+2x1-4)/2+1=5: 5x5x64

    h_conv3 = tf.nn.relu(conv2D(h_conv2, w_conv3, 1) + b_conv3)  # 第三次卷积，因为卷积核是3x3和步长是1((5+2x1-3)/1+1)，所以像素不变：5x5x64

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])  # 把5x5x64=1600展平
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)

    readout = tf.matmul(h_fc1, w_fc2) + b_fc2

    return s, readout


def train_network(s, readout, sess):
    a = tf.placeholder('float', [None, ACTIONS])
    y = tf.placeholder('float', [None])

    readout_action = tf.reduce_mean(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game_state = Game()

    D = deque()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    x_t, r_0 = game_state.step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)

    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=-1)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("model/best")

    if checkpoint and checkpoint.model_checkpoint_path:
        print(saver.restore(sess, checkpoint.model_checkpoint_path))
        print("成功加载训练模型")
        t = int(checkpoint.model_checkpoint_path.split('-')[1])+1
    else:
        print("重新训练")
        t = 0

    epsilon = INITIAL_EPSILON

    user = ''
    while True:
        for event in pygame.event.get():  # 需要事件循环，否则白屏
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        readout_t = readout.eval(feed_dict={s: [s_t]})[0]

        a_t = np.zeros(ACTIONS)
        action_index = 0

        if t % FRAME_PER_ACTION == 0:
            if np.random.random() <= epsilon:
                user = "随机"
                action_index = random.randrange(0, ACTIONS)
                a_t[action_index] = 1

            else:
                user = "AI"
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1

        if t > OBSERVE and epsilon > FINAL_EPSILON:
            epsilon = INITIAL_EPSILON - (INITIAL_EPSILON-FINAL_EPSILON) * t / EXPLORE
        elif t <= OBSERVE:
            epsilon = INITIAL_EPSILON
        else:
            epsilon = FINAL_EPSILON

        x_t1_colored, r_t = game_state.step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=-1)

        D.append((s_t, a_t, r_t, s_t1))

        if len(D) >= REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE and len(D) >= BATCH:
            '''
                这段代码是用于训练神经网络模型的一部分，使用了经验回放（Experience Replay）和Q-learning算法。
                首先，从经验回放记忆库 D 中随机抽取 BATCH 个样本作为 minibatch。
                将抽取的样本拆分为对应的状态 (s_j_batch)、动作 (a_batch)、奖励 (r_batch) 和下一个状态 (s_j1_batch)。
                创建一个空的 y_batch 列表，用于存储目标Q值。
                使用读出网络（readout）计算每个样本的下一个状态的Q值 (readout_j1_batch)。
                对于每个样本，计算目标Q值 y，使用公式 y = r + GAMMA * max(Q(s', a'))，其中 r 是奖励，GAMMA 是折扣因子，Q(s', a') 是下一个状态的最大Q值。
                运行训练步骤 (train_step)，将目标Q值 (y_batch)、动作 (a_batch) 和状态 (s_j_batch) 提供给模型，以更新模型的权重。
                这段代码的目的是根据当前状态和下一个状态的Q值来训练神经网络模型，使其能够预测最优的动作值。
            '''
            minibatch = random.sample(D, BATCH)

            s_j_batch = [d[0] for d in minibatch]

            a_batch = [d[1] for d in minibatch]

            r_batch = [d[2] for d in minibatch]

            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []

            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})

            for i in range(0, len(minibatch)):
                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch,
            })

        s_t = s_t1
        t += 1

        if t % 50000 == 0:
            saver.save(sess, 'model/train/ckpt', global_step=t)

        if t <= OBSERVE:
            state = '观测期'
        elif EXPLORE > t > OBSERVE:
            state = '探索期'
        else:
            state = '自主训练期'

        if action_index == 1:
            act = '左移'
        elif action_index == 2:
            act = '右移'
        else:
            act = '不动'
        print(f"状态：{state}, 次数：{t}, {user}动作：{act}, 得分：{r_t}"+", AI控制权重：{:.4f}".format(1-epsilon))


def play_game():
    sess = tf.InteractiveSession()
    s, readout = create_network()
    train_network(s, readout, sess)


def main():
    play_game()


if __name__ == "__main__":
    main()
