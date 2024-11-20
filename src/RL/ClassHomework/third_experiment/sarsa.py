import numpy as np
import pygame
from time import time, sleep
from random import randint
import random

# parameters of maze
frame_interval = 0.01 # control run speed
n = 4 # size of maze
penalities = 4 # no. of red blocks

# parameters of q learning
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
epsilon = 1.0
epsilon_delta = 0.0001

# init maze
action = -1
next_state_action = -1
current_pos = (0,0)
reward = np.zeros((n,n))
terminals = []

background = (51,51,51)
screen = pygame.display.set_mode(((2*n+1)*100,n*100))
colors = [[background for i in range(n)] for j in range(n)]

while penalities > 0:
    i = randint(0,n-1)
    j = randint(0,n-1)
    if reward[i][j]==0 and (i,j)!=(0,0) and (i,j)!=(n-1,n-1):
        reward[i][j] = -1
        penalities -= 1
        colors[i][j] = (255,0,0)
        terminals.append((i,j))
        
reward[n-1][n-1] = 1
colors[n-1][n-1] = (0,255,0)
terminals.append((n-1,n-1))



# init q learning
Q = {}
for i in range(n):
    for j in range(n):
        Q[(i,j)] = [0] * 4

def sarsa(current_state, next_state):
    global Q, alpha, gamma, reward, action, next_state_action
    Q[current_state][action] += alpha * (reward[next_state[0], next_state[1]] + gamma * (Q[next_state][next_state_action] - Q[current_state][action]))


def select_action(pos):
    if np.random.random() <= epsilon:
        action = randint(0, 3)
    else:
        action = np.argmax(Q[tuple(pos)])
    return action
      
def step():
    
    global current_pos, epsilon_delta, epsilon,action, next_state_action
    #此处需要实现动作选取，注意此处action和q-learning的区别
    if action == -1:
        action = select_action(current_pos)
    else:
        action = next_state_action

    new_pos = list(current_pos)
    if action==0 and new_pos[0]>0: #move up
        new_pos[0] -= 1
    elif action==1 and new_pos[0]<n-1: #move down
        new_pos[0] += 1
    elif action==2 and new_pos[1]>0: #move left
        new_pos[1] -= 1
    elif action==3 and new_pos[1]<n-1: #move right
        new_pos[1] += 1
    new_pos = tuple(new_pos)
    next_state_action = select_action(new_pos)
    if new_pos not in terminals:
        sarsa(current_pos, new_pos)
    else:
        sarsa(current_pos, new_pos)
        # 进行游戏状态初始化
        new_pos = (0, 0)
        action = -1
        next_state_action = -1

    if epsilon > 0.05:
        epsilon -= epsilon_delta
    current_pos = new_pos

def layout():
    for i in range(n):
        for j in range(n):
            pygame.draw.rect(screen, (255,255,255), (j*100,i*100,100,100), 0)
            pygame.draw.rect(screen, colors[i][j], (j*100+3,i*100+3,94,94), 0)
            pygame.draw.rect(screen, (255,255,255), ((j+n+1)*100,i*100,100,100), 0)
            vij = max(Q[(i,j)])
            qc = (0,int(255*vij),0) if vij>0 else (int(-255*vij),0,0)
            pygame.draw.rect(screen, qc, ((j+n+1)*100+3,i*100+3,94,94), 0)
    pygame.draw.circle(screen, (25,129,230), (current_pos[1]*100+50,current_pos[0]*100+50), 30, 0)
      
sleep_time = frame_interval
running = True
while running:
    step()
    sleep(sleep_time)
    screen.fill(background)
    layout()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            sleep_time = 0 if sleep_time>0 else frame_interval
    pygame.display.flip()
pygame.quit()
