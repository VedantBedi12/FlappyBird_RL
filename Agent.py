import pygame
import sys
import random
import torch
import numpy as np
from collections import deque
import os
from model import Linear_QNet, QTrainer
from helper import plot
from FlappyBirdRL import FlappyBirdGame

MAX_MEMORY = 10000
BATCH_SIZE = 1000
LR = 0.01

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.1  # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 64, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        state = game.get_game_state()
        state = np.array(state)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(0.01, 200 - self.n_games)
        final_move = [0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # print("prediciton is", prediction)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            # print("final move is", final_move)
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = FlappyBirdGame()
    if os.path.isfile('./model/model.pth'):
        agent.model.load()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        g_reward, done, score = game.play_step(final_move)
        print("reward is", g_reward)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, g_reward, state_new, done)
        agent.remember(state_old, final_move, g_reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            total_score += score

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
