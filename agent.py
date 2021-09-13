import torch
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np

from collections import namedtuple, deque
from game import SnakeGameAI
from model import QNetwork
from helper import plot

BUFFER_SIZE = 100000
BATCH_SIZE = 512
LR = 0.001
GAMMA = 0.99


class Agent:

    def __init__(self, state_size=11, action_size=3, seed=42):
        self.seed = random.seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR)

        self.replay_buffer = ReplayBuffer()

    def act(self, state, eps):
        actions = np.identity(3)
        if random.random() < eps:
            return actions[random.randint(0, 2)]

        state = torch.tensor(state, dtype=torch.float)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        return actions[torch.argmax(action_values).item()]

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        next_state = torch.unsqueeze(
            torch.tensor(next_state, dtype=torch.float), 0)
        action = torch.unsqueeze(torch.tensor(action, dtype=torch.float), 0)
        reward = torch.unsqueeze(torch.tensor(reward, dtype=torch.float), 0)
        done = torch.unsqueeze(torch.tensor(done, dtype=torch.uint8), 0)

        self.learn((state, action, reward, next_state, done))

    def step_batch(self):
        if len(self.replay_buffer) > BATCH_SIZE:
            experiences = self.replay_buffer.sample()
            return self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        Q_target_next = torch.max(self.qnetwork(
            next_states), dim=1)[0].unsqueeze(1)
        Q_target = rewards + (GAMMA * Q_target_next * (1 - dones))

        idx = torch.tensor([[x] for x in torch.argmax(actions, dim=1).numpy()])
        Q_expected = self.qnetwork(states).gather(1, idx)

        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_expected, Q_target)
        loss.backward()

        self.optimizer.step()
        return loss.item()


class ReplayBuffer():
    def __init__(self, seed=42):
        self.seed = random.seed(seed)

        self.batch_size = BATCH_SIZE
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names=[
            "state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(
            state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, BATCH_SIZE)

        states = torch.tensor(
            np.vstack([e.state for e in experiences]), dtype=torch.float)
        actions = torch.tensor(
            np.vstack([e.action for e in experiences]), dtype=torch.float)
        rewards = torch.tensor(
            np.vstack([e.reward for e in experiences]), dtype=torch.float)
        next_states = torch.tensor(
            np.vstack([e.next_state for e in experiences]), dtype=torch.float)
        dones = torch.tensor(
            np.vstack([e.done for e in experiences]), dtype=torch.uint8)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


def train(agent, game, n_episodes=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.98):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    max_score = 0
    eps = eps_start

    for i_episodes in range(1, n_episodes+1):
        state = game.reset()
        while True:
            action = agent.act(state, eps)
            reward, done, score = game.step(action)
            next_state = game.get_state()
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if score > max_score:
                max_score = score
                agent.qnetwork.save()
            if done:
                loss = agent.step_batch()
                break
        eps = max(eps_end, eps_decay*eps)

        print('Game', i_episodes, 'Score',
              score, 'Max Score:', max_score, 'Loss', loss)

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / i_episodes
        print(total_score, mean_score)
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    agent = Agent()
    game = SnakeGameAI()
    train(agent, game)
