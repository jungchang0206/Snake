import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from snake import SnakeGame
import os
import pickle

MODEL_PATH = 'snake_dqn.pth'
EPSILON_PATH = 'epsilon.pkl'

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.criterion(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train():
    game = SnakeGame()
    state_size = len(game._get_state())
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    batch_size = 64
    episodes = 35

    # Load model and epsilon if they exist
    if os.path.exists(MODEL_PATH):
        agent.model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded existing model weights.")
    if os.path.exists(EPSILON_PATH):
        with open(EPSILON_PATH, 'rb') as f:
            agent.epsilon = pickle.load(f)
        print("Loaded epsilon value.")

    for e in range(episodes):
        state = game.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get action
            action_idx = agent.act(state)
            action = [0, 0, 0]
            action[action_idx] = 1
            
            # Perform action
            reward, done, score = game.play_step(action)
            next_state = game._get_state()
            
            # Remember and train
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
                # Save after each episode
                torch.save(agent.model.state_dict(), MODEL_PATH)
                with open(EPSILON_PATH, 'wb') as f:
                    pickle.dump(agent.epsilon, f)
                break

        agent.replay(batch_size)
            
    return agent

if __name__ == '__main__':
    agent = train()
    torch.save(agent.model.state_dict(), MODEL_PATH)
    with open(EPSILON_PATH, 'wb') as f:
        pickle.dump(agent.epsilon, f)