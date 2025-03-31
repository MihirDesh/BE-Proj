import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from env.warehouse_env import WarehouseEnv

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(input_dim=3, output_dim=10).to(self.device)
        self.target_model = DQN(input_dim=3, output_dim=10).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.batch_size = 32
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 10)  # Random action
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state_tensor)).item()

    def train(self, episodes=500):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                self.memory.append((state, action, reward, next_state))
                state = next_state

                if len(self.memory) > self.batch_size:
                    self.update_model()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.target_model.load_state_dict(self.model.state_dict())  # Update target network periodically

            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {self.epsilon:.4f}")

    def update_model(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)

        # Compute current Q-values
        q_values = self.model(states).gather(1, actions).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values

        loss = self.criterion(q_values, target_q_values)
        self.optimizer
