import torch
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.warehouse_env import WarehouseEnv

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the neural network model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_size)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state, dtype=np.float32),
                            np.array(action, dtype=np.int64),
                            np.array(reward, dtype=np.float32),
                            np.array(next_state, dtype=np.float32),
                            np.array(done, dtype=np.bool_)))

        if len(self.memory) > 10000:  # Maintain replay buffer size
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state), dim=1).item()  # Fixed indexing

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool, device=self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        max_next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Initialize Environment & Agent
env = WarehouseEnv()
agent = DQNAgent(state_size=3, action_size=10)  # Match DQN input/output dimensions

num_episodes = 100
episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(50):  # Max steps per episode
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        total_reward += reward
        state = next_state
        if done:
            break

    episode_rewards.append(total_reward)  # Store reward per episode
    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    # Decay exploration rate
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

# Save trained model
torch.save(agent.model.state_dict(), "../models/dqn_model.pth")
print("\n✅ Model training completed. Saved as dqn_model.pth!")

# Save rewards for later analysis
with open("episode_rewards.pkl", "wb") as f:
    pickle.dump(episode_rewards, f)
print("✅ Rewards saved to episode_rewards.pkl!")

# Plot Reward Trend
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Trend Over Episodes")
plt.grid(True)
plt.show()

# Moving Average for Smoother Trend
window_size = 10
moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, alpha=0.5, label="Raw Rewards")
plt.plot(range(window_size, len(episode_rewards) + 1), moving_avg, color='red', label="Moving Average")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Smoothed Reward Trend")
plt.legend()
plt.grid(True)
plt.show()




