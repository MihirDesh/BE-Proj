import torch
import numpy as np
import random

# Dummy Environment (Replace with your real environment)
class ManufacturingEnv:
    def __init__(self):
        self.state_size = 5
        self.action_size = 3
        self.state = np.random.rand(self.state_size)

    def reset(self):
        self.state = np.random.rand(self.state_size)
        return self.state

    def step(self, action):
        next_state = np.random.rand(self.state_size)
        reward = np.random.rand() * 10  # Random reward for testing
        done = np.random.rand() > 0.9  # Ends episode randomly
        return next_state, reward, done, {}

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9  # Faster decay for quick testing
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 8  # Smaller batch for quick tests
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Smaller model for faster execution
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, action_size)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state, dtype=np.float32),
                            np.array(action, dtype=np.int64),
                            np.array(reward, dtype=np.float32),
                            np.array(next_state, dtype=np.float32),
                            np.array(done, dtype=np.bool_)))

        if len(self.memory) > 1000:
            self.memory.pop(0)  # Prevent memory overflow

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

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
env = ManufacturingEnv()
agent = DQNAgent(state_size=5, action_size=3)

# Run only 5 episodes for testing
num_episodes = 5
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(20):  # Shorter episodes
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        total_reward += reward
        state = next_state
        if done:
            break

    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    # Faster epsilon decay for quick results
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

print("\n✅ Testing Completed!")
