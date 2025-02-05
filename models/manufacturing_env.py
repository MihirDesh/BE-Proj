import numpy as np
import gym
from gym import spaces

class ManufacturingEnv(gym.Env):
    def __init__(self):
        super(ManufacturingEnv, self).__init__()

        # Define the action space (e.g., adjusting production rates)
        self.action_space = spaces.Discrete(3)  # Example: 0 = decrease, 1 = maintain, 2 = increase
        
        # Define the observation space (inventory levels, demand, etc.)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(5,), dtype=np.float32)
        
        # Initialize state
        self.state = np.random.uniform(low=0, high=1000, size=(5,))
    
    def step(self, action):
        # Apply action logic (dummy example)
        self.state += np.random.uniform(-10, 10, size=(5,))
        reward = -np.sum(np.abs(self.state - 500))  # Reward based on how balanced the inventory is
        done = False  # Define termination condition
        
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.random.uniform(low=0, high=1000, size=(5,))
        return self.state
