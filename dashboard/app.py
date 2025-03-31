import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.warehouse_env import WarehouseEnv


import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from env.warehouse_env import WarehouseEnv
from models.dqn import DQN, Agent

# Initialize environment and agent
env = WarehouseEnv()
agent = Agent(env)

# Streamlit UI
st.title("📦 Warehouse Inventory Management with DQN")

# Sidebar Parameters
st.sidebar.header("⚙️ Model Settings")
episodes = st.sidebar.slider("Number of Episodes", 100, 1000, 500, 50)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
epsilon_decay = st.sidebar.slider("Epsilon Decay Rate", 0.90, 0.999, 0.995, 0.001)

# Apply new settings
agent.epsilon_decay = epsilon_decay
agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=learning_rate)

# Training Section
st.subheader("📊 Training Progress")
train_button = st.button("Train Agent")

if train_button:
    rewards_history = []
    st.write("**Training Started...**")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.append((state, action, reward, next_state))
            state = next_state
            total_reward += reward

            if len(agent.memory) > agent.batch_size:
                agent.update_model()

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        agent.target_model.load_state_dict(agent.model.state_dict())

        rewards_history.append(total_reward)
        if (episode + 1) % 50 == 0:
            st.write(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")

    st.write("**Training Complete!**")

    # Plot rewards
    fig, ax = plt.subplots()
    ax.plot(range(1, episodes + 1), rewards_history, label="Total Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Performance Over Time")
    ax.legend()
    st.pyplot(fig)

# Save & Load Model Section
st.sidebar.subheader("💾 Model Management")
save_button = st.sidebar.button("Save Model")
load_button = st.sidebar.button("Load Model")

if save_button:
    torch.save(agent.model.state_dict(), "dqn_model.pth")
    st.sidebar.success("Model Saved Successfully!")

if load_button:
    agent.model.load_state_dict(torch.load("dqn_model.pth"))
    agent.model.eval()
    st.sidebar.success("Model Loaded Successfully!")

# Simulation Section
st.subheader("🕹️ Run Simulation")
run_simulation = st.button("Run Trained Agent")

if run_simulation:
    state = env.reset()
    done = False
    inventory_history = [state[0]]

    while not done:
        action = agent.select_action(state)
        next_state, _, done, _ = env.step(action)
        inventory_history.append(next_state[0])
        state = next_state
        time.sleep(0.1)

    # Plot Inventory Levels
    fig, ax = plt.subplots()
    ax.plot(inventory_history, label="Inventory Level")
    ax.axhline(y=env.max_inventory / 2, color="r", linestyle="--", label="Target Level")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Inventory")
    ax.set_title("Warehouse Inventory Over Time")
    ax.legend()
    st.pyplot(fig)
