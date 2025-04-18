import torch
import numpy as np
from simple_driving.envs.simple_driving_env import SimpleDrivingEnv
from simple_driving.agents.dqn_agent import DQNAgent
from simple_driving.agents.utils import save_model, load_model
import time

def test_model(model_path, episodes=10, render=True):
    # Initialize environment and agent
    env = SimpleDrivingEnv(isDiscrete=True, renders=render)
    agent = DQNAgent(input_dim=2, output_dim=9, epsilon=0.0)  # Set epsilon to 0 to act greedily

    # Load the trained model
    load_model(agent.q_network, model_path)
    agent.q_network.eval()

    print(f"Testing model from: {model_path}")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.select_action(state)  # Should be greedy (epsilon = 0)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            step += 1

            if render:
                time.sleep(0.01)

        print(f"Episode {episode+1}/{episodes} finished in {step} steps with reward {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    # Replace this path with your actual model path
    model_path = "simple_driving/agents/weights/load_model.pth"
    test_model(model_path=model_path, episodes=5, render=True)
