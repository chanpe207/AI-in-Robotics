import gymnasium as gym
from simple_driving.envs.simple_driving_env import SimpleDrivingEnv
from simple_driving.agents.dqn_agent import DQNAgent
from simple_driving.agents.utils import save_model, load_model
import torch
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# Tracking variables
reward_history = []
average_rewards = []
rolling_reward = deque(maxlen=100)

loss_history = []
average_loss = []
rolling_loss = deque(maxlen=100)

epsilon_history = []

q_value_history = []
average_q_values = []
rolling_q_value = deque(maxlen=100)

# Setup live plot
plt.ion()
fig, axs = plt.subplots(1, 4, figsize=(20, 4))

# Line placeholders
reward_line, = axs[0].plot([], [], label="Avg Reward (100 eps)")
loss_line, = axs[1].plot([], [], label="Avg Loss (100 eps)")
epsilon_line, = axs[2].plot([], [], label="Epsilon")
qval_line, = axs[3].plot([], [], label="Avg Q-value (100 eps)")

# Subplot formatting
axs[0].set_title("Average Reward")
axs[1].set_title("Loss")
axs[2].set_title("Epsilon")
axs[3].set_title("Q-value")

for ax in axs:
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.legend()

   

def train():
    # Initialize environment and agent
    env = SimpleDrivingEnv(isDiscrete=True, renders=False)
    agent = DQNAgent(input_dim=2, output_dim=9)  # 2 for goal position, 9 for actions

    # Before loading
    print("Before loading:", agent.q_network.fc1.weight[0][:5])

    # Load weights
    load_model(agent.q_network, "simple_driving/agents/weights/best_model_load.pth")

    # After loading
    print("After loading:", agent.q_network.fc1.weight[0][:5])

    first_run = True
    best_reward = 0

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store experience in memory
            agent.store_experience(state, action, reward, next_state, done)

            # Train the agent
            agent.train()

            # Update the state
            state = next_state
            total_reward += reward


            # # Save if it’s the best performance so far
            # if total_reward > best_reward:
            #     best_reward = total_reward
            #     save_model(agent.q_network, "simple_driving.agents.weights.best_model.pth")

        # print(f"Done")

        # Save if it’s the best performance so far
        if total_reward > best_reward or first_run == True:
            best_reward = total_reward
            first_run = False
            save_model(agent.q_network, "simple_driving/agents/weights/best_model.pth")
            print("Best Model Saved")

        
        # Save every N episodes
        if episode % 100 == 0:
            save_model(agent.q_network, f"simple_driving/agents/weights/weights_ep{episode}.pth")
            print(f"Episode {episode} Weights Saved")
        
        print(f"Episode {episode+1}: Total Reward: {total_reward}: Best Reward: {best_reward}")

        # Update the target network periodically
        if episode % 10 == 0:
            agent.update_target_network()

        # # Print the total reward at the end of each episode
        # if (episode + 1) % 10 == 0:
        #     print(f"Episode {episode+1}: Total Reward: {total_reward}")

        # === Reward ===
        reward_history.append(total_reward)
        rolling_reward.append(total_reward)
        average_rewards.append(np.mean(rolling_reward))

        # === Epsilon ===
        epsilon_history.append(agent.epsilon)

        # === Loss ===
        if hasattr(agent, 'last_loss') and agent.last_loss is not None:
            rolling_loss.append(agent.last_loss)
            average_loss.append(np.mean(rolling_loss))
        else:
            average_loss.append(None)

        # === Q-value ===
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = agent.q_network(state_tensor).numpy()[0]
            rolling_q_value.append(np.mean(q_vals))
            average_q_values.append(np.mean(rolling_q_value))

        # === Update plot data ===
        reward_line.set_xdata(np.arange(len(average_rewards)))
        reward_line.set_ydata(average_rewards)

        loss_line.set_xdata(np.arange(len(average_loss)))
        loss_line.set_ydata(average_loss)

        epsilon_line.set_xdata(np.arange(len(epsilon_history)))
        epsilon_line.set_ydata(epsilon_history)

        qval_line.set_xdata(np.arange(len(average_q_values)))
        qval_line.set_ydata(average_q_values)

        # Rescale and refresh
        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()


        # Decay the epsilon after every episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    
    plt.ioff()
    plt.tight_layout()
    plt.savefig("simple_driving/agents/weights/training_metrics.png")
    plt.show()





if __name__ == "__main__":
    train()
