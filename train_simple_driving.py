import gymnasium as gym
from simple_driving.envs.simple_driving_env import SimpleDrivingEnv
from simple_driving.agents.dqn_agent import DQNAgent
from simple_driving.agents.utils import save_model, load_model
   

def train():
    # Initialize environment and agent
    env = SimpleDrivingEnv(isDiscrete=True, renders=True)
    agent = DQNAgent(input_dim=2, output_dim=9)  # 2 for goal position, 9 for actions
    load_model(agent.q_network, "simple_driving/agents/weights/best_model_load.pth")
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

            # # Save every N episodes
            # if episode % 100 == 0:
            #     save_model(agent.q_network, f"simple_driving.agents.weights.weights_ep{episode}.pth")

            # # Save if it’s the best performance so far
            # if total_reward > best_reward:
            #     best_reward = total_reward
            #     save_model(agent.q_network, "simple_driving.agents.weights.best_model.pth")

        print(f"Done")

        # Save if it’s the best performance so far
        if total_reward > best_reward or first_run == True:
            best_reward = total_reward
            first_run = False
            save_model(agent.q_network, "simple_driving/agents/weights/best_model.pth")
            print("Best Model Saved")
        
        print(f"Episode {episode+1}: Total Reward: {total_reward}: Best Reward: {best_reward}")

        # Update the target network periodically
        if episode % 10 == 0:
            agent.update_target_network()

        # Print the total reward at the end of each episode
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}: Total Reward: {total_reward}")

if __name__ == "__main__":
    train()
