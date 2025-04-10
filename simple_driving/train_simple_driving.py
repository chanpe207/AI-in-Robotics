import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from simple_driving.simple_driving_env import SimpleDrivingEnv  # Update this path as needed

def main():
    # Create the environment
    env = SimpleDrivingEnv(isDiscrete=True, renders=False)

    # (Optional) Check if the environment is valid
    check_env(env, warn=True)

    # Initialize DQN model with a Multi-layer Perceptron policy
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=0.1,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.1,
        exploration_final_eps=0.05
    )

    # Train the model
    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("simple_driving_dqn_model")

    # Clean up
    env.close()
    print(f"Training complete. Model saved to 'simple_driving_dqn_model'.")

if __name__ == "__main__":
    main()
