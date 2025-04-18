Files:

train_simple_driving.py
    Training script, runs for 1000 episodes and creates graph of the average Reward, Loss and Q-values for the last 100 episodes, as well as the showing the epsilon every episode. Weights and graphs are saved under simple_driving/agents/weights. The final weights are renamed manually as "load_model.pth" for use in the test script.

test_simple_driving.py
    Test script to render the simulation of the car driving to a random goal using the DQN agent and the trained weights from "simple_driving/agents/weights/load_model.pth". Runs 5 tests.

simple_driving/envs/simple_driving_env.py
    Modified script from https://github.com/fredsukkar/Gym-Medium-Post/blob/main/simple_driving/envs/simple_driving_env.py . Step function modified to incorporate discrete action spaces and rewards.

simple_driving/agents/dqn_agent.py
    Deep Q-Network agent to incorporate learning using a neural-network and epsilon-greedy function that incorporates prior knowledge.

simple_driving/agents/utils.py
    Utility functions for saving and loading weights.

