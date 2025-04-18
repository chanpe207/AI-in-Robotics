import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=1e-3, gamma=0.99, batch_size=64, epsilon=0.95, epsilon_decay=0.99, epsilon_min=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Create Q-network and target Q-network
        self.q_network = QNetwork(input_dim, output_dim)
        self.target_network = QNetwork(input_dim, output_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Initialize target network with the same weights

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = []  # Experience replay buffer

        self.prior_probs = np.array([
            0.10,  # Reverse + Left
            0.10,  # Reverse + Straight
            0.10,  # Reverse + Right
            0.05,  # No move + Left
            0.05,  # No move + Straight
            0.05,  # No move + Right
            0.20,  # Forward + Left
            0.20,  # Forward + Straight
            0.20,  # Forward + Right
            ]) # Makes epsilon-greedy more likely to choose forward/backward action
        self.prior_probs /= self.prior_probs.sum()  # Normalize, just in case

    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            # return np.random.choice(self.output_dim, p=self.prior_probs) # Biased random action based on prior probabilities
            return np.random.choice(self.output_dim)  # Random action
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()  # Choose action with highest Q-value

    def store_experience(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        # Sample a random batch from memory
        # batch = np.random.choice(self.memory, self.batch_size, replace=False)
        batch = random.sample(self.memory, self.batch_size)
        return zip(*batch)


    def train(self):
        # Only train if enough samples in memory
        if len(self.memory) < self.batch_size:
            # print(f"Batching") # Debugging
            return

        # Sample a batch from the experience replay buffer
        states, actions, rewards, next_states, dones = self.sample_batch()

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values for the current states
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values (using the target network)
        next_q_values = self.target_network(next_states)
        next_q_values_max = next_q_values.max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values_max * (1 - dones))

        # Compute loss (mean squared error)
        loss = nn.MSELoss()(q_values, target_q_values)

        # print(f"MSE Loss: {loss}") # Debugging

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # Update epsilon (for exploration-exploitation balance)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

        # At the end of agent.train()
        self.last_loss = loss.item()


    def update_target_network(self):
        # Periodically update the target network with the Q-network weights
        self.target_network.load_state_dict(self.q_network.state_dict())
