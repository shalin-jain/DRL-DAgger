import gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import deque
from tqdm import tqdm

class Policy(nn.Module):
    """Policy Network"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

def expert_policy(observation):
    """
    Generate expert action given current observation which attempts to first slow descent, then correct angle.

    Args:
        observation: (torch.Tensor) [x, y, x', y', theta, omega, left contact, right contact]
    """
    x, y, x_vel, y_vel, theta, omega, l, r = observation
    
    # landed
    if l and r:
        return [1, 0, 0, 0]

    action = [1, 0, 0, 0]

    # slow fall
    if y_vel < -0.3:
       return [0, 0, 1, 0]
    
    # stop horizontal drive
    if abs(x) > 0.1:
        if x > 0:
            action = [0, 1, 0, 0]  # push right
        else:
            action = [0, 0, 0, 1]  # push left
    
    # maintain low tilt angle
    if abs(theta) > 0.1:
        if theta < 0:
            action = [0, 1, 0, 0]  
        else:
            action = [0, 0, 0, 1] 

    return action

def dagger(env, policy, expert_policy, iterations=1000, rollout_length=100, batch_size=64, lr=1e-3):
    """
    Run training with dataset aggregation.

    Args:
        env: (gym environment)
        policy: (nn.Module) policy network
        expert_policy: (fn) maps observations to optimal actions
        iterations: (int) number of training iterations
        rollout_length: (int) max number of environment steps
        batch_size: (int) batch size for updates
        lr: (float) learning rate
    
    Returns:
        (nn.Module) trained policy
    """

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = deque(maxlen=500000)  # replay buffer
    losses = []     # store losses for plotting
    returns = []    # store returns for plotting
    
    for i in tqdm(range(iterations)):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        episode_return = 0 # track episode return
        
        for _ in range(rollout_length):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().argmax()  # choose policy action 
            expert_action = expert_policy(obs)  # label observation with expert action
            dataset.append((obs, expert_action))
            obs, reward, done, _, _ = env.step(action)  # step environment
            episode_return += reward
            if done:
                obs = env.reset()
                obs = obs[0] if isinstance(obs, tuple) else obs
        
        returns.append(episode_return)
        
        # train on aggregated dataset
        if len(dataset) > batch_size:
            batch = np.random.choice(len(dataset), batch_size, replace=False)   # sample batch from dataset
            batch_states = torch.tensor([dataset[i][0] for i in batch], dtype=torch.float32)    # get states
            batch_actions = torch.tensor([dataset[i][1] for i in batch], dtype=torch.float32)      # get actions
            
            optimizer.zero_grad()
            pred_actions = policy(batch_states)
            loss = loss_fn(pred_actions, batch_actions)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            tqdm.write(f"Iter {i+1}: Loss = {loss.item():.4f}, Reward = {episode_return:.2f}")
    
    # Plot training loss
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Iters')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.savefig("training_loss.png")
    
    # Plot returns
    plt.figure()
    plt.plot(returns)
    plt.xlabel('Iters')
    plt.ylabel('Returns')
    plt.title('Train Episode Returns')
    plt.savefig("episode_returns.png")
    
    print("training done.")
    return policy

# Visualize trained policy and save as GIF
def visualize_policy(env, policy, episodes=2, gif_filename="trained_policy.gif"):
    frames = []
    for episode in range(episodes):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        done = False
        total_reward = 0
        
        while not done:
            frame = env.render()
            frames.append(frame)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().argmax()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        
        print(f"episode {episode+1}: reward = {total_reward}")
    
    env.close()
    imageio.mimsave(gif_filename, frames, fps=30)
    print(f"saved gif as {gif_filename}")

if __name__ == '__main__':
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)

    # Initialize environment and policy
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    policy = Policy(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)

    # Train the policy using DAgger
    trained_policy = dagger(env, policy, expert_policy)

    # Visualize trained policy and save as GIF
    visualize_policy(env, trained_policy)
