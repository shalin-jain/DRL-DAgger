import gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from collections import deque
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def train_ppo_expert(env_id="LunarLander-v2", total_timesteps=200000):
    """Helper function to train a PPO expert using Stable Baselines"""
    save_path = "ppo_expert.zip"
    if os.path.exists(save_path):
        print(f"Loading pre-trained PPO model from {save_path}")
        model = PPO.load(save_path)
        return model
    env = gym.make(env_id)
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Saved PPO expert model to {save_path}")
    return model


def expert_policy_ppo(observation, ppo_model):
    """Call the trained PPO stable baselines expert"""
    obs = np.array(observation)
    if len(obs.shape) == 1:
        obs = obs.reshape(1, -1)
    action, _ = ppo_model.predict(obs, deterministic=True)
    return int(action[0])


def visualize_expert(ppo_model, env, episodes=3, gif_filename="ppo_expert.gif"):
    """Visualize the trained PPO stable baselines expert"""
    frames = []
    for episode in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total_reward = 0

        while not done:
            frame = env.render()
            frames.append(frame)
            action = expert_policy_ppo(obs, ppo_model)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        print(f"PPO Expert Episode {episode+1}: Total Reward = {total_reward}")

    env.close()
    imageio.mimsave(gif_filename, frames, fps=30)
    print(f"Saved PPO expert gif as {gif_filename}")


def evaluate_expert_rewards(ppo_model, env, n_episodes=50):
    """Evaluate the trained PPO stable baselines expert"""
    rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total_reward = 0
        while not done:
            action = expert_policy_ppo(obs, ppo_model)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print(f"Evaluation Episode {episode+1}: Reward = {total_reward}")

    plt.figure()
    plt.plot(rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Expert Evaluation Rewards")
    plt.savefig("ppo_expert_rewards.png")
    plt.close()
    print("Saved PPO expert reward curve as ppo_expert_rewards.png")
    return rewards


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
        return self.fc3(x)


def dagger(
    env,
    policy,
    expert_policy_fn,
    expert_model,
    iterations=500,
    rollout_length=100,
    batch_size=64,
    lr=1e-3,
    update_steps=5
):
    """
    Main Dagger Training Loop

    Args:
        env: (gym_env)
        policy: (nn.Module) policy network
        expert_policy_fn: (function) calls expert model
        expert_model: (stable_baselines) trained expert model
        iterations: (int) iterations of dataset aggregation
        rollout_length: (int) maximum rollout length
        batch_size: (int) batch size of updates
        lr: (float) learning rate
        update_steps: (int) number of updates performed per iteration
    
    Returns:
        (nn.Module) trained policy
    """

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataset = deque(maxlen=500000)  # replay buffer
    losses = []     # store losses for plotting
    returns = []    # store returns for plotting

    for i in tqdm(range(iterations)):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        episode_return = 0 # track episode return
        
        # rollout environment
        done = False
        env_steps = 0
        while not done and env_steps < rollout_length:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_tensor)
            pred_action = torch.argmax(logits, dim=1).item()

            # get expert action
            expert_action = expert_policy_fn(obs, expert_model)
            dataset.append((obs, expert_action))    # add labeled action to dataset

            # step environment
            obs, reward, done, _, _ = env.step(pred_action)
            episode_return += reward
            env_steps += 1

            if done:
                obs = env.reset()
                obs = obs[0] if isinstance(obs, tuple) else obs

        returns.append(episode_return)

        # update policy
        if len(dataset) > batch_size:
            # run multiple updates per iteration
            for j in range(update_steps):
                # sample batch of states and actions
                batch_indices = np.random.choice(len(dataset), batch_size, replace=False)
                batch_states = torch.tensor(
                    [dataset[i][0] for i in batch_indices], dtype=torch.float32
                )
                batch_actions = torch.tensor(
                    [dataset[i][1] for i in batch_indices], dtype=torch.long
                )

                # perform policy update
                optimizer.zero_grad()
                logits = policy(batch_states)
                loss = loss_fn(logits, batch_actions)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                tqdm.write(
                    f"Iter {i+1} Step {j+1}: Loss = {loss.item():.4f}, Return = {episode_return:.2f}"
                )

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Update Steps")
    plt.ylabel("Loss")
    plt.title("DAgger Training Loss")
    plt.savefig("dagger_training_loss.png")

    plt.figure()
    plt.plot(returns)
    plt.xlabel("Iterations")
    plt.ylabel("Return")
    plt.title("DAgger Episode Returns")
    plt.savefig("dagger_episode_returns.png")

    print("DAgger Training done.")
    return policy


def evaluate_policy(env, policy, episodes=50, vis_episodes=0, gif_filename='dagger_policy.gif'):
    frames = []
    rewards = []
    for episode in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total_reward = 0

        while not done:
            if episode < vis_episodes:
                frame = env.render()
                frames.append(frame)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_tensor)
            action = torch.argmax(logits, dim=1).item()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print(f"DAgger Policy Episode {episode+1}: Total Reward = {total_reward}")
    
    # plot evaluation
    plt.figure()
    plt.plot(rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DAgger Evaluation Rewards")
    plt.savefig("dagger_eval_episode_returns.png")
    plt.close()
    print("Saved DAgger evaluation returns dagger_eval_episode_returns.png")

    
    avg_episode_reward = np.mean(np.array(rewards))
    print("-----------------------------------------------------")
    print(f"DAgger Average Episode Reward = {avg_episode_reward}")
    print("-----------------------------------------------------")

    env.close()
    if frames:
        imageio.mimsave(gif_filename, frames, fps=30)
    print(f"Saved DAgger policy gif as {gif_filename}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    print("Training PPO expert...")
    ppo_expert = train_ppo_expert(env_id="LunarLander-v2", total_timesteps=200000)

    # print("Visualizing PPO expert...")
    # env_expert = gym.make("LunarLander-v2", render_mode="rgb_array")
    # visualize_expert(ppo_expert, env_expert, episodes=3, gif_filename="ppo_expert.gif")

    # print("Evaluating PPO expert rewards...")
    # env_eval = gym.make("LunarLander-v2", render_mode="rgb_array")
    # evaluate_expert_rewards(ppo_expert, env_eval, n_episodes=50)

    policy = Policy(
        input_dim=env.observation_space.shape[0], output_dim=env.action_space.n
    )

    print("Training DAgger policy using PPO expert labels...")
    trained_policy = dagger(
        env,
        policy,
        expert_policy_ppo,
        ppo_expert,
        iterations=500,
        rollout_length=500,
        batch_size=64,
        lr=1e-3,
        update_steps=5
    )

    print("Visualizing DAgger trained policy...")
    env_dagger = gym.make("LunarLander-v2", render_mode="rgb_array")
    evaluate_policy(
        env_dagger, trained_policy, episodes=50, vis_episodes=3
    )