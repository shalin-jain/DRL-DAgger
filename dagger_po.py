import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from popgym.envs.noisy_position_only_cartpole import NoisyPositionOnlyCartPole
from popgym.wrappers.markovian import Markovian
from popgym.core.observability import Observability
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import matplotlib.pyplot as plt

# Hyperparameters
HIDDEN_SIZE = 128
BATCH_SIZE = 64
UPDATE_STEPS = 1
TOTAL_EPOCHS = 250
EPISODE_LENGTH = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUPolicy(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.gru = nn.GRU(input_size, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, action_size)
        
    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)
        x = self.fc(x)
        return x, hidden

def train_ppo_expert(env_id="NoisyPositionOnlyCartPole-v0", total_timesteps=200_000):
    """Train PPO Expert with full observability"""

    save_path = "ppo_cartpole_expert.zip"
    
    if os.path.exists(save_path):
        return PPO.load(save_path)
    
    base_env = NoisyPositionOnlyCartPole()
    env = Markovian(base_env, Observability.FULL)
    env = DummyVecEnv([lambda: env])
    
    model = PPO("MlpPolicy", env, verbose=1)
    
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    return model

def train_dagger(env, expert, epochs=TOTAL_EPOCHS):
    """Train BC policy with DAgger, where DAgger is partially observable"""
    policy = GRUPolicy(env.observation_space['obs'].shape[0], env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    dataset = [] 
    loss_history = []   # for plotting
    return_history = [] # for plotting

    obs, _ = env.reset()
    done = False
    hidden = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)

    # seed initial dataset
    for _ in range(1000):
        action, _ = expert.predict(obs['state'])
        dataset.append((obs['obs'], action, hidden.clone()))
        obs, _, _, _, done = env.step(action)
        if done:
            obs, _ = env.reset()
    
    # dagger training loop
    for epoch in range(epochs):
        total_loss = 0.0    # for logging
        for _ in range(UPDATE_STEPS):
            batch_indices = np.random.choice(len(dataset), BATCH_SIZE, replace=False)
            obs_batch = torch.tensor(np.array([dataset[i][0] for i in batch_indices]), dtype=torch.float32).to(DEVICE)
            act_batch = torch.tensor(np.array([dataset[i][1] for i in batch_indices]), dtype=torch.long).to(DEVICE)
            hidden_batch = torch.stack([
                dataset[i][2] for i in batch_indices
            ]).permute(1, 0, 2)

            obs_batch = obs_batch.unsqueeze(0)
            pred_actions, _ = policy(obs_batch, hidden_batch)
            loss = nn.CrossEntropyLoss()(pred_actions.squeeze(0), act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss / UPDATE_STEPS)

        # DAgger data collection
        obs, _ = env.reset()
        done = False
        hidden = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
        timesteps = 0
        while not done and timesteps < EPISODE_LENGTH:
            obs_tensor = torch.FloatTensor(obs['obs']).to(DEVICE)
            with torch.no_grad():
                policy_act, hidden = policy(obs_tensor.unsqueeze(0), hidden)
                action = policy_act.squeeze(0).argmax().item()

            expert_act, _ = expert.predict(obs['state'])
            dataset.append((obs['obs'], expert_act, hidden.detach().clone()))
            obs, _, _, _, done = env.step(action)
            if done:
                obs, _ = env.reset()
                done = False
            timesteps += 1

        # Evaluate policy policy every epoch
        ret = np.mean(evaluate_policy(env, policy, episodes=5, silent=True))
        return_history.append(ret)
        print(f"Epoch {epoch+1} | Loss: {loss_history[-1]:.4f} | Avg Return: {ret:.2f}")

    plot_training_stats(loss_history, return_history)
    return policy

def evaluate_policy(env, policy_fn, episodes=10, silent=False, visualize = True):
    returns = []
    images = []
    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        hidden_state = None

        timesteps = 0
        while not done and timesteps < EPISODE_LENGTH:
            if isinstance(policy_fn, PPO):
                action, _ = policy_fn.predict(obs)
            else:
                obs_tensor = torch.FloatTensor(obs['obs']).unsqueeze(0).to(DEVICE)
                if hidden_state is None:
                    hidden_state = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
                with torch.no_grad():
                    action_logits, hidden_state = policy_fn(obs_tensor, hidden_state)
                    action = action_logits.squeeze(0).argmax().item()
            obs, reward, _, _, done = env.step(action)
            total_reward += reward
            timesteps +=1
        if not silent:
            print(f'episode reward: {total_reward}')
        returns.append(total_reward)
    return returns

def plot_training_stats(loss_history, return_history):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(loss_history, label="Training Loss", color="red")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(return_history, label="Avg Return (5 evals)", color="green")
    axs[1].set_ylabel("Return")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()

    plt.suptitle("DAgger Training Progress")
    plt.tight_layout()
    plt.savefig('dagger_training_progress.png')
    plt.close()

def plot_returns(expert_returns, dagger_returns):
    plt.figure(figsize=(10, 6))
    plt.plot(expert_returns, label="PPO Expert", color="blue", linestyle="--")
    plt.plot(dagger_returns, label="DAgger policy", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Returns Comparison: PPO Expert vs DAgger policy")
    plt.legend()
    plt.savefig('dagger_final_comparison.png')

if __name__ == "__main__":
    expert_env = Markovian(NoisyPositionOnlyCartPole(), Observability.FULL)
    policy_env = Markovian(NoisyPositionOnlyCartPole(), Observability.FULL_AND_PARTIAL)

    print("Training PPO Expert...")
    expert_model = train_ppo_expert()

    print("Training DAgger policy...")
    dagger_model = train_dagger(policy_env, expert_model)

    print("Evaluating PPO Expert...")
    expert_returns = evaluate_policy(expert_env, expert_model, episodes=50, visualize=True)

    print("Evaluating DAgger policy...")
    dagger_returns = evaluate_policy(policy_env, dagger_model, episodes=50, visualize=True)

    print("Plotting final results...")
    plot_returns(expert_returns, dagger_returns)
