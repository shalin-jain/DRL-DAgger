import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import matplotlib.pyplot as plt
import imageio

# Hyperparameters
HIDDEN_SIZE = 128
BATCH_SIZE = 64
UPDATE_STEPS = 5
TOTAL_EPOCHS = 250
EPISODE_LENGTH = 500
DROPOUT_RATE = 0.7  # probability with which observations are zero-ed out
DROPOUT_MASK = np.array([1, 1, 1, 1, 1, 1, 1, 1])   # valid observation indices to apply dropout on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 0

SAVE_ROOT = None

TRAIN_NEW_GRU_POLICY = True
TRAIN_NEW_MLP_POLICY = True
TRAIN_NEW_EXPERT_POLICY = True



class EnvWrapper():
    def __init__(self, env, dropout_rate=0.0, dropout_mask=np.array([1, 1, 1, 1, 1, 1, 1, 1])):
        self.env = env
        self.observation_space = env.observation_space

        assert dropout_mask.shape == self.observation_space.shape, f"expected dropout mask shape {self.observation_space.shape}, got {dropout_mask.shape}"
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask
        self.action_space = env.action_space

    def reset(self):
        obs, info = self.env.reset()

        # apply observation dropout
        p = np.random.uniform(size=obs.shape[0])
        p = np.where(self.dropout_mask == 1, p, 0)
        obs_partial = np.where(p < self.dropout_rate, 0, obs)
        
        return (obs, obs_partial), info
    
    def render(self):
        return self.env.render()
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # apply observation dropout
        p = np.random.uniform(size=obs.shape[0])
        p = np.where(self.dropout_mask == 1, p, 0)
        obs_partial = np.where(p < self.dropout_rate, 0, obs)

        return (obs, obs_partial), reward, done, truncated, info

class GRUPolicy(nn.Module):
    """GRU followed by FC"""

    def __init__(self, input_size, action_size):
        super().__init__()
        self.gru = nn.GRU(input_size, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, action_size)
        
    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)
        x = self.fc(x)
        return x, hidden

class MLPPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_ppo_expert(env_id="LunarLander-v2", total_timesteps=200_000):
    """Train the PPO expert or load if already exists"""

    save_path = "ppo_expert.zip"
    
    if os.path.exists(save_path):
        print('Expert exists')
        return PPO.load(save_path)
    else:
        print('Expert not found. Start training expert.')
    
    env = DummyVecEnv([lambda: gym.make(env_id)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    return model

def train_dagger(env, expert, epochs=TOTAL_EPOCHS):
    """Train DAgger to imitate the expert"""

    policy = GRUPolicy(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    dataset = [] 
    loss_history = []   
    return_history = []

    (obs, obs_partial), _ = env.reset()
    done = False
    hidden = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)

    # seed intial dataset with high quality expert actions
    for _ in range(1000):
        action, _ = expert.predict(obs)
        dataset.append((obs_partial, action, hidden.clone()))
        (obs, obs_partial), _, done, _, _ = env.step(action)
        if done:
            (obs, obs_partial), _ = env.reset()
    
    # main DAgger training loop
    for epoch in range(epochs):
        total_loss = 0.0    # for logging and plotting

        # run multiple updates steps per epoch    
        for _ in range(UPDATE_STEPS):
            batch_indices = np.random.choice(len(dataset), BATCH_SIZE, replace=False)
            obs_batch = torch.tensor(np.array([dataset[i][0] for i in batch_indices]), dtype=torch.float32).to(DEVICE)
            act_batch = torch.tensor(np.array([dataset[i][1] for i in batch_indices]), dtype=torch.long).to(DEVICE)
            hidden_batch = torch.stack([dataset[i][2] for i in batch_indices]).permute(1, 0, 2)

            obs_batch = obs_batch.unsqueeze(0)
            pred_actions, _ = policy(obs_batch, hidden_batch)
            loss = nn.CrossEntropyLoss()(pred_actions.squeeze(0), act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss / UPDATE_STEPS)

        # collect transitions for DAgger dataset
        (obs, obs_partial), _ = env.reset()
        done = False
        hidden = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
        timesteps = 0
        while not done and timesteps < EPISODE_LENGTH:
            prev_hidden = hidden.clone()
            obs_tensor = torch.FloatTensor(obs_partial).to(DEVICE)
            with torch.no_grad():
                policy_act, hidden = policy(obs_tensor.unsqueeze(0), hidden)
                action = policy_act.squeeze(0).argmax().item()

            expert_act, _ = expert.predict(obs)
            dataset.append((obs_partial, expert_act, prev_hidden.detach().clone()))
            (obs, obs_partial), _, done, _, _ = env.step(action)
            timesteps += 1

        ret = np.mean(evaluate_policy(env, policy, episodes=5, silent=True))
        return_history.append(ret)
        print(f"Epoch {epoch+1} | Loss: {loss_history[-1]:.4f} | Avg Return: {ret:.2f}")

    # plot loss and returns
    plot_training_stats(loss_history, return_history)

    return policy


def train_dagger_baseline(env, expert, epochs=TOTAL_EPOCHS):
    """Train DAgger to imitate the expert"""

    policy = MLPPolicy(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    dataset = [] 
    loss_history = []   
    return_history = []

    (obs, obs_partial), _ = env.reset()
    done = False
    hidden = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)

    # seed intial dataset with high quality expert actions
    for _ in range(1000):
        action, _ = expert.predict(obs)
        dataset.append((obs_partial, action))
        (obs, obs_partial), _, done, _, _ = env.step(action)
        if done:
            (obs, obs_partial), _ = env.reset()
    
    # main DAgger training loop
    for epoch in range(epochs):
        total_loss = 0.0    # for logging and plotting

        # run multiple updates steps per epoch    
        for _ in range(UPDATE_STEPS):
            batch_indices = np.random.choice(len(dataset), BATCH_SIZE, replace=False)
            obs_batch = torch.tensor(np.array([dataset[i][0] for i in batch_indices]), dtype=torch.float32).to(DEVICE)
            act_batch = torch.tensor(np.array([dataset[i][1] for i in batch_indices]), dtype=torch.long).to(DEVICE)

            obs_batch = obs_batch
            pred_actions = policy(obs_batch)
            loss = nn.CrossEntropyLoss()(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss / UPDATE_STEPS)

        # collect transitions for DAgger dataset
        (obs, obs_partial), _ = env.reset()
        done = False
        timesteps = 0
        while not done and timesteps < EPISODE_LENGTH:
            obs_tensor = torch.FloatTensor(obs_partial).to(DEVICE)
            with torch.no_grad():
                policy_act = policy(obs_tensor)
                action = policy_act.squeeze(0).argmax().item()

            expert_act, _ = expert.predict(obs)
            dataset.append((obs_partial, expert_act))
            (obs, obs_partial), _, done, _, _ = env.step(action)
            timesteps += 1

        ret = np.mean(evaluate_policy(env, policy, episodes=5, silent=True))
        return_history.append(ret)
        print(f"Epoch {epoch+1} | Loss: {loss_history[-1]:.4f} | Avg Return: {ret:.2f}")

    # plot loss and returns
    plot_training_stats(loss_history, return_history)

    return policy



def evaluate_policy(env, policy_fn, episodes=10, silent=False, visualize=False, n_episodes=3, gif_filename='lunarlander.gif'):
    """Evalaute the policy, optionally render to gif"""

    returns = []
    frames = []
    for i in range(episodes):
        if isinstance(env, EnvWrapper):
            (obs, obs_partial), _ = env.reset()
        else:
            obs, _ = env.reset()
        done = False
        total_reward = 0
        hidden_state = None

        timesteps = 0
        while not done and timesteps < EPISODE_LENGTH:
            if isinstance(policy_fn, PPO):
                action, _ = policy_fn.predict(obs)
            elif isinstance(policy_fn, MLPPolicy):
                obs_tensor = torch.tensor(obs_partial, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                logits = policy_fn(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
            else:
                obs_tensor = torch.FloatTensor(obs_partial).unsqueeze(0).to(DEVICE)
                if hidden_state is None:
                    hidden_state = torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE)
                with torch.no_grad():
                    action_logits, hidden_state = policy_fn(obs_tensor.unsqueeze(0), hidden_state)
                    action = action_logits.squeeze(0).argmax().item()

            if isinstance(env, EnvWrapper):
                (obs, obs_partial), reward, done, _, _ = env.step(action)
            else:
                obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            timesteps += 1

            if visualize and i < n_episodes:
                frame = env.render()
                frames.append(frame)

        returns.append(total_reward)
        if not silent:
            print(f'Episode {i + 1} reward: {total_reward}')

        # Visualize the first n episodes
        if visualize and i == n_episodes and frames:
            imageio.mimsave(gif_filename, frames, fps=30)

    return returns


def plot_training_stats(loss_history, return_history):
    """Plot loss and return collected from training stats"""

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
    plt.savefig(os.path.join(SAVE_ROOT, 'dagger_training_progress.png'))
    plt.close()

def plot_returns(expert_returns, dagger_returns, dagger_baseline_returns):
    """Compare DAgger policy performance to expert"""

    plt.figure(figsize=(10, 6))
    plt.plot(expert_returns, label="PPO Expert", color="blue", linestyle="--")
    plt.plot(dagger_returns, label="DAgger policy", color="green")
    plt.plot(dagger_baseline_returns, label="DAgger policy (baseline)", color="yellow")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid()
    plt.title("Returns Comparison: PPO Expert vs DAgger policy")
    plt.legend()
    plt.savefig(os.path.join(SAVE_ROOT, 'dagger_final_comparison.png'))

if __name__ == "__main__":
    for exp in range(5):
        RANDOM_SEED = exp
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        env = gym.make("LunarLander-v2")
        wrapped_env = EnvWrapper(env, DROPOUT_RATE, DROPOUT_MASK)

        SAVE_ROOT = os.path.join('dropout_{x}'.format(x=DROPOUT_RATE), 'seed{x}'.format(x=RANDOM_SEED))
        if not os.path.exists(SAVE_ROOT):
            os.makedirs(SAVE_ROOT)
        
        print("Training PPO Expert...")
        expert_model = train_ppo_expert()

        if TRAIN_NEW_GRU_POLICY:

            print("Training DAgger policy...")
            dagger_model = train_dagger(wrapped_env, expert_model)

            print("Saving DAgger policy...")
            save_path = os.path.join(SAVE_ROOT, "dagger_lunarlander_gru.pt")
            torch.save(dagger_model.state_dict(), save_path)
        else:
            print("Loading DAgger policy")
            dagger_model = GRUPolicy(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
            dagger_model.load_state_dict(torch.load(os.path.join(SAVE_ROOT, 'dagger_lunarlander_gru.pt'), map_location=DEVICE))

        
        if TRAIN_NEW_MLP_POLICY:

            print("Training DAgger policy (MLP)...")
            dagger_model_baseline = train_dagger_baseline(wrapped_env, expert_model)

            print("Saving DAgger policy (MLP)...")
            save_path = os.path.join(SAVE_ROOT, "dagger_lunarlander_mlp.pt")
            torch.save(dagger_model_baseline.state_dict(), save_path)

        else:
            print("Loading DAgger policy (baseline)")
            dagger_model_baseline = MLPPolicy(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
            dagger_model_baseline.load_state_dict(torch.load(os.path.join(SAVE_ROOT, 'dagger_lunarlander_mlp.pt'), map_location=DEVICE))

        print("Evaluating PPO Expert...")
        expert_returns = evaluate_policy(
            gym.make("LunarLander-v2", render_mode="rgb_array"),
            expert_model,
            episodes=50,
            visualize=True,
            gif_filename=os.path.join(SAVE_ROOT, 'expert_policy.gif')
        )

        print("Evaluating DAgger policy...")
        dagger_returns = evaluate_policy(
            EnvWrapper(gym.make("LunarLander-v2", render_mode="rgb_array"), DROPOUT_RATE, DROPOUT_MASK),
            dagger_model,
            episodes=50,
            visualize=True,
            gif_filename=os.path.join(SAVE_ROOT, 'dagger_policy.gif')
        )

        print("Evaluating DAgger policy (baseline)...")
        dagger_baseline_returns = evaluate_policy(
            EnvWrapper(gym.make("LunarLander-v2", render_mode="rgb_array"), DROPOUT_RATE, DROPOUT_MASK),
            dagger_model_baseline,
            episodes=50,
            visualize=True,
            gif_filename=os.path.join(SAVE_ROOT, 'dagger_policy_baseline.gif')
        )

        print("Saving results")
        np.save(os.path.join(SAVE_ROOT, 'expert_returns.npy'), expert_returns)
        np.save(os.path.join(SAVE_ROOT, 'dagger_returns.npy'), dagger_returns)
        np.save(os.path.join(SAVE_ROOT, 'dagger_baseline_returns.npy'), dagger_baseline_returns)

        print("Plotting final results...")
        plot_returns(expert_returns, dagger_returns, dagger_baseline_returns)
