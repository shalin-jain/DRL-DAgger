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
DROPOUT_RATE = 0.0  # probability with which observations are zero-ed out
DROPOUT_MASK = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # valid observation indices to apply dropout on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnvWrapper():
    def __init__(self, env, dropout_rate=0.0, dropout_mask=np.array([1, 1, 1, 1, 1, 1, 1, 1])):
        self.env = env
        self.observation_space = env.observation_space

        assert dropout_mask.shape == self.observation_space.shape, f"expected dropout mask shape {self.observation_space.shape}, got {dropout_mask.shape}"
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask
        self.action_space = env.action_space

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()


    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # apply observation dropout
        p = np.random.uniform(size=obs.shape[0])
        p = np.where(self.dropout_mask == 1, p, 0)
        obs = np.where(p < self.dropout_rate, 0, obs)

        return obs, reward, done, truncated, info


class LSTMPolicy(nn.Module):
    """LSTM followed by FC"""

    def __init__(self, input_size, action_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, action_size)


    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

    def init_hidden(self, batch_size=1):
        # Initialize both hidden state and cell state
        return (torch.zeros(1, batch_size, HIDDEN_SIZE).to(DEVICE),
                torch.zeros(1, batch_size, HIDDEN_SIZE).to(DEVICE))


def train_ppo_expert(env_id="LunarLander-v2", total_timesteps=200_000):
    """Train the PPO expert or load if already exists"""

    save_path = "ppo_expert.zip"


    if os.path.exists(save_path):
        return PPO.load(save_path)

    env = DummyVecEnv([lambda: gym.make(env_id)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    return model



def train_dagger(env, expert, epochs=TOTAL_EPOCHS):
    """Train DAgger to imitate the expert"""

    policy = LSTMPolicy(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    dataset = []
    loss_history = []
    dataset = []
    loss_history = []
    return_history = []

    obs, _ = env.reset()
    done = False
    hidden = policy.init_hidden()

    # seed initial dataset with high quality expert actions
    for _ in range(1000):
        action, _ = expert.predict(obs)
        dataset.append((obs, action, (hidden[0].clone(), hidden[1].clone())))
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
            hidden = policy.init_hidden()

    # main DAgger training loop
    for epoch in range(epochs):
        total_loss = 0.0  # for logging and plotting
        total_loss = 0.0  # for logging and plotting

        # run multiple updates steps per epoch
        # run multiple updates steps per epoch
        for _ in range(UPDATE_STEPS):
            batch_indices = np.random.choice(len(dataset), BATCH_SIZE, replace=False)
            obs_batch = torch.tensor(np.array([dataset[i][0] for i in batch_indices]), dtype=torch.float32).to(DEVICE)
            act_batch = torch.tensor(np.array([dataset[i][1] for i in batch_indices]), dtype=torch.long).to(DEVICE)

            # Prepare LSTM hidden states (h, c)
            h_batch = torch.stack([dataset[i][2][0] for i in batch_indices]).view(1, BATCH_SIZE, HIDDEN_SIZE)
            c_batch = torch.stack([dataset[i][2][1] for i in batch_indices]).view(1, BATCH_SIZE, HIDDEN_SIZE)
            hidden_batch = (h_batch, c_batch)

            obs_batch = obs_batch.unsqueeze(0)
            pred_actions, _ = policy(obs_batch, hidden_batch)
            loss = nn.CrossEntropyLoss()(pred_actions.squeeze(0), act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss / UPDATE_STEPS)

        # collect transitions for DAgger dataset
        obs, _ = env.reset()
        done = False
        hidden = policy.init_hidden()
        timesteps = 0
        while not done and timesteps < EPISODE_LENGTH:
            prev_hidden = (hidden[0].clone(), hidden[1].clone())
            obs_tensor = torch.FloatTensor(obs).to(DEVICE)
            with torch.no_grad():
                policy_act, hidden = policy(obs_tensor.unsqueeze(0).unsqueeze(0), hidden)
                action = policy_act.squeeze(0).argmax().item()

            expert_act, _ = expert.predict(obs)
            dataset.append((obs, expert_act, prev_hidden))
            obs, _, done, _, _ = env.step(action)
            timesteps += 1

        ret = np.mean(evaluate_policy(env, policy, episodes=5, silent=True))
        return_history.append(ret)
        print(f"Epoch {epoch + 1} | Loss: {loss_history[-1]:.4f} | Avg Return: {ret:.2f}")

    # plot loss and returns
    plot_training_stats(loss_history, return_history)

    return policy


def evaluate_policy(env, policy_fn, episodes=10, silent=False, visualize=False, n_episodes=3,
                    gif_filename='lunarlander.gif'):
    """Evaluate the policy, optionally render to gif"""

    returns = []
    frames = []
    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        if isinstance(policy_fn, LSTMPolicy):
            hidden_state = policy_fn.init_hidden()
        else:
            hidden_state = None

        timesteps = 0
        while not done and timesteps < EPISODE_LENGTH:
            if isinstance(policy_fn, PPO):
                action, _ = policy_fn.predict(obs)
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action_logits, hidden_state = policy_fn(obs_tensor.unsqueeze(0), hidden_state)
                    action = action_logits.squeeze(0).argmax().item()

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
        if visualize and i == n_episodes - 1 and frames:
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

    plt.suptitle("LSTM DAgger Training Progress")
    plt.tight_layout()
    plt.savefig('lstm_dagger_training_progress.png')
    plt.close()


def plot_returns(expert_returns, dagger_returns):
    """Compare DAgger policy performance to expert"""

    plt.figure(figsize=(10, 6))
    plt.plot(expert_returns, label="PPO Expert", color="blue", linestyle="--")
    plt.plot(dagger_returns, label="LSTM DAgger policy", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Returns Comparison: PPO Expert vs LSTM DAgger policy")
    plt.legend()
    plt.savefig('lstm_dagger_final_comparison.png')


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    env = gym.make("LunarLander-v2")
    wrapped_env = EnvWrapper(env, DROPOUT_RATE, DROPOUT_MASK)

    print("Training PPO Expert...")
    expert_model = train_ppo_expert()

    print("Training LSTM DAgger policy...")
    dagger_model = train_dagger(wrapped_env, expert_model)

    print("Saving LSTM DAgger policy...")
    save_path = "dagger_lunarlander_lstm.pt"
    torch.save(dagger_model.state_dict(), save_path)

    print("Evaluating PPO Expert...")
    expert_returns = evaluate_policy(
        gym.make("LunarLander-v2", render_mode="rgb_array"),
        expert_model,
        episodes=50,
        visualize=True,
        gif_filename='expert_policy.gif'
    )

    print("Evaluating LSTM DAgger policy...")
    dagger_returns = evaluate_policy(
        EnvWrapper(gym.make("LunarLander-v2", render_mode="rgb_array"), DROPOUT_RATE, DROPOUT_MASK),
        dagger_model,
        episodes=50,
        visualize=True,
        gif_filename='lstm_dagger_policy.gif'
    )

    print("Plotting final results...")
    plot_returns(expert_returns, dagger_returns)