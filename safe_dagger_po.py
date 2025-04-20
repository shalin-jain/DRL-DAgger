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
import time
from collections import deque
import copy


HIDDEN_SIZE = 128
BATCH_SIZE = 64
UPDATE_STEPS = 5
TOTAL_EPOCHS = 250
EPISODE_LENGTH = 500
DROPOUT_RATE = 0
DROPOUT_MASK = np.array([1, 1, 1, 1, 1, 1, 1, 1])
POLICY_LR = 1e-4
SAFETY_LR = 1e-4
SAFETY_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 0

SAVE_ROOT = None


TRAIN_NEW_EXPERT_POLICY = False
TRAIN_NEW_MLP_POLICY = True
TRAIN_NEW_GRU_POLICY = True
TRAIN_NEW_SAFEDAGGER_GRU_POLICY = True


class EnvWrapper:
    def __init__(
        self, env, dropout_rate=0.0, dropout_mask=np.array([1, 1, 1, 1, 1, 1, 1, 1])
    ):
        self.env = env
        self.observation_space = env.observation_space

        assert (
            dropout_mask.shape == self.observation_space.shape
        ), f"expected dropout mask shape {self.observation_space.shape}, got {dropout_mask.shape}"
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask
        self.action_space = env.action_space
        self._max_episode_steps = getattr(env, "_max_episode_steps", 1000)

    def reset(self, seed=None):

        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()

        p = np.random.uniform(size=obs.shape[0])
        p = np.where(self.dropout_mask == 1, p, 0)
        obs_partial = np.where(p < self.dropout_rate, 0, obs)

        return (obs, obs_partial), info

    def render(self):
        return self.env.render()

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        p = np.random.uniform(size=obs.shape[0])
        p = np.where(self.dropout_mask == 1, p, 0)
        obs_partial = np.where(p < self.dropout_rate, 0, obs)

        terminated = done or truncated

        return (
            (obs, obs_partial),
            reward,
            terminated,
            info,
        )

    def close(self):
        self.env.close()


class GRUPolicy(nn.Module):
    """GRU followed by FC"""

    def __init__(self, input_size, action_size, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x, hidden):

        x, hidden = self.gru(x, hidden)

        x = self.fc(x)

        return x, hidden

    def init_hidden(self, batch_size=1):

        return torch.zeros(1, batch_size, self.hidden_size, device=DEVICE)


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


class GRUSafetyClassifier(nn.Module):
    """GRU followed by FC for safety classification"""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):

        x, hidden = self.gru(x, hidden)

        x = self.fc(x)

        return x, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=DEVICE)


def train_ppo_expert(env_id="LunarLander-v2", total_timesteps=200_000):
    """Train the PPO expert or load if already exists"""
    save_path = "ppo_expert.zip"
    if os.path.exists(save_path) and not TRAIN_NEW_EXPERT_POLICY:
        print("Expert exists and TRAIN_NEW_EXPERT_POLICY=False. Loading expert.")
        return PPO.load(save_path)
    else:
        print("Training new expert or retraining requested.")

        env = DummyVecEnv([lambda: gym.make(env_id)])
        model = PPO("MlpPolicy", env, verbose=1, seed=RANDOM_SEED)
        model.learn(total_timesteps=total_timesteps)
        model.save(save_path)
        print(f"Expert saved to {save_path}")
        return model


def train_dagger_gru(env, expert, policy, epochs=TOTAL_EPOCHS):
    """Train DAgger GRU policy to imitate the expert"""
    print("\n--- Training Standard DAgger (GRU) ---")
    start_time = time.time()

    optimizer = optim.Adam(policy.parameters(), lr=POLICY_LR)

    dataset = deque(maxlen=200000)
    loss_history = []
    return_history = []
    policy_loss_fn = nn.CrossEntropyLoss()

    print("Seeding initial dataset...")
    (obs, obs_partial), _ = env.reset(seed=RANDOM_SEED)
    hidden = policy.init_hidden()
    seed_steps = 1000
    for _ in range(seed_steps):
        action, _ = expert.predict(obs, deterministic=True)

        dataset.append((copy.deepcopy(obs_partial), action, hidden.detach().clone()))
        (obs, obs_partial), _, terminated, _ = env.step(action)

        with torch.no_grad():
            obs_tensor = (
                torch.tensor(obs_partial, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )
            _, hidden = policy(obs_tensor, hidden)

        if terminated:
            (obs, obs_partial), _ = env.reset(seed=RANDOM_SEED)
            hidden = policy.init_hidden()
    print(f"Dataset seeded with {len(dataset)} samples.")

    for epoch in range(epochs):
        policy.train()
        total_loss = 0.0

        if len(dataset) < BATCH_SIZE:
            print(
                f"Warning: Dataset size ({len(dataset)}) < BATCH_SIZE ({BATCH_SIZE}). Skipping training updates."
            )
            loss_history.append(loss_history[-1] if loss_history else 0)
        else:
            for _ in range(UPDATE_STEPS):
                batch_indices = np.random.choice(
                    len(dataset), BATCH_SIZE, replace=False
                )

                obs_batch = (
                    torch.tensor(
                        np.array([dataset[i][0] for i in batch_indices]),
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                    .to(DEVICE)
                )
                act_batch = torch.tensor(
                    np.array([dataset[i][1] for i in batch_indices]), dtype=torch.long
                ).to(DEVICE)
                hidden_batch = torch.cat([dataset[i][2] for i in batch_indices], dim=1)

                pred_actions_logits, _ = policy(obs_batch, hidden_batch)
                loss = policy_loss_fn(pred_actions_logits.squeeze(0), act_batch)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                total_loss += loss.item()
            loss_history.append(total_loss / UPDATE_STEPS)

        policy.eval()
        (obs, obs_partial), _ = env.reset(seed=RANDOM_SEED + epoch + 1)
        hidden = policy.init_hidden()
        terminated = False
        timesteps = 0
        while not terminated and timesteps < EPISODE_LENGTH:
            prev_hidden = hidden.detach().clone()
            obs_tensor = (
                torch.tensor(obs_partial, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():
                policy_act_logits, hidden = policy(obs_tensor, hidden)
                action = policy_act_logits.squeeze(0).squeeze(0).argmax().item()

            expert_act, _ = expert.predict(obs, deterministic=True)

            dataset.append((copy.deepcopy(obs_partial), expert_act, prev_hidden))

            (obs, obs_partial), _, terminated, _ = env.step(action)
            timesteps += 1

        avg_ret = np.mean(
            evaluate_policy(
                env, policy, expert_for_eval=expert, episodes=5, silent=True
            )
        )
        return_history.append(avg_ret)
        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {loss_history[-1]:.4f} | Avg Eval Return: {avg_ret:.2f}"
        )

    end_time = time.time()
    print(
        f"Standard DAgger (GRU) training finished in {end_time - start_time:.2f} seconds."
    )

    history = {"losses": loss_history, "returns": return_history}
    return policy, history


def train_dagger_mlp(env, expert, policy, epochs=TOTAL_EPOCHS):
    """Train DAgger MLP policy to imitate the expert"""
    print("\n--- Training Standard DAgger (MLP) ---")
    start_time = time.time()

    optimizer = optim.Adam(policy.parameters(), lr=POLICY_LR)
    dataset = deque(maxlen=200000)
    loss_history = []
    return_history = []
    policy_loss_fn = nn.CrossEntropyLoss()

    print("Seeding initial dataset...")
    (obs, obs_partial), _ = env.reset(seed=RANDOM_SEED)
    seed_steps = 1000
    for _ in range(seed_steps):
        action, _ = expert.predict(obs, deterministic=True)
        dataset.append((copy.deepcopy(obs_partial), action))
        (obs, obs_partial), _, terminated, _ = env.step(action)
        if terminated:
            (obs, obs_partial), _ = env.reset(seed=RANDOM_SEED)
    print(f"Dataset seeded with {len(dataset)} samples.")

    for epoch in range(epochs):
        policy.train()
        total_loss = 0.0

        if len(dataset) < BATCH_SIZE:
            print(
                f"Warning: Dataset size ({len(dataset)}) < BATCH_SIZE ({BATCH_SIZE}). Skipping training updates."
            )
            loss_history.append(loss_history[-1] if loss_history else 0)
        else:
            for _ in range(UPDATE_STEPS):
                batch_indices = np.random.choice(
                    len(dataset), BATCH_SIZE, replace=False
                )
                obs_batch = torch.tensor(
                    np.array([dataset[i][0] for i in batch_indices]),
                    dtype=torch.float32,
                ).to(DEVICE)
                act_batch = torch.tensor(
                    np.array([dataset[i][1] for i in batch_indices]), dtype=torch.long
                ).to(DEVICE)

                pred_actions_logits = policy(obs_batch)
                loss = policy_loss_fn(pred_actions_logits, act_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss_history.append(total_loss / UPDATE_STEPS)

        policy.eval()
        (obs, obs_partial), _ = env.reset(seed=RANDOM_SEED + epoch + 1)
        terminated = False
        timesteps = 0
        while not terminated and timesteps < EPISODE_LENGTH:
            obs_tensor = (
                torch.tensor(obs_partial, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            )
            with torch.no_grad():
                policy_act_logits = policy(obs_tensor)
                action = policy_act_logits.squeeze(0).argmax().item()

            expert_act, _ = expert.predict(obs, deterministic=True)
            dataset.append((copy.deepcopy(obs_partial), expert_act))

            (obs, obs_partial), _, terminated, _ = env.step(action)
            timesteps += 1

        avg_ret = np.mean(
            evaluate_policy(
                env, policy, expert_for_eval=expert, episodes=5, silent=True
            )
        )
        return_history.append(avg_ret)
        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {loss_history[-1]:.4f} | Avg Eval Return: {avg_ret:.2f}"
        )

    end_time = time.time()
    print(
        f"Standard DAgger (MLP) training finished in {end_time - start_time:.2f} seconds."
    )

    history = {"losses": loss_history, "returns": return_history}
    return policy, history


def train_safedagger_gru(env, expert, policy, safety_classifier, epochs=TOTAL_EPOCHS):
    """Train SafeDAgger GRU policy"""
    print("\n--- Training SafeDAgger (GRU) ---")
    start_time = time.time()

    policy_optimizer = optim.Adam(policy.parameters(), lr=POLICY_LR)
    safety_optimizer = optim.Adam(safety_classifier.parameters(), lr=SAFETY_LR)

    policy_dataset = deque(maxlen=200000)
    safety_dataset = deque(maxlen=200000)

    policy_loss_history = []
    safety_loss_history = []
    return_history = []
    intervention_history = []

    policy_loss_fn = nn.CrossEntropyLoss()
    safety_loss_fn = nn.BCEWithLogitsLoss()

    print("Seeding initial policy dataset...")
    (obs, obs_partial), _ = env.reset(seed=RANDOM_SEED)
    hidden = policy.init_hidden()
    seed_steps = 1000
    for _ in range(seed_steps):
        action, _ = expert.predict(obs, deterministic=True)
        policy_dataset.append(
            (copy.deepcopy(obs_partial), action, hidden.detach().clone())
        )
        (obs, obs_partial), _, terminated, _ = env.step(action)

        with torch.no_grad():
            obs_tensor = (
                torch.tensor(obs_partial, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )
            _, hidden = policy(obs_tensor, hidden)
        if terminated:
            (obs, obs_partial), _ = env.reset(seed=RANDOM_SEED)
            hidden = policy.init_hidden()
    print(f"Policy dataset seeded with {len(policy_dataset)} samples.")

    for epoch in range(epochs):
        policy.train()
        safety_classifier.train()
        total_policy_loss = 0.0
        total_safety_loss = 0.0

        update_performed = False
        if len(policy_dataset) >= BATCH_SIZE and len(safety_dataset) >= BATCH_SIZE:
            update_performed = True
            for _ in range(UPDATE_STEPS):

                batch_indices_p = np.random.choice(
                    len(policy_dataset), BATCH_SIZE, replace=False
                )
                obs_batch_p = (
                    torch.tensor(
                        np.array([policy_dataset[i][0] for i in batch_indices_p]),
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                    .to(DEVICE)
                )
                act_batch_p = torch.tensor(
                    np.array([policy_dataset[i][1] for i in batch_indices_p]),
                    dtype=torch.long,
                ).to(DEVICE)
                hidden_batch_p = torch.cat(
                    [policy_dataset[i][2] for i in batch_indices_p], dim=1
                )

                policy_optimizer.zero_grad()
                pred_actions_logits_p, _ = policy(obs_batch_p, hidden_batch_p)
                policy_loss = policy_loss_fn(
                    pred_actions_logits_p.squeeze(0), act_batch_p
                )
                policy_loss.backward()
                policy_optimizer.step()
                total_policy_loss += policy_loss.item()

                batch_indices_s = np.random.choice(
                    len(safety_dataset), BATCH_SIZE, replace=False
                )
                obs_batch_s = (
                    torch.tensor(
                        np.array([safety_dataset[i][0] for i in batch_indices_s]),
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                    .to(DEVICE)
                )
                label_batch_s = (
                    torch.tensor(
                        np.array([safety_dataset[i][1] for i in batch_indices_s]),
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .to(DEVICE)
                )
                hidden_batch_s = torch.cat(
                    [safety_dataset[i][2] for i in batch_indices_s], dim=1
                )

                safety_optimizer.zero_grad()
                pred_safety_logits_s, _ = safety_classifier(obs_batch_s, hidden_batch_s)
                safety_loss = safety_loss_fn(pred_safety_logits_s, label_batch_s)
                safety_loss.backward()
                safety_optimizer.step()
                total_safety_loss += safety_loss.item()

            policy_loss_history.append(total_policy_loss / UPDATE_STEPS)
            safety_loss_history.append(total_safety_loss / UPDATE_STEPS)
        else:
            print(
                f"Warning: Dataset sizes (Policy: {len(policy_dataset)}, Safety: {len(safety_dataset)}) < BATCH_SIZE ({BATCH_SIZE}). Skipping training updates."
            )
            policy_loss_history.append(
                policy_loss_history[-1] if policy_loss_history else 0
            )
            safety_loss_history.append(
                safety_loss_history[-1] if safety_loss_history else 0
            )

        policy.eval()
        safety_classifier.eval()
        (obs, obs_partial), _ = env.reset(seed=RANDOM_SEED + epoch + 1)
        policy_hidden = policy.init_hidden()

        terminated = False
        timesteps = 0
        expert_interventions = 0
        steps_in_rollout = 0

        while not terminated and timesteps < EPISODE_LENGTH:
            steps_in_rollout += 1
            current_partial_obs = copy.deepcopy(obs_partial)
            prev_policy_hidden = policy_hidden.detach().clone()

            obs_tensor = (
                torch.tensor(current_partial_obs, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():

                student_logits, next_policy_hidden = policy(obs_tensor, policy_hidden)
                student_action = student_logits.squeeze(0).squeeze(0).argmax().item()
                safety_label = 1.0 if student_action == action else 0.0
                safety_dataset.append((obs_partial, safety_label, hidden.clone()))

                safety_logits, _ = safety_classifier(obs_tensor, policy_hidden)
                safety_pred = safety_logits.squeeze(0).squeeze(0).item()

                expert_action, _ = expert.predict(obs, deterministic=True)

            is_safe = safety_pred > SAFETY_THRESHOLD
            if is_safe:
                action_to_execute = student_action
            else:
                action_to_execute = expert_action
                expert_interventions += 1

            (obs, obs_partial), _, terminated, _ = env.step(action_to_execute)

            policy_hidden = next_policy_hidden

            policy_dataset.append(
                (current_partial_obs, expert_action, prev_policy_hidden)
            )

            safety_label = 1.0 if student_action == expert_action else 0.0
            safety_dataset.append(
                (current_partial_obs, safety_label, prev_policy_hidden)
            )

            timesteps += 1

        intervention_rate = (
            expert_interventions / steps_in_rollout if steps_in_rollout > 0 else 0
        )
        intervention_history.append(intervention_rate)

        avg_ret = np.mean(
            evaluate_policy(
                env, policy, expert_for_eval=expert, episodes=5, silent=True
            )
        )
        return_history.append(avg_ret)

        if update_performed:
            print(
                f"Epoch {epoch+1}/{epochs} | P_Loss: {policy_loss_history[-1]:.4f} | S_Loss: {safety_loss_history[-1]:.4f} | Interv: {intervention_rate:.2f} | Avg Eval Return: {avg_ret:.2f}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{epochs} | (No Updates) | Interv: {intervention_rate:.2f} | Avg Eval Return: {avg_ret:.2f}"
            )

    end_time = time.time()
    print(f"SafeDAgger (GRU) training finished in {end_time - start_time:.2f} seconds.")

    history = {
        "policy_losses": policy_loss_history,
        "safety_losses": safety_loss_history,
        "returns": return_history,
        "interventions": intervention_history,
    }
    return policy, history


def evaluate_policy(
    env,
    policy_fn,
    expert_for_eval=None,
    episodes=10,
    silent=False,
    visualize=False,
    n_vis_episodes=3,
    gif_filename="policy_eval.gif",
):
    """Evaluate the policy, optionally render to gif."""

    is_wrapped = isinstance(env, EnvWrapper)
    if isinstance(policy_fn, (GRUPolicy, MLPPolicy)) and not is_wrapped:
        print(
            "Warning: Evaluating MLP/GRU policy on non-wrapped env. Assuming full observation."
        )

    returns = []
    frames = []
    policy_name = "Unknown"
    if isinstance(policy_fn, PPO):
        policy_name = "PPO Expert"
    elif isinstance(policy_fn, MLPPolicy):
        policy_name = "MLP Policy"
    elif isinstance(policy_fn, GRUPolicy):
        policy_name = "GRU Policy"

    if not silent:
        print(f"\n--- Evaluating {policy_name} ---")

    for i in range(episodes):
        if is_wrapped:

            if expert_for_eval is None and isinstance(policy_fn, PPO):
                expert_for_eval = policy_fn
            (obs, obs_partial), _ = env.reset(seed=RANDOM_SEED + 1000 + i)
        else:

            obs, _ = env.reset(seed=RANDOM_SEED + 1000 + i)
            obs_partial = obs

        terminated = False
        total_reward = 0
        hidden_state = None

        timesteps = 0

        max_steps = getattr(env, "_max_episode_steps", EPISODE_LENGTH)

        while not terminated and timesteps < max_steps:
            action = None
            if isinstance(policy_fn, PPO):

                current_obs_for_expert = obs if is_wrapped else obs_partial
                action, _ = policy_fn.predict(
                    current_obs_for_expert, deterministic=True
                )
            elif isinstance(policy_fn, MLPPolicy):

                obs_tensor = (
                    torch.tensor(obs_partial, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(DEVICE)
                )
                with torch.no_grad():
                    logits = policy_fn(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
            elif isinstance(policy_fn, GRUPolicy):

                obs_tensor = (
                    torch.tensor(obs_partial, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(DEVICE)
                )
                if hidden_state is None:
                    hidden_state = policy_fn.init_hidden(batch_size=1)
                with torch.no_grad():
                    action_logits, hidden_state = policy_fn(obs_tensor, hidden_state)
                action = action_logits.squeeze(0).squeeze(0).argmax().item()
            else:
                raise TypeError("Unknown policy type for evaluation")

            if is_wrapped:
                (obs, obs_partial), reward, terminated, _ = env.step(action)
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                terminated = terminated or truncated
                obs_partial = obs

            total_reward += reward
            timesteps += 1

            should_render = (
                visualize
                and i < n_vis_episodes
                and hasattr(env.env, "render_mode")
                and env.env.render_mode == "rgb_array"
            )
            if should_render:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    print(f"Warning: Rendering failed during evaluation. {e}")
                    visualize = False

        returns.append(total_reward)
        if not silent and (episodes <= 20 or (i + 1) % (episodes // 10) == 0):
            print(f"Eval Episode {i + 1}/{episodes} reward: {total_reward:.2f}")

    if visualize and frames and gif_filename:
        print(f"Saving visualization to {gif_filename}...")
        try:
            imageio.mimsave(gif_filename, frames, fps=30)
        except Exception as e:
            print(f"Error saving gif {gif_filename}: {e}")
    elif visualize and not frames:
        print(
            f"No frames recorded for {gif_filename}, visualization might require 'rgb_array' render mode."
        )

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    if not silent:
        print(f"--- {policy_name} Evaluation Summary ({episodes} episodes) ---")
        print(f"Average Return: {avg_return:.2f} +/- {std_return:.2f}")
        print("-" * (30 + len(policy_name)))

    return returns


def plot_training_comparison(history, title_prefix, filename):
    """Plot loss and return collected from training history dictionary."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    if "policy_losses" in history:
        axs[0].plot(
            history["policy_losses"],
            label=f"{title_prefix} Policy Loss",
            color="red",
            alpha=0.8,
        )
    elif "losses" in history:
        axs[0].plot(
            history["losses"],
            label=f"{title_prefix} Policy Loss",
            color="red",
            alpha=0.8,
        )
    axs[0].set_ylabel("Policy Loss")
    axs[0].legend(loc="upper left")
    axs[0].grid(True)

    if "safety_losses" in history:
        ax0_twin = axs[0].twinx()
        ax0_twin.plot(
            history["safety_losses"],
            label=f"{title_prefix} Safety Loss",
            color="green",
            linestyle="--",
            alpha=0.6,
        )
        ax0_twin.set_ylabel("Safety Loss (BCE)", color="green")
        ax0_twin.tick_params(axis="y", labelcolor="green")
        ax0_twin.legend(loc="upper right")

    axs[1].plot(
        history["returns"],
        label=f"{title_prefix} Avg Eval Return",
        color="blue",
        alpha=0.8,
    )
    axs[1].set_ylabel("Return")
    axs[1].legend(loc="upper left")
    axs[1].grid(True)

    if "interventions" in history:
        ax1_twin = axs[1].twinx()
        ax1_twin.plot(
            history["interventions"],
            label=f"{title_prefix} Intervention Rate",
            color="purple",
            linestyle=":",
            alpha=0.6,
        )
        ax1_twin.set_ylabel("Expert Intervention Rate", color="purple")
        ax1_twin.tick_params(axis="y", labelcolor="purple")
        ax1_twin.set_ylim(0, 1.05)
        ax1_twin.legend(loc="upper right")

    axs[1].set_xlabel("Epoch")
    fig.suptitle(f"{title_prefix} Training Progress")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(SAVE_ROOT, filename))
    print(f"Saved training plot: {filename}")
    plt.close(fig)


def plot_final_returns(returns_dict):
    """Compare final evaluation returns of multiple policies."""
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(returns_dict)))
    markers = [
        "o",
        "s",
        "^",
        "d",
        "*",
        "p",
    ]

    all_returns = np.concatenate(list(returns_dict.values()))
    min_ret, max_ret = np.min(all_returns), np.max(all_returns)
    bins = np.linspace(min_ret, max_ret, 25)

    for i, (name, returns) in enumerate(returns_dict.items()):
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        label_text = f"{name}\n(Mean: {mean_ret:.1f} +/- {std_ret:.1f})"

        plt.hist(
            returns,
            bins=bins,
            alpha=0.6,
            label=label_text,
            density=True,
            color=colors[i],
        )

    plt.xlabel("Total Return per Episode")
    plt.ylabel("Density")
    plt.grid(True, axis="y", alpha=0.5)
    plt.title(
        f"Final Policy Evaluation Comparison ({len(next(iter(returns_dict.values())))} episodes each)"
    )
    plt.legend(fontsize="small")
    plt.tight_layout()
    save_path = os.path.join(SAVE_ROOT, "final_returns_comparison_hist.png")
    plt.savefig(save_path)
    print(f"Saved final returns comparison plot: {os.path.basename(save_path)}")
    plt.close()


if __name__ == "__main__":

    num_experiments = 1

    all_expert_returns = []
    all_dagger_mlp_returns = []
    all_dagger_gru_returns = []
    all_safedagger_gru_returns = []

    for exp in range(num_experiments):
        print(f"\n{'='*20} Experiment {exp+1}/{num_experiments} {'='*20}")
        RANDOM_SEED = exp
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)

        env_expert_train = gym.make("LunarLander-v2")
        env_dagger_train = EnvWrapper(
            gym.make("LunarLander-v2"), DROPOUT_RATE, DROPOUT_MASK
        )
        env_eval_expert = gym.make("LunarLander-v2", render_mode="rgb_array")
        env_eval_dagger = EnvWrapper(
            gym.make("LunarLander-v2", render_mode="rgb_array"),
            DROPOUT_RATE,
            DROPOUT_MASK,
        )

        SAVE_ROOT = os.path.join(
            "results", f"dropout_{DROPOUT_RATE}", f"seed_{RANDOM_SEED}"
        )
        if not os.path.exists(SAVE_ROOT):
            os.makedirs(SAVE_ROOT)
        print(f"Results will be saved in: {SAVE_ROOT}")

        print("\n--- Preparing PPO Expert ---")
        expert_model = train_ppo_expert()

        dagger_mlp_model = None
        dagger_mlp_history = None
        if TRAIN_NEW_MLP_POLICY:
            policy_mlp = MLPPolicy(
                env_dagger_train.observation_space.shape[0],
                env_dagger_train.action_space.n,
            ).to(DEVICE)
            dagger_mlp_model, dagger_mlp_history = train_dagger_mlp(
                env_dagger_train, expert_model, policy_mlp, epochs=TOTAL_EPOCHS
            )
            save_path = os.path.join(SAVE_ROOT, "dagger_lunarlander_mlp.pt")
            torch.save(dagger_mlp_model.state_dict(), save_path)
            print(f"DAgger MLP policy saved to {save_path}")
            plot_training_comparison(
                dagger_mlp_history, "DAgger MLP", "dagger_mlp_training_progress.png"
            )
        else:
            print("Loading DAgger MLP policy...")
            load_path = os.path.join(SAVE_ROOT, "dagger_lunarlander_mlp.pt")
            if os.path.exists(load_path):
                dagger_mlp_model = MLPPolicy(
                    env_dagger_train.observation_space.shape[0],
                    env_dagger_train.action_space.n,
                ).to(DEVICE)
                dagger_mlp_model.load_state_dict(
                    torch.load(load_path, map_location=DEVICE)
                )
            else:
                print(
                    f"Error: DAgger MLP model file not found at {load_path}. Cannot load."
                )

                dagger_mlp_model = None

        dagger_gru_model = None
        dagger_gru_history = None
        if TRAIN_NEW_GRU_POLICY:
            policy_gru = GRUPolicy(
                env_dagger_train.observation_space.shape[0],
                env_dagger_train.action_space.n,
            ).to(DEVICE)
            dagger_gru_model, dagger_gru_history = train_dagger_gru(
                env_dagger_train, expert_model, policy_gru, epochs=TOTAL_EPOCHS
            )
            save_path = os.path.join(SAVE_ROOT, "dagger_lunarlander_gru.pt")
            torch.save(dagger_gru_model.state_dict(), save_path)
            print(f"DAgger GRU policy saved to {save_path}")
            plot_training_comparison(
                dagger_gru_history, "DAgger GRU", "dagger_gru_training_progress.png"
            )
        else:
            print("Loading DAgger GRU policy...")
            load_path = os.path.join(SAVE_ROOT, "dagger_lunarlander_gru.pt")
            if os.path.exists(load_path):
                dagger_gru_model = GRUPolicy(
                    env_dagger_train.observation_space.shape[0],
                    env_dagger_train.action_space.n,
                ).to(DEVICE)
                dagger_gru_model.load_state_dict(
                    torch.load(load_path, map_location=DEVICE)
                )
            else:
                print(f"Error: DAgger GRU model file not found at {load_path}.")
                dagger_gru_model = None

        safedagger_gru_model = None
        safedagger_gru_history = None
        if TRAIN_NEW_SAFEDAGGER_GRU_POLICY:
            policy_safedagger_gru = GRUPolicy(
                env_dagger_train.observation_space.shape[0],
                env_dagger_train.action_space.n,
            ).to(DEVICE)
            safety_classifier = GRUSafetyClassifier(
                env_dagger_train.observation_space.shape[0]
            ).to(DEVICE)
            safedagger_gru_model, safedagger_gru_history = train_safedagger_gru(
                env_dagger_train,
                expert_model,
                policy_safedagger_gru,
                safety_classifier,
                epochs=TOTAL_EPOCHS,
            )
            save_path = os.path.join(SAVE_ROOT, "safedagger_lunarlander_gru.pt")
            torch.save(safedagger_gru_model.state_dict(), save_path)
            print(f"SafeDAgger GRU policy saved to {save_path}")
            plot_training_comparison(
                safedagger_gru_history,
                "SafeDAgger GRU",
                "safedagger_gru_training_progress.png",
            )
        else:
            print("Loading SafeDAgger GRU policy...")
            load_path = os.path.join(SAVE_ROOT, "safedagger_lunarlander_gru.pt")
            if os.path.exists(load_path):
                safedagger_gru_model = GRUPolicy(
                    env_dagger_train.observation_space.shape[0],
                    env_dagger_train.action_space.n,
                ).to(DEVICE)
                safedagger_gru_model.load_state_dict(
                    torch.load(load_path, map_location=DEVICE)
                )
            else:
                print(f"Error: SafeDAgger GRU model file not found at {load_path}.")
                safedagger_gru_model = None

        n_eval_episodes = 50
        vis_ep = 1

        print("\n--- Final Evaluations ---")
        expert_returns = evaluate_policy(
            env_eval_expert,
            expert_model,
            episodes=n_eval_episodes,
            visualize=vis_ep > 0,
            n_vis_episodes=vis_ep,
            gif_filename=os.path.join(SAVE_ROOT, "expert_policy_final.gif"),
        )
        all_expert_returns.append(expert_returns)

        dagger_mlp_returns = [np.nan] * n_eval_episodes
        if dagger_mlp_model:
            dagger_mlp_returns = evaluate_policy(
                env_eval_dagger,
                dagger_mlp_model,
                expert_for_eval=expert_model,
                episodes=n_eval_episodes,
                visualize=vis_ep > 0,
                n_vis_episodes=vis_ep,
                gif_filename=os.path.join(SAVE_ROOT, "dagger_mlp_policy_final.gif"),
            )
        all_dagger_mlp_returns.append(dagger_mlp_returns)

        dagger_gru_returns = [np.nan] * n_eval_episodes
        if dagger_gru_model:
            dagger_gru_returns = evaluate_policy(
                env_eval_dagger,
                dagger_gru_model,
                expert_for_eval=expert_model,
                episodes=n_eval_episodes,
                visualize=vis_ep > 0,
                n_vis_episodes=vis_ep,
                gif_filename=os.path.join(SAVE_ROOT, "dagger_gru_policy_final.gif"),
            )
        all_dagger_gru_returns.append(dagger_gru_returns)

        safedagger_gru_returns = [np.nan] * n_eval_episodes
        if safedagger_gru_model:
            safedagger_gru_returns = evaluate_policy(
                env_eval_dagger,
                safedagger_gru_model,
                expert_for_eval=expert_model,
                episodes=n_eval_episodes,
                visualize=vis_ep > 0,
                n_vis_episodes=vis_ep,
                gif_filename=os.path.join(SAVE_ROOT, "safedagger_gru_policy_final.gif"),
            )
        all_safedagger_gru_returns.append(safedagger_gru_returns)

        print("\nSaving evaluation results...")
        np.save(os.path.join(SAVE_ROOT, "expert_returns.npy"), expert_returns)
        if dagger_mlp_model:
            np.save(
                os.path.join(SAVE_ROOT, "dagger_mlp_returns.npy"), dagger_mlp_returns
            )
        if dagger_gru_model:
            np.save(
                os.path.join(SAVE_ROOT, "dagger_gru_returns.npy"), dagger_gru_returns
            )
        if safedagger_gru_model:
            np.save(
                os.path.join(SAVE_ROOT, "safedagger_gru_returns.npy"),
                safedagger_gru_returns,
            )

        print("\nPlotting final results for this seed...")
        returns_to_plot = {
            "Expert PPO": expert_returns,
        }
        if dagger_mlp_model:
            returns_to_plot["DAgger MLP"] = dagger_mlp_returns
        if dagger_gru_model:
            returns_to_plot["DAgger GRU"] = dagger_gru_returns
        if safedagger_gru_model:
            returns_to_plot["SafeDAgger GRU"] = safedagger_gru_returns

        if len(returns_to_plot) > 1:
            plot_final_returns(returns_to_plot)
        else:
            print(
                "Not enough policies trained/loaded to generate final comparison plot."
            )

        env_expert_train.close()
        env_dagger_train.close()
        env_eval_expert.close()
        env_eval_dagger.close()

    print(f"\n{'='*20} All Experiments Finished {'='*20}")

    if num_experiments > 1:
        print("\n--- Aggregated Results Across Seeds ---")

        mean_expert = np.mean([np.mean(r) for r in all_expert_returns])
        std_expert = np.std([np.mean(r) for r in all_expert_returns])
        print(f"Expert PPO      : {mean_expert:.2f} +/- {std_expert:.2f}")

        valid_mlp = [r for r in all_dagger_mlp_returns if not np.isnan(r).all()]
        if valid_mlp:
            mean_mlp = np.mean([np.mean(r) for r in valid_mlp])
            std_mlp = np.std([np.mean(r) for r in valid_mlp])
            print(
                f"DAgger MLP        : {mean_mlp:.2f} +/- {std_mlp:.2f} ({len(valid_mlp)} seeds)"
            )

        valid_gru = [r for r in all_dagger_gru_returns if not np.isnan(r).all()]
        if valid_gru:
            mean_gru = np.mean([np.mean(r) for r in valid_gru])
            std_gru = np.std([np.mean(r) for r in valid_gru])
            print(
                f"DAgger GRU        : {mean_gru:.2f} +/- {std_gru:.2f} ({len(valid_gru)} seeds)"
            )

        valid_safegru = [r for r in all_safedagger_gru_returns if not np.isnan(r).all()]
        if valid_safegru:
            mean_safegru = np.mean([np.mean(r) for r in valid_safegru])
            std_safegru = np.std([np.mean(r) for r in valid_safegru])
            print(
                f"SafeDAgger GRU    : {mean_safegru:.2f} +/- {std_safegru:.2f} ({len(valid_safegru)} seeds)"
            )
