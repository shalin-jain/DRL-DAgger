import gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time

from collections import deque
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import copy


def train_ppo_expert(env_id="LunarLander-v2", total_timesteps=200000):
    """Helper function to train a PPO expert using Stable Baselines"""
    save_path = "ppo_expert.zip"
    if os.path.exists(save_path):
        print(f"Loading pre-trained PPO model from {save_path}")
        model = PPO.load(save_path)
        return model
    env = gym.make(env_id)
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=0)
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
        frame_count = 0
        max_frames = 1000

        while not done and frame_count < max_frames:
            frame = env.render()
            frames.append(frame)
            action = expert_policy_ppo(obs, ppo_model)
            step_result = env.step(action)
            obs = step_result[0]
            reward = step_result[1]
            done = step_result[2]

            if len(step_result) > 4:
                info = step_result[4]
            else:
                done = step_result[2] or step_result[3]
                info = {}

            total_reward += reward
            frame_count += 1
        print(f"PPO Expert Episode {episode+1}: Total Reward = {total_reward}")

    env.close()
    if frames:
        imageio.mimsave(gif_filename, frames, fps=30)
        print(f"Saved PPO expert gif as {gif_filename}")
    else:
        print("No frames recorded for PPO expert gif.")


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
            step_result = env.step(action)
            obs = step_result[0]
            reward = step_result[1]
            done = step_result[2]

            if len(step_result) > 4:
                info = step_result[4]
            else:
                done = step_result[2] or step_result[3]
                info = {}
            total_reward += reward
        rewards.append(total_reward)

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print("-" * 30)
    print(f"PPO Expert Evaluation ({n_episodes} episodes):")
    print(f"Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print("-" * 30)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker=".", linestyle="-")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Expert Evaluation Rewards")
    plt.grid(True)
    plt.savefig("ppo_expert_rewards.png")
    plt.close()
    print("Saved PPO expert reward curve as ppo_expert_rewards.png")
    return rewards


class Policy(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class SafetyClassifier(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def standard_dagger(
    env,
    policy,
    expert_policy_fn,
    expert_model,
    iterations=500,
    rollout_length=100,
    batch_size=64,
    lr=1e-3,
    update_steps=5,
):
    """Standard Dagger Training Loop (originally called dagger)"""
    print("--- Running Standard DAgger ---")
    start_time = time.time()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataset = deque(maxlen=500000)
    losses = []
    returns = []

    for i in tqdm(range(iterations), desc="Standard DAgger Iterations"):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        episode_return = 0

        done = False
        env_steps = 0
        while not done and env_steps < rollout_length:
            current_obs = copy.deepcopy(obs)
            obs_tensor = torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0)

            policy.eval()
            with torch.no_grad():
                logits = policy(obs_tensor)
                pred_action = torch.argmax(logits, dim=1).item()
            policy.train()

            expert_action = expert_policy_fn(current_obs, expert_model)
            dataset.append((current_obs, expert_action))

            step_result = env.step(pred_action)
            obs = step_result[0]
            reward = step_result[1]
            done = step_result[2]

            if len(step_result) > 4:
                info = step_result[4]
            else:
                done = step_result[2] or step_result[3]
                info = {}

            episode_return += reward
            env_steps += 1

            if done:

                pass

        returns.append(episode_return)

        iter_loss = 0.0
        if len(dataset) >= batch_size:
            policy.train()

            for j in range(update_steps):

                batch_indices = np.random.choice(
                    len(dataset), batch_size, replace=False
                )
                batch_states = torch.tensor(
                    [dataset[k][0] for k in batch_indices], dtype=torch.float32
                )
                batch_actions = torch.tensor(
                    [dataset[k][1] for k in batch_indices], dtype=torch.long
                )

                optimizer.zero_grad()
                logits = policy(batch_states)
                loss = loss_fn(logits, batch_actions)
                loss.backward()
                optimizer.step()
                iter_loss += loss.item()

            avg_loss = iter_loss / update_steps
            losses.append(avg_loss)
            if i % 50 == 0 or i == iterations - 1:
                tqdm.write(
                    f"Std DAgger Iter {i+1}: Loss = {avg_loss:.4f}, Return = {episode_return:.2f}"
                )
        elif len(dataset) > 0:
            losses.append(losses[-1] if losses else 0)

    end_time = time.time()
    print(f"Standard DAgger training finished in {end_time - start_time:.2f} seconds.")

    history = {"losses": losses, "returns": returns}
    return policy, history


def safedagger(
    env,
    policy,
    safety_classifier,
    expert_policy_fn,
    expert_model,
    iterations=500,
    rollout_length=100,
    batch_size=64,
    policy_lr=1e-3,
    safety_lr=1e-3,
    update_steps=5,
    safety_threshold=0.0,
):
    """Main SafeDAgger Training Loop"""
    print("--- Running SafeDAgger ---")
    start_time = time.time()

    policy_optimizer = optim.Adam(policy.parameters(), lr=policy_lr)
    safety_optimizer = optim.Adam(safety_classifier.parameters(), lr=safety_lr)

    policy_loss_fn = nn.CrossEntropyLoss()
    safety_loss_fn = nn.BCEWithLogitsLoss()

    policy_dataset = deque(maxlen=500000)
    safety_dataset = deque(maxlen=500000)

    policy_losses = []
    safety_losses = []
    returns = []
    safety_interventions = []

    for i in tqdm(range(iterations), desc="SafeDAgger Iterations"):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        episode_return = 0
        done = False
        env_steps = 0
        expert_chosen_count = 0
        steps_in_iteration = 0

        while not done and env_steps < rollout_length:
            current_obs = copy.deepcopy(obs)
            obs_tensor = torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0)

            policy.eval()
            safety_classifier.eval()

            with torch.no_grad():
                student_logits = policy(obs_tensor)
                student_action = torch.argmax(student_logits, dim=1).item()
                expert_action = expert_policy_fn(current_obs, expert_model)
                safety_logit = safety_classifier(obs_tensor)
                safety_pred = safety_logit.item()

            is_safe = safety_pred > safety_threshold
            action_to_execute = student_action if is_safe else expert_action
            if not is_safe:
                expert_chosen_count += 1

            step_result = env.step(action_to_execute)
            obs = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            if len(step_result) > 4:
                info = step_result[4]
            else:
                done = step_result[2] or step_result[3]
                info = {}

            episode_return += reward
            env_steps += 1
            steps_in_iteration += 1

            policy_dataset.append((current_obs, expert_action))
            safety_label = 1.0 if student_action == expert_action else 0.0
            safety_dataset.append((current_obs, safety_label))

            if done:
                pass

        returns.append(episode_return)
        intervention_rate = (
            expert_chosen_count / steps_in_iteration if steps_in_iteration > 0 else 0
        )
        safety_interventions.append(intervention_rate)

        iter_policy_loss = 0.0
        iter_safety_loss = 0.0
        if len(policy_dataset) >= batch_size and len(safety_dataset) >= batch_size:
            policy.train()
            safety_classifier.train()

            for j in range(update_steps):

                policy_batch_indices = np.random.choice(
                    len(policy_dataset), batch_size, replace=False
                )
                policy_batch_states = torch.tensor(
                    [policy_dataset[k][0] for k in policy_batch_indices],
                    dtype=torch.float32,
                )
                policy_batch_actions = torch.tensor(
                    [policy_dataset[k][1] for k in policy_batch_indices],
                    dtype=torch.long,
                )
                policy_optimizer.zero_grad()
                logits = policy(policy_batch_states)
                loss = policy_loss_fn(logits, policy_batch_actions)
                loss.backward()
                policy_optimizer.step()
                iter_policy_loss += loss.item()

                safety_batch_indices = np.random.choice(
                    len(safety_dataset), batch_size, replace=False
                )
                safety_batch_states = torch.tensor(
                    [safety_dataset[k][0] for k in safety_batch_indices],
                    dtype=torch.float32,
                )
                safety_batch_labels = torch.tensor(
                    [safety_dataset[k][1] for k in safety_batch_indices],
                    dtype=torch.float32,
                ).unsqueeze(1)
                safety_optimizer.zero_grad()
                safety_logits = safety_classifier(safety_batch_states)
                safety_loss = safety_loss_fn(safety_logits, safety_batch_labels)
                safety_loss.backward()
                safety_optimizer.step()
                iter_safety_loss += safety_loss.item()

            avg_policy_loss = iter_policy_loss / update_steps
            avg_safety_loss = iter_safety_loss / update_steps
            policy_losses.append(avg_policy_loss)
            safety_losses.append(avg_safety_loss)

            if i % 50 == 0 or i == iterations - 1:
                tqdm.write(
                    f"SafeDAgger Iter {i+1}: P_Loss={avg_policy_loss:.4f}, S_Loss={avg_safety_loss:.4f}, Ret={episode_return:.2f}, Interv={intervention_rate:.2f}"
                )

        elif len(policy_dataset) > 0:
            policy_losses.append(policy_losses[-1] if policy_losses else 0)
            safety_losses.append(safety_losses[-1] if safety_losses else 0)

    end_time = time.time()
    print(f"SafeDAgger training finished in {end_time - start_time:.2f} seconds.")

    history = {
        "policy_losses": policy_losses,
        "safety_losses": safety_losses,
        "returns": returns,
        "interventions": safety_interventions,
    }
    return policy, history


def evaluate_policy(
    env,
    policy,
    episodes=50,
    vis_episodes=0,
    gif_filename="policy_eval.gif",
    policy_name="Policy",
):
    """Evaluates the final student policy and returns the list of rewards."""
    print(f"\n--- Evaluating {policy_name} ---")
    frames = []
    rewards = []
    policy.eval()

    for episode in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total_reward = 0
        frame_count = 0
        max_frames = 1000

        while not done and (vis_episodes == 0 or frame_count < max_frames):
            render_this_frame = episode < vis_episodes and RENDER_MODE == "rgb_array"
            if render_this_frame:
                frame = env.render()
                frames.append(frame)

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = policy(obs_tensor)
                action = torch.argmax(logits, dim=1).item()

            step_result = env.step(action)
            obs = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            if len(step_result) > 4:
                info = step_result[4]
            else:
                done = step_result[2] or step_result[3]
                info = {}

            total_reward += reward
            if render_this_frame:
                frame_count += 1

        rewards.append(total_reward)
        if episodes <= 20 or episode % (episodes // 10) == 0:
            print(
                f"{policy_name} Eval Episode {episode+1}/{episodes}: Reward = {total_reward:.2f}"
            )

    avg_episode_reward = np.mean(np.array(rewards))
    std_episode_reward = np.std(np.array(rewards))
    print("-" * 50)
    print(f"{policy_name} Final Policy Evaluation ({episodes} episodes):")
    print(f"Average Reward: {avg_episode_reward:.2f} +/- {std_episode_reward:.2f}")
    print("-" * 50)

    env.close()
    if frames and gif_filename:
        try:
            imageio.mimsave(gif_filename, frames, fps=30)
            print(f"Saved {policy_name} evaluation gif as {gif_filename}")
        except Exception as e:
            print(f"Error saving gif {gif_filename}: {e}")
    elif vis_episodes > 0 and gif_filename:
        print(f"No frames recorded for {policy_name} gif ({gif_filename}).")

    return rewards


def plot_comparison(
    dagger_history,
    safedagger_history,
    dagger_eval_rewards,
    safedagger_eval_rewards,
    expert_avg_reward=None,
):
    """Generates plots comparing DAgger and SafeDAgger"""
    print("\n--- Generating Comparison Plots ---")
    iterations = len(dagger_history["returns"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("DAgger vs SafeDAgger Comparison", fontsize=16)

    axs[0, 0].plot(
        dagger_history["returns"], label="DAgger Training Returns", alpha=0.8
    )
    axs[0, 0].plot(
        safedagger_history["returns"], label="SafeDAgger Training Returns", alpha=0.8
    )
    axs[0, 0].set_xlabel("Training Iterations")
    axs[0, 0].set_ylabel("Episode Return")
    axs[0, 0].set_title("Returns During Training Rollouts")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    dagger_losses = dagger_history["losses"]
    safedagger_policy_losses = safedagger_history["policy_losses"]
    if len(dagger_losses) < iterations:
        dagger_losses.extend([dagger_losses[-1]] * (iterations - len(dagger_losses)))
    if len(safedagger_policy_losses) < iterations:
        safedagger_policy_losses.extend(
            [safedagger_policy_losses[-1]]
            * (iterations - len(safedagger_policy_losses))
        )

    axs[0, 1].plot(dagger_losses[:iterations], label="DAgger Policy Loss", alpha=0.8)
    axs[0, 1].plot(
        safedagger_policy_losses[:iterations], label="SafeDAgger Policy Loss", alpha=0.8
    )

    ax2 = axs[0, 1].twinx()
    safedagger_safety_losses = safedagger_history["safety_losses"]
    if len(safedagger_safety_losses) < iterations:
        safedagger_safety_losses.extend(
            [safedagger_safety_losses[-1]]
            * (iterations - len(safedagger_safety_losses))
        )
    ax2.plot(
        safedagger_safety_losses[:iterations],
        label="SafeDAgger Safety Loss (Right Axis)",
        color="green",
        linestyle="--",
        alpha=0.6,
    )
    ax2.set_ylabel("Safety Loss (BCE)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    axs[0, 1].set_xlabel("Training Iterations")
    axs[0, 1].set_ylabel("Policy Loss (CrossEntropy)")
    axs[0, 1].set_title("Training Losses")

    lines, labels = axs[0, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")
    axs[0, 1].grid(True)

    bins = np.linspace(
        min(min(dagger_eval_rewards), min(safedagger_eval_rewards)),
        max(max(dagger_eval_rewards), max(safedagger_eval_rewards)),
        20,
    )
    axs[1, 0].hist(
        dagger_eval_rewards,
        bins=bins,
        alpha=0.7,
        label="DAgger Final Rewards",
        density=True,
    )
    axs[1, 0].hist(
        safedagger_eval_rewards,
        bins=bins,
        alpha=0.7,
        label="SafeDAgger Final Rewards",
        density=True,
    )

    dagger_mean = np.mean(dagger_eval_rewards)
    safedagger_mean = np.mean(safedagger_eval_rewards)
    axs[1, 0].axvline(
        dagger_mean,
        color="blue",
        linestyle="dashed",
        linewidth=1,
        label=f"DAgger Mean: {dagger_mean:.1f}",
    )
    axs[1, 0].axvline(
        safedagger_mean,
        color="red",
        linestyle="dashed",
        linewidth=1,
        label=f"SafeDAgger Mean: {safedagger_mean:.1f}",
    )
    if expert_avg_reward is not None:
        axs[1, 0].axvline(
            expert_avg_reward,
            color="black",
            linestyle="dotted",
            linewidth=1.5,
            label=f"Expert Mean: {expert_avg_reward:.1f}",
        )

    axs[1, 0].set_xlabel("Total Reward per Episode")
    axs[1, 0].set_ylabel("Density")
    axs[1, 0].set_title(
        f"Distribution of Final Policy Evaluation Rewards ({len(dagger_eval_rewards)} episodes)"
    )
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(
        safedagger_history["interventions"],
        label="Expert Intervention Rate",
        color="purple",
        alpha=0.8,
    )
    axs[1, 1].set_xlabel("Training Iterations")
    axs[1, 1].set_ylabel("Fraction of Steps Using Expert")
    axs[1, 1].set_title("SafeDAgger Safety Interventions During Training")
    axs[1, 1].set_ylim(0, 1.05)
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("dagger_vs_safedagger_comparison.png")
    plt.close()
    print("Saved comparison plots to dagger_vs_safedagger_comparison.png")


if __name__ == "__main__":

    SEED = 42
    ENV_ID = "LunarLander-v2"
    RENDER_MODE = "rgb_array"
    EXPERT_TIMESTEPS = 200000
    DAGGER_ITERATIONS = 200
    ROLLOUT_LENGTH = 500
    BATCH_SIZE = 128
    POLICY_LR = 5e-4
    SAFETY_LR = 5e-4
    UPDATE_STEPS = 10
    SAFETY_THRESHOLD = 0.0
    EVAL_EPISODES = 50
    VIS_EPISODES = 0

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("--- Preparing Expert ---")

    ppo_expert = train_ppo_expert(env_id=ENV_ID, total_timesteps=EXPERT_TIMESTEPS)

    env_expert_eval = gym.make(
        ENV_ID, render_mode=RENDER_MODE if RENDER_MODE and VIS_EPISODES > 0 else None
    )
    expert_rewards = evaluate_expert_rewards(
        ppo_expert, env_expert_eval, n_episodes=EVAL_EPISODES
    )
    expert_avg_reward = np.mean(expert_rewards)
    env_expert_eval.close()

    print("\n--- Training Standard DAgger ---")
    env_dagger_train = gym.make(ENV_ID, render_mode=None)
    policy_dagger = Policy(
        input_dim=env_dagger_train.observation_space.shape[0],
        output_dim=env_dagger_train.action_space.n,
    )
    trained_policy_dagger, dagger_history = standard_dagger(
        env_dagger_train,
        policy_dagger,
        expert_policy_ppo,
        ppo_expert,
        iterations=DAGGER_ITERATIONS,
        rollout_length=ROLLOUT_LENGTH,
        batch_size=BATCH_SIZE,
        lr=POLICY_LR,
        update_steps=UPDATE_STEPS,
    )
    env_dagger_train.close()

    env_dagger_eval = gym.make(
        ENV_ID, render_mode=RENDER_MODE if RENDER_MODE and VIS_EPISODES > 0 else None
    )
    dagger_eval_rewards = evaluate_policy(
        env_dagger_eval,
        trained_policy_dagger,
        episodes=EVAL_EPISODES,
        vis_episodes=VIS_EPISODES,
        gif_filename="dagger_final_policy.gif",
        policy_name="Standard DAgger",
    )

    print("\n--- Training SafeDAgger ---")
    env_safedagger_train = gym.make(ENV_ID, render_mode=None)

    policy_safedagger = Policy(
        input_dim=env_safedagger_train.observation_space.shape[0],
        output_dim=env_safedagger_train.action_space.n,
    )
    safety_classifier_safedagger = SafetyClassifier(
        input_dim=env_safedagger_train.observation_space.shape[0]
    )
    trained_policy_safedagger, safedagger_history = safedagger(
        env_safedagger_train,
        policy_safedagger,
        safety_classifier_safedagger,
        expert_policy_ppo,
        ppo_expert,
        iterations=DAGGER_ITERATIONS,
        rollout_length=ROLLOUT_LENGTH,
        batch_size=BATCH_SIZE,
        policy_lr=POLICY_LR,
        safety_lr=SAFETY_LR,
        update_steps=UPDATE_STEPS,
        safety_threshold=SAFETY_THRESHOLD,
    )
    env_safedagger_train.close()

    env_safedagger_eval = gym.make(
        ENV_ID, render_mode=RENDER_MODE if RENDER_MODE and VIS_EPISODES > 0 else None
    )
    safedagger_eval_rewards = evaluate_policy(
        env_safedagger_eval,
        trained_policy_safedagger,
        episodes=EVAL_EPISODES,
        vis_episodes=VIS_EPISODES,
        gif_filename="safedagger_final_policy.gif",
        policy_name="SafeDAgger",
    )

    plot_comparison(
        dagger_history,
        safedagger_history,
        dagger_eval_rewards,
        safedagger_eval_rewards,
        expert_avg_reward,
    )

    print("\nComparison script finished.")
