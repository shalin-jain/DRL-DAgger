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
from collections import deque
import time
from scipy.stats import entropy  # 确保导入 entropy
import random  # 用于安全数据集采样

# Hyperparameters
HIDDEN_SIZE = 128
BATCH_SIZE = 64
UPDATE_STEPS = 5
TOTAL_EPOCHS = 100
EPISODE_LENGTH = 500
DROPOUT_RATE = 0.0  # Set to non-zero to test dropout
DROPOUT_MASK = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # Match LunarLander obs shape
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERFORMANCE_THRESHOLD = 200  # Target average return for convergence

# --- SafeDAgger Specific Hyperparameters ---
KL_THRESHOLD = 5.0  # <<-- INCREASED THRESHOLD
SAFETY_THRESHOLD = 0.5  # Threshold for safety policy score
EVAL_EPISODES_DURING_TRAINING = 10  # <<-- Increased for stability
EPS = 1e-8  # Small epsilon for numerical stability in log/entropy


# --- EnvWrapper, MLPPolicy, SafetyPolicy, train_ppo_expert, evaluate_policy, plot_* ---
# (Keep these classes/functions as they were in the original code)
# ... (Existing code for these parts) ...


class EnvWrapper:
    def __init__(
        self, env, dropout_rate=0.0, dropout_mask=np.array([1, 1, 1, 1, 1, 1, 1, 1])
    ):
        self.env = env
        self.observation_space = env.observation_space
        # Correct observation space check if needed (LunarLander-v2 is Box(8,))
        if isinstance(self.observation_space, gym.spaces.Box):
            expected_shape = self.observation_space.shape
        else:
            # Handle other space types if necessary, for now assume Box
            expected_shape = (
                self.observation_space.n,
            )  # Example for Discrete, adjust as needed

        assert (
            dropout_mask.shape == expected_shape
        ), f"Expected dropout mask shape {expected_shape}, got {dropout_mask.shape}"

        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask
        self.action_space = env.action_space
        # Ensure dropout_mask is boolean or float for numpy operations
        self.dropout_mask = self.dropout_mask.astype(float)

    def reset(self, seed=None, options=None):
        # Pass seed and options if the underlying env supports Gym API >= 0.26
        # For older Gym versions, just call reset()
        try:
            return self.env.reset(seed=seed, options=options)
        except TypeError:
            # Older Gym version or env doesn't support seed/options in reset
            # Warning: Seeding might not be perfectly reproducible here
            obs = self.env.reset()
            return obs, {}  # Return info dict expected by newer gym

    def render(self):
        return self.env.render()

    def step(self, action):
        # Handle potential tuple return for obs, info in newer Gym
        step_return = self.env.step(action)

        # Unpack based on expected length (common variations)
        if len(step_return) == 4:  # Older Gym: obs, reward, done, info
            obs, reward, done, info = step_return
            truncated = False  # Assume not truncated in older API
        elif (
            len(step_return) == 5
        ):  # Newer Gym: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = step_return
            done = terminated or truncated  # Combine termination conditions
        else:
            raise ValueError(
                f"Unexpected number of return values from env.step: {len(step_return)}"
            )

        # Apply dropout mask based on the mask shape
        p = np.random.uniform(size=obs.shape)
        # Ensure dropout_mask aligns with obs dimensions if obs is multi-dimensional
        mask_indices = np.where(self.dropout_mask == 1)[
            0
        ]  # Get indices where dropout is allowed

        # Only apply dropout to the specified dimensions
        obs_copy = obs.copy()  # Work on a copy
        if mask_indices.size > 0:
            p_subset = p[mask_indices]
            obs_copy[mask_indices] = np.where(
                p_subset < self.dropout_rate, 0, obs_copy[mask_indices]
            )

        # Return using the newer 5-tuple format for consistency downstream
        return obs_copy, reward, terminated, truncated, info


class MLPPolicy(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output logits
        return x


class SafetyPolicy(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        # Input size is observation size + action logits size
        self.fc1 = nn.Linear(input_size + action_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 1)
        # Sigmoid activation is applied here because we use BCELoss later
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)  # Output probability (0 to 1)
        return x


def train_ppo_expert(
    env_id="LunarLander-v2", total_timesteps=200_000, load_if_exists=True
):
    save_path = "ppo_expert.zip"
    if load_if_exists and os.path.exists(save_path):
        print(f"Loading existing expert model from {save_path}")
        return PPO.load(save_path)
    print("Training new PPO Expert...")
    # Ensure DummyVecEnv uses the correct env creation lambda
    env = DummyVecEnv([lambda: gym.make(env_id)])
    model = PPO("MlpPolicy", env, verbose=1, device=DEVICE)  # Use DEVICE
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Expert model saved to {save_path}")
    return model


def evaluate_policy(
    env,
    policy_fn,  # Can be SB3 model or PyTorch nn.Module
    episodes=10,
    max_episode_length=EPISODE_LENGTH,  # Use consistent length
    silent=False,
    visualize=False,
    n_episodes_render=3,  # Renamed for clarity
    gif_filename="policy_eval.gif",
):
    returns = []
    failures = []  # Failure defined as return < -100 for LunarLander
    frames = []
    is_sb3_model = isinstance(policy_fn, PPO)

    for i in range(episodes):
        # Seed environment for potentially more reproducible evaluation
        # Note: Seeding VecEnvs or wrapped envs might need careful handling
        # For a single gym env:
        # obs, info = env.reset(seed=random.randint(0, 1_000_000))
        obs, info = env.reset()  # Simpler reset for broad compatibility

        done = False
        truncated = False  # Initialize truncated flag
        total_reward = 0
        timesteps = 0
        episode_frames = []

        while not (done or truncated) and timesteps < max_episode_length:
            if is_sb3_model:
                action, _ = policy_fn.predict(
                    obs, deterministic=True
                )  # Use deterministic actions for eval
            else:  # PyTorch model
                # Ensure policy is in eval mode (affects dropout, batchnorm)
                policy_fn.eval()
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                )
                with torch.no_grad():
                    action_logits = policy_fn(obs_tensor)
                    # Get the action with the highest logit
                    action = action_logits.squeeze(0).argmax().item()

            # Take step using the 5-tuple return format
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated  # Update done based on terminated flag

            total_reward += reward
            timesteps += 1

            if visualize and i < n_episodes_render:
                # Check if render_mode='rgb_array' was set during env creation
                try:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)
                    else:
                        if (
                            i == 0 and timesteps == 1
                        ):  # Print warning only once per eval
                            print(
                                "Warning: env.render() returned None. Cannot create GIF."
                            )
                        visualize = False  # Disable visualization if rendering fails
                except Exception as e:
                    if i == 0 and timesteps == 1:
                        print(
                            f"Warning: env.render() failed with error: {e}. Cannot create GIF."
                        )
                    visualize = False  # Disable visualization if rendering fails

        returns.append(total_reward)
        # Define failure condition (e.g., for LunarLander)
        failures.append(1.0 if total_reward < -100 else 0.0)

        if not silent:
            print(
                f"Evaluation Episode {i + 1}/{episodes} | Return: {total_reward:.2f} | Steps: {timesteps}"
            )

        if visualize and i < n_episodes_render and episode_frames:
            frames.extend(episode_frames)  # Add frames for this episode

    # Save GIF after collecting frames from relevant episodes
    if visualize and frames:
        print(f"Saving evaluation GIF to {gif_filename}...")
        # Ensure frames are in uint8 format for imageio
        frames_uint8 = [f.astype(np.uint8) for f in frames]
        imageio.mimsave(gif_filename, frames_uint8, fps=30)
        print("GIF saved.")

    return returns, failures


# --- CORRECTED train_imitation function ---
def train_imitation(env, expert, epochs=TOTAL_EPOCHS, use_safedagger=False):
    obs_space_shape = env.observation_space.shape[0]
    act_space_n = env.action_space.n

    policy = MLPPolicy(obs_space_shape, act_space_n).to(DEVICE)
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    safety_policy = None
    safety_optimizer = None
    safety_dataset = None
    if use_safedagger:
        safety_policy = SafetyPolicy(obs_space_shape, act_space_n).to(DEVICE)
        safety_optimizer = optim.Adam(safety_policy.parameters(), lr=1e-4)
        # Use a larger safety dataset if memory allows, helps balance estimation
        safety_dataset = deque(maxlen=20000)

    # Main imitation dataset
    dataset = deque(maxlen=20000)

    # History tracking
    loss_history = []
    return_history = []
    query_history = []
    query_ratio_history = []
    failure_history = []
    deviation_history = []  # Average KL divergence per epoch
    time_history = []
    safety_loss_history = []  # Track safety loss if using SafeDAgger
    convergence_epoch = TOTAL_EPOCHS

    print("Seeding initial dataset with expert actions...")
    obs, _ = env.reset()
    done = False
    initial_seed_steps = 0
    # Seed with more steps initially if needed
    while initial_seed_steps < 1000:  # Use step count instead of fixed loop
        action, _ = expert.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Add expert demonstration to main dataset
        dataset.append((obs.copy(), action))  # Store copy of obs

        # Add initial "safe" data for safety policy pre-training
        if use_safedagger:
            with torch.no_grad():
                policy.eval()  # Ensure policy is in eval mode
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                )
                # Get policy's initial action logits for this state
                action_logits = policy(obs_tensor).squeeze(0).cpu()  # Store on CPU
                # Initially, label expert actions as safe (KL div = 0 implicitly)
                # Or calculate KL=0 explicitly - let's assume safe initially for pretrain
                is_safe = 1.0
                safety_dataset.append((obs.copy(), action_logits.clone(), is_safe))

        obs = next_obs
        initial_seed_steps += 1
        if done:
            obs, _ = env.reset()
            done = False  # Reset done flag
    print(f"Initial dataset seeded with {len(dataset)} interactions.")
    if use_safedagger:
        print(f"Initial safety dataset size: {len(safety_dataset)}")

    # Pre-train safety policy if using SafeDAgger
    if use_safedagger and len(safety_dataset) > BATCH_SIZE:
        print("Pre-training safety policy...")
        safety_policy.train()  # Set safety policy to train mode
        for _ in range(100):  # Number of pre-training steps
            # Sample batch from safety dataset
            batch_indices = np.random.choice(
                len(safety_dataset), min(BATCH_SIZE, len(safety_dataset)), replace=False
            )
            batch = [safety_dataset[i] for i in batch_indices]
            obs_batch_s = torch.tensor(
                np.array([item[0] for item in batch]), dtype=torch.float32
            ).to(DEVICE)
            logits_batch_s = torch.stack([item[1] for item in batch]).to(
                DEVICE
            )  # Stack logits tensors
            labels_batch_s = torch.tensor(
                [item[2] for item in batch], dtype=torch.float32
            ).to(DEVICE)

            # Prepare input for safety policy: state + action_logits
            safety_inputs = torch.cat([obs_batch_s, logits_batch_s], dim=1)
            safety_pred = safety_policy(safety_inputs).squeeze(
                -1
            )  # Remove last dim if necessary

            # --- Calculate Balanced Weights for BCELoss ---
            num_total = len(labels_batch_s)
            num_positive = torch.sum(labels_batch_s == 1.0).item()
            num_negative = num_total - num_positive

            weights = torch.ones_like(labels_batch_s)  # Default weights
            if num_positive > 0 and num_negative > 0:
                # Weight = total / (2 * count_of_class) - balances importance
                weight_for_positive = num_total / (2.0 * num_positive)
                weight_for_negative = num_total / (2.0 * num_negative)
                weights = torch.where(
                    labels_batch_s == 1.0, weight_for_positive, weight_for_negative
                )
            # --- End Weight Calculation ---

            # Apply weighted BCELoss
            bce_loss_fn = nn.BCELoss(reduction="none").to(
                DEVICE
            )  # Calculate element-wise loss first
            unweighted_loss = bce_loss_fn(safety_pred, labels_batch_s)
            weighted_loss = unweighted_loss * weights
            safety_loss = weighted_loss.mean()  # Average the weighted loss

            safety_optimizer.zero_grad()
            safety_loss.backward()
            safety_optimizer.step()
        print(
            f"Safety policy pre-training finished. Last pre-train loss: {safety_loss.item():.4f}"
        )

    # Main training loop
    print("Starting main training loop...")
    for epoch in range(epochs):
        start_time = time.time()
        policy.train()  # Set policy to train mode for dropout etc.
        total_policy_loss_epoch = 0.0
        total_safety_loss_epoch = 0.0
        safety_updates_in_epoch = 0

        # Train main policy using aggregated dataset
        if len(dataset) >= BATCH_SIZE:  # Only train if enough data
            for _ in range(UPDATE_STEPS):
                # Sample batch from main dataset
                batch_indices = np.random.choice(
                    len(dataset), BATCH_SIZE, replace=False
                )
                # Correctly extract obs and actions
                obs_batch = torch.tensor(
                    np.array([dataset[i][0] for i in batch_indices]),
                    dtype=torch.float32,
                ).to(DEVICE)
                act_batch = torch.tensor(
                    np.array([dataset[i][1] for i in batch_indices]), dtype=torch.long
                ).to(
                    DEVICE
                )  # Actions are long type for CrossEntropy

                # Predict actions and calculate policy loss
                pred_action_logits = policy(obs_batch)
                policy_loss = nn.CrossEntropyLoss()(pred_action_logits, act_batch)

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                total_policy_loss_epoch += policy_loss.item()
        else:
            print(
                f"Epoch {epoch+1}: Skipping policy training, dataset size {len(dataset)} < {BATCH_SIZE}"
            )

        avg_policy_loss = (
            total_policy_loss_epoch / UPDATE_STEPS
            if UPDATE_STEPS > 0 and len(dataset) >= BATCH_SIZE
            else 0.0
        )
        loss_history.append(avg_policy_loss)

        # Collect new transitions using the current policy (and potentially expert)
        obs, _ = env.reset()
        done = False
        truncated = False
        timesteps = 0
        epoch_expert_queries = 0
        epoch_total_interactions = 0
        epoch_action_deviations = []

        # Set policy to evaluation mode for deterministic action selection during rollout
        policy.eval()
        if use_safedagger:
            safety_policy.eval()  # Also set safety policy to eval mode

        while not (done or truncated) and timesteps < EPISODE_LENGTH:
            epoch_total_interactions += 1
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action_logits = policy(obs_tensor).squeeze(0)  # Get logits from policy
                policy_probs_tensor = torch.softmax(action_logits, dim=0)
                policy_probs = policy_probs_tensor.cpu().numpy()
                policy_act = policy_probs_tensor.argmax().item()

                expert_act, _ = expert.predict(obs, deterministic=True)

                # Calculate KL divergence for safety check and logging
                expert_probs = np.zeros_like(policy_probs)
                expert_probs[expert_act] = 1.0
                # Add EPS to both for stability
                kl_div = entropy(policy_probs + EPS, expert_probs + EPS)
                epoch_action_deviations.append(kl_div)

                # Determine action to take and add data
                action_to_execute = policy_act  # Default action

                if use_safedagger:
                    # 1) Determine the "true" safety label for this state-action pair
                    # Use the KL divergence criterion with the adjusted threshold
                    is_safe = float(kl_div < KL_THRESHOLD)
                    # Add to safety dataset (state, policy_logits, is_safe_label)
                    # Store logits on CPU to avoid GPU memory buildup in deque
                    safety_dataset.append(
                        (obs.copy(), action_logits.cpu().clone(), is_safe)
                    )

                    # 2) Use the *safety policy* to predict if the *policy's action* is safe
                    # Prepare input: state + policy_logits
                    safety_input = torch.cat(
                        [obs_tensor.squeeze(0), action_logits], dim=0
                    ).unsqueeze(0)
                    safety_score = safety_policy(
                        safety_input
                    ).item()  # Get safety prediction

                    # 3) Decide whether to query the expert based on the safety score
                    if safety_score < SAFETY_THRESHOLD:  # Predicted unsafe
                        action_to_execute = expert_act  # Use expert action
                        epoch_expert_queries += 1
                    # else: action_to_execute remains policy_act (predicted safe)

                # Add the state and the *expert's* action to the main imitation dataset (DAgger principle)
                dataset.append((obs.copy(), expert_act))

                # Execute the chosen action (either policy's or expert's) in the environment
                next_obs, reward, terminated, truncated, info = env.step(
                    action_to_execute
                )
                done = terminated  # Update done status

                obs = next_obs
                timesteps += 1

        # Update safety policy using data collected in the safety_dataset
        avg_safety_loss = 0.0
        if use_safedagger and len(safety_dataset) >= BATCH_SIZE:
            safety_policy.train()  # Set safety policy to train mode for update
            # Perform multiple updates to the safety policy per epoch if desired
            for _ in range(UPDATE_STEPS):  # Can use a different number of updates
                # Sample batch from safety dataset
                # Ensure batch size is not larger than dataset size
                current_batch_size = min(BATCH_SIZE, len(safety_dataset))
                # Use random.sample for efficiency with deque
                batch = random.sample(list(safety_dataset), current_batch_size)
                # batch_indices = np.random.choice(len(safety_dataset), current_batch_size, replace=False)
                # batch = [safety_dataset[i] for i in batch_indices]

                obs_batch_s = torch.tensor(
                    np.array([item[0] for item in batch]), dtype=torch.float32
                ).to(DEVICE)
                logits_batch_s = torch.stack([item[1] for item in batch]).to(DEVICE)
                labels_batch_s = torch.tensor(
                    [item[2] for item in batch], dtype=torch.float32
                ).to(DEVICE)

                # Prepare input for safety policy
                safety_inputs = torch.cat([obs_batch_s, logits_batch_s], dim=1)
                safety_pred = safety_policy(safety_inputs).squeeze(
                    -1
                )  # Ensure shape compatibility

                # --- Calculate Balanced Weights for BCELoss ---
                num_total = len(labels_batch_s)
                num_positive = torch.sum(labels_batch_s == 1.0).item()
                num_negative = num_total - num_positive

                weights = torch.ones_like(labels_batch_s)  # Default weights
                if num_positive > 0 and num_negative > 0:
                    weight_for_positive = num_total / (2.0 * num_positive)
                    weight_for_negative = num_total / (2.0 * num_negative)
                    weights = torch.where(
                        labels_batch_s == 1.0, weight_for_positive, weight_for_negative
                    )
                # --- End Weight Calculation ---

                # Apply weighted BCELoss
                bce_loss_fn = nn.BCELoss(reduction="none").to(DEVICE)
                unweighted_loss = bce_loss_fn(safety_pred, labels_batch_s)
                weighted_loss = unweighted_loss * weights
                safety_loss = weighted_loss.mean()

                safety_optimizer.zero_grad()
                safety_loss.backward()
                safety_optimizer.step()
                total_safety_loss_epoch += safety_loss.item()
                safety_updates_in_epoch += 1

            avg_safety_loss = (
                total_safety_loss_epoch / safety_updates_in_epoch
                if safety_updates_in_epoch > 0
                else 0.0
            )
            safety_loss_history.append(avg_safety_loss)

        # Calculate metrics for the epoch
        query_ratio = (
            epoch_expert_queries / epoch_total_interactions
            if epoch_total_interactions > 0
            else 0.0
        )
        query_ratio_history.append(query_ratio)
        query_history.append(epoch_expert_queries)
        avg_deviation = (
            np.mean(epoch_action_deviations) if epoch_action_deviations else 0.0
        )
        deviation_history.append(avg_deviation)

        # Evaluate current policy performance
        # Use a clean environment instance for evaluation
        eval_env = gym.make(env.env.spec.id)  # Get env id and create new instance
        # We pass the policy nn.Module directly to evaluate_policy
        returns, failures = evaluate_policy(
            eval_env, policy, episodes=EVAL_EPISODES_DURING_TRAINING, silent=True
        )
        eval_env.close()  # Close the eval env

        avg_return = np.mean(returns)
        avg_failure_rate = np.mean(failures)
        return_history.append(avg_return)
        failure_history.append(avg_failure_rate)

        # Check for convergence
        if avg_return >= PERFORMANCE_THRESHOLD and convergence_epoch == TOTAL_EPOCHS:
            convergence_epoch = epoch + 1

        epoch_time = time.time() - start_time
        time_history.append(epoch_time)

        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{epochs} | Policy Loss: {avg_policy_loss:.4f} | "
            f"Avg Return: {avg_return:.2f} | "
            f"Expert Queries: {epoch_expert_queries} ({query_ratio:.2%}) | "
            f"Failure Rate: {avg_failure_rate:.2%} | "
            f"Avg KL Div: {avg_deviation:.4f} | "
            f"{f'Safety Loss: {avg_safety_loss:.4f} | ' if use_safedagger else ''}"
            f"Time: {epoch_time:.2f}s"
        )

    # Prepare final statistics dictionary
    stats = {
        "loss": loss_history,
        "returns": return_history,
        "queries": query_history,
        "query_ratios": query_ratio_history,
        "failures": failure_history,
        "deviations": deviation_history,
        "times": time_history,
        "convergence": convergence_epoch,
    }
    if use_safedagger:
        stats["safety_loss"] = safety_loss_history

    return policy, stats


def plot_extended_comparison(dagger_stats, safedagger_stats):
    num_plots = 7  # Base number of plots
    if "safety_loss" in safedagger_stats:
        num_plots += 1  # Add plot for safety loss

    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
    plot_idx = 0

    # Loss
    axs[plot_idx].plot(
        dagger_stats["loss"], label="DAgger Policy Loss", color="red", alpha=0.8
    )
    axs[plot_idx].plot(
        safedagger_stats["loss"],
        label="SafeDAgger Policy Loss",
        color="orange",
        alpha=0.8,
    )
    if "safety_loss" in safedagger_stats:
        axs[plot_idx].plot(
            safedagger_stats["safety_loss"],
            label="SafeDAgger Safety Loss",
            color="purple",
            linestyle="--",
            alpha=0.6,
        )
    axs[plot_idx].set_ylabel("Loss")
    axs[plot_idx].legend()
    axs[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

    # Returns
    axs[plot_idx].plot(dagger_stats["returns"], label="DAgger Return", color="green")
    axs[plot_idx].plot(
        safedagger_stats["returns"], label="SafeDAgger Return", color="lime"
    )
    # Add performance threshold line
    axs[plot_idx].axhline(
        y=PERFORMANCE_THRESHOLD,
        color="gray",
        linestyle=":",
        label=f"Perf. Threshold ({PERFORMANCE_THRESHOLD})",
    )
    axs[plot_idx].set_ylabel("Avg Return")
    axs[plot_idx].legend()
    axs[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

    # Expert Queries
    axs[plot_idx].plot(
        dagger_stats["queries"], label="DAgger Queries", color="blue"
    )  # Should be 0 for DAgger
    axs[plot_idx].plot(
        safedagger_stats["queries"], label="SafeDAgger Queries", color="cyan"
    )
    axs[plot_idx].set_ylabel("Expert Queries per Epoch")
    axs[plot_idx].legend()
    axs[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

    # Query Ratio
    axs[plot_idx].plot(
        dagger_stats["query_ratios"], label="DAgger Query Ratio", color="purple"
    )  # Should be 0
    axs[plot_idx].plot(
        safedagger_stats["query_ratios"],
        label="SafeDAgger Query Ratio",
        color="magenta",
    )
    axs[plot_idx].set_ylabel("Query Ratio")
    axs[plot_idx].set_ylim(0, 1.05)  # Set Y limit for ratio
    axs[plot_idx].legend()
    axs[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

    # Failure Rate
    axs[plot_idx].plot(
        dagger_stats["failures"], label="DAgger Failure Rate", color="brown"
    )
    axs[plot_idx].plot(
        safedagger_stats["failures"], label="SafeDAgger Failure Rate", color="pink"
    )
    axs[plot_idx].set_ylabel("Failure Rate")
    axs[plot_idx].set_ylim(0, 1.05)  # Set Y limit for rate
    axs[plot_idx].legend()
    axs[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

    # Action Deviation (KL)
    axs[plot_idx].plot(
        dagger_stats["deviations"], label="DAgger Action Deviation (KL)", color="gray"
    )
    axs[plot_idx].plot(
        safedagger_stats["deviations"],
        label="SafeDAgger Action Deviation (KL)",
        color="silver",
    )
    # Add KL threshold line
    axs[plot_idx].axhline(
        y=KL_THRESHOLD,
        color="black",
        linestyle=":",
        label=f"KL Threshold ({KL_THRESHOLD})",
    )
    axs[plot_idx].set_ylabel("Avg Action Deviation (KL)")
    axs[plot_idx].legend()
    axs[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

    # Training Time
    axs[plot_idx].plot(
        np.cumsum(dagger_stats["times"]), label="DAgger Cumulative Time", color="black"
    )
    axs[plot_idx].plot(
        np.cumsum(safedagger_stats["times"]),
        label="SafeDAgger Cumulative Time",
        color="darkgray",
    )
    axs[plot_idx].set_ylabel("Cumulative Time (s)")
    axs[plot_idx].set_xlabel("Epoch")
    axs[plot_idx].legend()
    axs[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

    # Add Safety Loss plot if available
    if "safety_loss" in safedagger_stats:
        axs[plot_idx].plot(
            safedagger_stats["safety_loss"],
            label="SafeDAgger Safety Loss",
            color="indigo",
        )
        axs[plot_idx].set_ylabel("Safety Policy Loss")
        axs[plot_idx].set_xlabel("Epoch")
        axs[plot_idx].legend()
        axs[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    plt.suptitle("DAgger vs SafeDAgger: Extended Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout to prevent title overlap
    plt.savefig("comparison_safedagger_vs_dagger_corrected.png")
    plt.close()
    print("Comparison plot saved as comparison_safedagger_vs_dagger_corrected.png")


def plot_returns(
    expert_returns_eval, dagger_returns_eval, safedagger_returns_eval, num_eval_episodes
):
    plt.figure(figsize=(12, 7))
    # Plot individual episode returns as points with transparency
    x_expert = np.arange(len(expert_returns_eval))
    x_dagger = np.arange(len(dagger_returns_eval)) + len(
        expert_returns_eval
    )  # Offset x-axis
    x_safedagger = (
        np.arange(len(safedagger_returns_eval))
        + len(expert_returns_eval)
        + len(dagger_returns_eval)
    )

    plt.scatter(
        x_expert,
        expert_returns_eval,
        label=f"Expert (Eval, N={len(expert_returns_eval)})",
        color="blue",
        alpha=0.5,
        s=10,
    )
    plt.scatter(
        x_dagger,
        dagger_returns_eval,
        label=f"DAgger (Eval, N={len(dagger_returns_eval)})",
        color="green",
        alpha=0.5,
        s=10,
    )
    plt.scatter(
        x_safedagger,
        safedagger_returns_eval,
        label=f"SafeDAgger (Eval, N={len(safedagger_returns_eval)})",
        color="orange",
        alpha=0.5,
        s=10,
    )

    # Add lines for average returns
    avg_expert = np.mean(expert_returns_eval)
    avg_dagger = np.mean(dagger_returns_eval)
    avg_safedagger = np.mean(safedagger_returns_eval)

    plt.axhline(
        avg_expert, color="blue", linestyle="--", label=f"Avg Expert: {avg_expert:.2f}"
    )
    plt.axhline(
        avg_dagger, color="green", linestyle="--", label=f"Avg DAgger: {avg_dagger:.2f}"
    )
    plt.axhline(
        avg_safedagger,
        color="orange",
        linestyle="--",
        label=f"Avg SafeDAgger: {avg_safedagger:.2f}",
    )

    plt.xlabel("Evaluation Episode Index (Grouped by Policy)")
    plt.ylabel("Return")
    plt.title(f"Final Policy Evaluation Returns (N={num_eval_episodes} episodes each)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.4)
    plt.savefig("final_returns_comparison_corrected.png")
    plt.close()
    print("Final returns plot saved as final_returns_comparison_corrected.png")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Environment Setup ---
    ENV_ID = "LunarLander-v2"
    # For visualization in evaluate_policy
    # Make sure render_mode='rgb_array' is supported and used for GIF creation
    try:
        eval_render_env = gym.make(ENV_ID, render_mode="rgb_array")
        RENDER_MODE_EVAL = "rgb_array"
        print(
            f"Using render_mode='{RENDER_MODE_EVAL}' for GIF generation during evaluation."
        )
    except Exception as e:
        print(
            f"Warning: Could not create environment with render_mode='rgb_array'. GIFs disabled. Error: {e}"
        )
        eval_render_env = gym.make(ENV_ID)  # Fallback without render_mode
        RENDER_MODE_EVAL = None  # Indicate rendering is not available
    eval_render_env.close()  # Close test env

    # Create the environment for training (no render mode needed here)
    env = gym.make(ENV_ID)
    # Wrap the environment for potential dropout (dropout_rate is 0.0 currently)
    wrapped_env = EnvWrapper(env, DROPOUT_RATE, DROPOUT_MASK)

    # --- Expert Training ---
    print("--- Training/Loading PPO Expert ---")
    expert_model = train_ppo_expert(
        env_id=ENV_ID, total_timesteps=200_000, load_if_exists=True
    )  # Load if possible

    # --- DAgger Training ---
    print("\n--- Training DAgger policy ---")
    np.random.seed(42)  # Set seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    dagger_model, dagger_stats = train_imitation(
        wrapped_env, expert_model, epochs=TOTAL_EPOCHS, use_safedagger=False
    )
    torch.save(dagger_model.state_dict(), "dagger_lunarlander_mlp_corrected.pt")
    print("DAgger model saved.")

    # --- SafeDAgger Training ---
    print("\n--- Training SafeDAgger policy ---")
    np.random.seed(42)  # Reset seed for fair comparison
    torch.manual_seed(42)
    random.seed(42)
    safedagger_model, safedagger_stats = train_imitation(
        wrapped_env, expert_model, epochs=TOTAL_EPOCHS, use_safedagger=True
    )
    torch.save(safedagger_model.state_dict(), "safedagger_lunarlander_mlp_corrected.pt")
    print("SafeDAgger model saved.")

    # --- Final Evaluation ---
    print("\n--- Evaluating Final Policies ---")
    N_FINAL_EVAL_EPISODES = 50  # Number of episodes for final evaluation

    # Create envs for evaluation, using render_mode if available
    def make_eval_env():
        # Creates env with render mode if possible, otherwise standard env
        try:
            return gym.make(ENV_ID, render_mode=RENDER_MODE_EVAL)
        except:
            return gym.make(ENV_ID)

    print("Evaluating PPO Expert...")
    eval_env_expert = make_eval_env()
    expert_returns, _ = evaluate_policy(
        eval_env_expert,
        expert_model,
        episodes=N_FINAL_EVAL_EPISODES,
        visualize=(RENDER_MODE_EVAL is not None),  # Only visualize if render mode works
        gif_filename="expert_policy_final_corrected.gif",
        n_episodes_render=3,  # Number of episodes to include in GIF
    )
    eval_env_expert.close()

    print("Evaluating DAgger policy...")
    eval_env_dagger = make_eval_env()
    # Load the trained model state for evaluation
    dagger_model.load_state_dict(
        torch.load("dagger_lunarlander_mlp_corrected.pt", map_location=DEVICE)
    )
    dagger_returns, _ = evaluate_policy(
        eval_env_dagger,
        dagger_model,  # Pass the PyTorch model instance
        episodes=N_FINAL_EVAL_EPISODES,
        visualize=(RENDER_MODE_EVAL is not None),
        gif_filename="dagger_policy_final_corrected.gif",
        n_episodes_render=3,
    )
    eval_env_dagger.close()

    print("Evaluating SafeDAgger policy...")
    eval_env_safedagger = make_eval_env()
    # Load the trained model state for evaluation
    safedagger_model.load_state_dict(
        torch.load("safedagger_lunarlander_mlp_corrected.pt", map_location=DEVICE)
    )
    safedagger_returns, _ = evaluate_policy(
        eval_env_safedagger,
        safedagger_model,  # Pass the PyTorch model instance
        episodes=N_FINAL_EVAL_EPISODES,
        visualize=(RENDER_MODE_EVAL is not None),
        gif_filename="safedagger_policy_final_corrected.gif",
        n_episodes_render=3,
    )
    eval_env_safedagger.close()

    # --- Plotting ---
    print("\n--- Plotting Results ---")
    print("Plotting extended comparison...")
    plot_extended_comparison(dagger_stats, safedagger_stats)

    print("Plotting final returns...")
    plot_returns(
        expert_returns, dagger_returns, safedagger_returns, N_FINAL_EVAL_EPISODES
    )

    # --- Print Convergence ---
    print("\n--- Convergence Summary ---")
    print(
        f"DAgger Convergence Epoch (Return >= {PERFORMANCE_THRESHOLD}): {dagger_stats['convergence']}"
    )
    print(
        f"SafeDAgger Convergence Epoch (Return >= {PERFORMANCE_THRESHOLD}): {safedagger_stats['convergence']}"
    )

    # --- Print Final Average Returns ---
    print("\n--- Final Average Performance ---")
    print(
        f"Expert Avg Return: {np.mean(expert_returns):.2f} +/- {np.std(expert_returns):.2f}"
    )
    print(
        f"DAgger Avg Return: {np.mean(dagger_returns):.2f} +/- {np.std(dagger_returns):.2f}"
    )
    print(
        f"SafeDAgger Avg Return: {np.mean(safedagger_returns):.2f} +/- {np.std(safedagger_returns):.2f}"
    )

    print("\nScript finished.")
