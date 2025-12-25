"""
Deeper experiments focused on INT2 heterogeneity.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
from tqdm import tqdm

from src.algorithms.ippo import ActorCritic
from src.quantization.quantize import quantize_team_heterogeneous


def load_trained_networks(checkpoint_path, env, device='cpu'):
    """Load trained networks from checkpoint."""
    networks = {}
    for agent in env.agents:
        obs_dim = env.observation_space(agent).shape[0]
        act_dim = env.action_space(agent).n
        networks[agent] = ActorCritic(obs_dim, act_dim, hidden_dim=128).to(device)
    
    state = torch.load(checkpoint_path, map_location=device)
    for agent in env.agents:
        networks[agent].load_state_dict(state[agent])
        networks[agent].eval()
    
    return networks


def evaluate_team(networks, env, n_episodes=50, deterministic=True, device='cpu'):
    """Evaluate a team of agents."""
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            actions = {}
            for agent in env.agents:
                obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = networks[agent].get_action(obs_tensor, deterministic=deterministic)
                actions[agent] = action.item()
            
            obs, rewards, terms, truncs, _ = env.step(actions)
            ep_reward += sum(rewards.values())
            done = all(terms.values()) or all(truncs.values())
        
        episode_rewards.append(ep_reward)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "rewards": episode_rewards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='experiments/baseline_checkpoint.pt')
    parser.add_argument('--n-episodes', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)
    env.reset(seed=args.seed)
    
    networks = load_trained_networks(args.checkpoint, env, args.device)
    agents = list(networks.keys())
    
    results = []
    
    # ============================================
    # EXPERIMENT A: How many FP32 agents needed to save INT2 team?
    # ============================================
    print("\n" + "="*60)
    print("EXPERIMENT A: Rescuing INT2 Teams")
    print("="*60)
    
    rescue_configs = [
        ("All INT2 (0 rescuers)", {a: 2 for a in agents}),
        ("1 INT8 + 2 INT2", {agents[0]: 8, agents[1]: 2, agents[2]: 2}),
        ("1 FP32 + 2 INT2", {agents[0]: 32, agents[1]: 2, agents[2]: 2}),
        ("2 INT8 + 1 INT2", {agents[0]: 8, agents[1]: 8, agents[2]: 2}),
        ("2 FP32 + 1 INT2", {agents[0]: 32, agents[1]: 32, agents[2]: 2}),
        ("All INT8 (baseline)", {a: 8 for a in agents}),
    ]
    
    for name, precision in tqdm(rescue_configs, desc="Rescue experiments"):
        q_networks = quantize_team_heterogeneous(networks, precision)
        metrics = evaluate_team(q_networks, env, n_episodes=args.n_episodes, device=args.device)
        results.append({
            "experiment": "rescue",
            "config_name": name,
            "precision": str(precision),
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
        })
        print(f"  {name}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    # ============================================
    # EXPERIMENT B: Which agent position to keep high-precision?
    # ============================================
    print("\n" + "="*60)
    print("EXPERIMENT B: Which Agent to Keep at INT8? (Others INT2)")
    print("="*60)
    
    position_configs = [
        ("agent_0 INT8, others INT2", {agents[0]: 8, agents[1]: 2, agents[2]: 2}),
        ("agent_1 INT8, others INT2", {agents[0]: 2, agents[1]: 8, agents[2]: 2}),
        ("agent_2 INT8, others INT2", {agents[0]: 2, agents[1]: 2, agents[2]: 8}),
    ]
    
    for name, precision in tqdm(position_configs, desc="Position experiments"):
        q_networks = quantize_team_heterogeneous(networks, precision)
        metrics = evaluate_team(q_networks, env, n_episodes=args.n_episodes, device=args.device)
        results.append({
            "experiment": "position",
            "config_name": name,
            "precision": str(precision),
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
        })
        print(f"  {name}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    # ============================================
    # EXPERIMENT C: Precision Gradient (all different)
    # ============================================
    print("\n" + "="*60)
    print("EXPERIMENT C: Precision Gradients")
    print("="*60)
    
    gradient_configs = [
        ("32-8-4", {agents[0]: 32, agents[1]: 8, agents[2]: 4}),
        ("32-4-2", {agents[0]: 32, agents[1]: 4, agents[2]: 2}),
        ("8-4-2", {agents[0]: 8, agents[1]: 4, agents[2]: 2}),
        ("32-2-2", {agents[0]: 32, agents[1]: 2, agents[2]: 2}),
        ("8-2-2", {agents[0]: 8, agents[1]: 2, agents[2]: 2}),
        ("4-2-2", {agents[0]: 4, agents[1]: 2, agents[2]: 2}),
    ]
    
    for name, precision in tqdm(gradient_configs, desc="Gradient experiments"):
        q_networks = quantize_team_heterogeneous(networks, precision)
        metrics = evaluate_team(q_networks, env, n_episodes=args.n_episodes, device=args.device)
        results.append({
            "experiment": "gradient",
            "config_name": name,
            "precision": str(precision),
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
        })
        print(f"  {name}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    # Save and plot
    df = pd.DataFrame(results)
    df.to_csv('experiments/int2_experiments.csv', index=False)
    
    # Plot rescue experiment
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot A: Rescue
    rescue_df = df[df['experiment'] == 'rescue']
    ax = axes[0]
    x = range(len(rescue_df))
    colors = ['red', 'orange', 'orange', 'yellow', 'yellow', 'green']
    ax.bar(x, rescue_df['mean_reward'], yerr=rescue_df['std_reward'], capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(rescue_df['config_name'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('How Many High-Precision Agents\nNeeded to Rescue INT2 Team?')
    ax.axhline(y=-90, color='gray', linestyle='--', label='Random Policy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot B: Position
    position_df = df[df['experiment'] == 'position']
    ax = axes[1]
    x = range(len(position_df))
    ax.bar(x, position_df['mean_reward'], yerr=position_df['std_reward'], capsize=5, color='purple', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['agent_0', 'agent_1', 'agent_2'])
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Which Agent to Keep at INT8?\n(Others at INT2)')
    ax.grid(True, alpha=0.3)
    
    # Plot C: Gradient
    gradient_df = df[df['experiment'] == 'gradient']
    ax = axes[2]
    x = range(len(gradient_df))
    ax.bar(x, gradient_df['mean_reward'], yerr=gradient_df['std_reward'], capsize=5, color='teal', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(gradient_df['config_name'], rotation=45, ha='right')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Mixed Precision Gradients')
    ax.axhline(y=-90, color='gray', linestyle='--', label='Random Policy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/int2_experiments.png', dpi=150)
    plt.close()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(df[['experiment', 'config_name', 'mean_reward', 'std_reward']].to_string(index=False))
    print(f"\nResults saved to experiments/int2_experiments.csv")
    print(f"Plot saved to experiments/int2_experiments.png")
    
    env.close()


if __name__ == '__main__':
    main()