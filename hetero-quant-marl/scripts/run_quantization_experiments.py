"""
Run heterogeneous quantization experiments on trained IPPO agents.
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
from src.quantization.quantize import (
    quantize_model, 
    quantize_team_heterogeneous,
    count_unique_weights
)


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
    """
    Evaluate a team of agents.
    
    Returns dict with mean_reward, std_reward, and per-episode rewards.
    """
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
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "rewards": episode_rewards,
    }


def run_experiments(networks, env, device='cpu', n_episodes=100):
    """Run all quantization experiments."""
    
    results = []
    agents = list(networks.keys())
    
    # ============================================
    # EXPERIMENT 1: Uniform Quantization Baselines
    # ============================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Uniform Quantization")
    print("="*60)
    
    uniform_configs = [
        ("FP32 (baseline)", {a: 32 for a in agents}),
        ("INT8 uniform", {a: 8 for a in agents}),
        ("INT4 uniform", {a: 4 for a in agents}),
        ("INT2 uniform", {a: 2 for a in agents}),
    ]
    
    for name, precision in tqdm(uniform_configs, desc="Uniform quantization"):
        q_networks = quantize_team_heterogeneous(networks, precision)
        metrics = evaluate_team(q_networks, env, n_episodes=n_episodes, device=device)
        
        # Count unique weights as sanity check
        unique_weights = {a: count_unique_weights(q_networks[a]) for a in agents}
        
        results.append({
            "experiment": "uniform",
            "config_name": name,
            "precision": str(precision),
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
            "unique_weights_agent0": unique_weights[agents[0]],
        })
        print(f"  {name}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    # ============================================
    # EXPERIMENT 2: Heterogeneous - Vary Fraction at INT4
    # ============================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Heterogeneous - Fraction at INT4")
    print("="*60)
    
    fraction_configs = [
        ("0/3 INT4 (all INT8)", {agents[0]: 8, agents[1]: 8, agents[2]: 8}),
        ("1/3 INT4", {agents[0]: 8, agents[1]: 8, agents[2]: 4}),
        ("2/3 INT4", {agents[0]: 8, agents[1]: 4, agents[2]: 4}),
        ("3/3 INT4 (all INT4)", {agents[0]: 4, agents[1]: 4, agents[2]: 4}),
    ]
    
    for name, precision in tqdm(fraction_configs, desc="Fraction experiments"):
        q_networks = quantize_team_heterogeneous(networks, precision)
        metrics = evaluate_team(q_networks, env, n_episodes=n_episodes, device=device)
        
        results.append({
            "experiment": "fraction",
            "config_name": name,
            "precision": str(precision),
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
        })
        print(f"  {name}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    # ============================================
    # EXPERIMENT 3: Which Agent to Quantize?
    # ============================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Which Agent to Quantize to INT4?")
    print("="*60)
    
    which_agent_configs = [
        ("agent_0 INT4", {agents[0]: 4, agents[1]: 8, agents[2]: 8}),
        ("agent_1 INT4", {agents[0]: 8, agents[1]: 4, agents[2]: 8}),
        ("agent_2 INT4", {agents[0]: 8, agents[1]: 8, agents[2]: 4}),
    ]
    
    for name, precision in tqdm(which_agent_configs, desc="Which agent"):
        q_networks = quantize_team_heterogeneous(networks, precision)
        metrics = evaluate_team(q_networks, env, n_episodes=n_episodes, device=device)
        
        results.append({
            "experiment": "which_agent",
            "config_name": name,
            "precision": str(precision),
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
        })
        print(f"  {name}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    # ============================================
    # EXPERIMENT 4: Extreme Heterogeneity
    # ============================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: Extreme Heterogeneity")
    print("="*60)
    
    extreme_configs = [
        ("1 FP32, 2 INT4", {agents[0]: 32, agents[1]: 4, agents[2]: 4}),
        ("1 FP32, 2 INT2", {agents[0]: 32, agents[1]: 2, agents[2]: 2}),
        ("Mixed 32-8-4", {agents[0]: 32, agents[1]: 8, agents[2]: 4}),
        ("Mixed 8-4-2", {agents[0]: 8, agents[1]: 4, agents[2]: 2}),
    ]
    
    for name, precision in tqdm(extreme_configs, desc="Extreme heterogeneity"):
        q_networks = quantize_team_heterogeneous(networks, precision)
        metrics = evaluate_team(q_networks, env, n_episodes=n_episodes, device=device)
        
        results.append({
            "experiment": "extreme",
            "config_name": name,
            "precision": str(precision),
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
        })
        print(f"  {name}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    return pd.DataFrame(results)


def plot_results(df, save_dir='experiments'):
    """Generate plots from results."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Uniform quantization comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    uniform_df = df[df['experiment'] == 'uniform']
    x = range(len(uniform_df))
    ax.bar(x, uniform_df['mean_reward'], yerr=uniform_df['std_reward'], capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(uniform_df['config_name'], rotation=15)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Uniform Quantization: All Agents Same Precision')
    ax.axhline(y=uniform_df[uniform_df['config_name'] == 'FP32 (baseline)']['mean_reward'].values[0], 
               color='r', linestyle='--', label='FP32 Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/uniform_quantization.png', dpi=150)
    plt.close()
    
    # Plot 2: Fraction at INT4
    fig, ax = plt.subplots(figsize=(10, 6))
    fraction_df = df[df['experiment'] == 'fraction']
    x = range(len(fraction_df))
    ax.bar(x, fraction_df['mean_reward'], yerr=fraction_df['std_reward'], capsize=5, alpha=0.7, color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(fraction_df['config_name'], rotation=15)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Heterogeneous Quantization: Varying Fraction at INT4')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fraction_int4.png', dpi=150)
    plt.close()
    
    # Plot 3: Which agent matters
    fig, ax = plt.subplots(figsize=(10, 6))
    which_df = df[df['experiment'] == 'which_agent']
    x = range(len(which_df))
    ax.bar(x, which_df['mean_reward'], yerr=which_df['std_reward'], capsize=5, alpha=0.7, color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(which_df['config_name'])
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Which Agent to Quantize? (Others at INT8)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/which_agent.png', dpi=150)
    plt.close()
    
    # Plot 4: All configs comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(df))
    colors = {'uniform': 'blue', 'fraction': 'green', 'which_agent': 'orange', 'extreme': 'red'}
    bar_colors = [colors[exp] for exp in df['experiment']]
    ax.bar(x, df['mean_reward'], yerr=df['std_reward'], capsize=3, alpha=0.7, color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(df['config_name'], rotation=45, ha='right')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('All Quantization Configurations')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/all_configs.png', dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='experiments/baseline_checkpoint.pt')
    parser.add_argument('--n-episodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='experiments')
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)
    env.reset(seed=args.seed)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    networks = load_trained_networks(args.checkpoint, env, args.device)
    
    # Verify baseline works
    print("\nVerifying loaded checkpoint...")
    baseline_metrics = evaluate_team(networks, env, n_episodes=20, device=args.device)
    print(f"Baseline (FP32): {baseline_metrics['mean_reward']:.2f} ± {baseline_metrics['std_reward']:.2f}")
    
    # Run experiments
    print(f"\nRunning quantization experiments ({args.n_episodes} episodes each)...")
    results_df = run_experiments(networks, env, device=args.device, n_episodes=args.n_episodes)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_df.to_csv(f'{args.output_dir}/quantization_results.csv', index=False)
    print(f"\nResults saved to {args.output_dir}/quantization_results.csv")
    
    # Generate plots
    plot_results(results_df, args.output_dir)
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(results_df[['config_name', 'mean_reward', 'std_reward']].to_string(index=False))
    
    env.close()


if __name__ == '__main__':
    main()