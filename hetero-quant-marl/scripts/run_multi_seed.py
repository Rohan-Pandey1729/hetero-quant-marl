"""
Run key experiments across multiple seeds for statistical significance.
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


def evaluate_team(networks, env, n_episodes=100, deterministic=True, device='cpu'):
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
    
    return np.mean(episode_rewards), np.std(episode_rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='experiments/baseline_checkpoint.pt')
    parser.add_argument('--n-episodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    # Key configurations to test
    configs = {
        "FP32 (baseline)": {"agent_0": 32, "agent_1": 32, "agent_2": 32},
        "INT8 uniform": {"agent_0": 8, "agent_1": 8, "agent_2": 8},
        "INT4 uniform": {"agent_0": 4, "agent_1": 4, "agent_2": 4},
        "INT2 uniform": {"agent_0": 2, "agent_1": 2, "agent_2": 2},
        "1 INT8 + 2 INT2": {"agent_0": 8, "agent_1": 2, "agent_2": 2},
        "2 INT8 + 1 INT2": {"agent_0": 8, "agent_1": 8, "agent_2": 2},
        "agent_0 INT8 (others INT2)": {"agent_0": 8, "agent_1": 2, "agent_2": 2},
        "agent_1 INT8 (others INT2)": {"agent_0": 2, "agent_1": 8, "agent_2": 2},
        "agent_2 INT8 (others INT2)": {"agent_0": 2, "agent_1": 2, "agent_2": 8},
    }
    
    seeds = [42, 123, 456]
    results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print('='*60)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)
        env.reset(seed=seed)
        
        networks = load_trained_networks(args.checkpoint, env, args.device)
        
        for config_name, precision in tqdm(configs.items(), desc=f"Seed {seed}"):
            q_networks = quantize_team_heterogeneous(networks, precision)
            mean_r, std_r = evaluate_team(q_networks, env, n_episodes=args.n_episodes, device=args.device)
            
            results.append({
                "seed": seed,
                "config": config_name,
                "mean_reward": mean_r,
                "std_reward": std_r,
            })
            print(f"  {config_name}: {mean_r:.2f} ± {std_r:.2f}")
        
        env.close()
    
    # Aggregate across seeds
    df = pd.DataFrame(results)
    
    agg = df.groupby('config').agg({
        'mean_reward': ['mean', 'std'],
    }).round(2)
    agg.columns = ['mean_across_seeds', 'std_across_seeds']
    agg = agg.reset_index()
    
    print("\n" + "="*60)
    print("AGGREGATED RESULTS (across 3 seeds)")
    print("="*60)
    print(agg.to_string(index=False))
    
    # Save
    df.to_csv('experiments/multi_seed_results.csv', index=False)
    agg.to_csv('experiments/multi_seed_aggregated.csv', index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Order for plotting
    order = [
        "FP32 (baseline)", "INT8 uniform", "INT4 uniform", "INT2 uniform",
        "1 INT8 + 2 INT2", "2 INT8 + 1 INT2",
        "agent_0 INT8 (others INT2)", "agent_1 INT8 (others INT2)", "agent_2 INT8 (others INT2)"
    ]
    agg_ordered = agg.set_index('config').loc[order].reset_index()
    
    x = range(len(agg_ordered))
    colors = ['blue', 'blue', 'blue', 'red', 'orange', 'green', 'purple', 'purple', 'purple']
    
    ax.bar(x, agg_ordered['mean_across_seeds'], yerr=agg_ordered['std_across_seeds'], 
           capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(agg_ordered['config'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Heterogeneous Quantization in MARL (3 seeds, 100 episodes each)')
    ax.axhline(y=-90, color='gray', linestyle='--', alpha=0.5, label='Random Policy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/multi_seed_results.png', dpi=150)
    plt.close()
    
    print(f"\nSaved to experiments/multi_seed_results.csv and experiments/multi_seed_results.png")


if __name__ == '__main__':
    main()