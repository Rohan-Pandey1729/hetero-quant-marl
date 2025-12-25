"""
Train baseline IPPO agents on simple_spread_v3.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3

from src.algorithms.ippo import IPPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=500000)
    parser.add_argument('--rollout-steps', type=int, default=1024)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-path', type=str, default='experiments/baseline_checkpoint.pt')
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)
    env.reset(seed=args.seed)
    
    print(f"Environment: simple_spread_v3")
    print(f"Agents: {env.agents}")
    print(f"Observation space: {env.observation_space('agent_0').shape}")
    print(f"Action space: {env.action_space('agent_0').n}")
    print(f"Device: {args.device}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Learning rate: {args.lr}")
    print(f"Total timesteps: {args.total_timesteps}")
    print("-" * 50)
    
    # Create trainer with better hyperparameters
    trainer = IPPOTrainer(
        env=env,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        device=args.device,
    )
    
    # Train
    rewards = trainer.train(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        log_interval=10,
    )
    
    # Save checkpoint
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    trainer.save(args.save_path)
    
    # Plot training curve
    if len(rewards) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.3, label='Episode Reward')
        
        # Smoothed curve
        window = min(100, len(rewards) // 4) if len(rewards) > 4 else 1
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), smoothed, label=f'Smoothed (window={window})')
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('IPPO Training on simple_spread_v3')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('experiments/training_curve.png', dpi=150)
        plt.close()
        print(f"Saved training curve to experiments/training_curve.png")
    
    # Final evaluation
    print("-" * 50)
    print("Final Evaluation (20 episodes, deterministic):")
    
    eval_rewards = []
    for ep in range(20):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            actions = {}
            for agent in env.agents:
                obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    action = trainer.networks[agent].get_action(obs_tensor, deterministic=True)
                actions[agent] = action.item()
            
            obs, rewards_dict, terms, truncs, _ = env.step(actions)
            ep_reward += sum(rewards_dict.values())
            done = all(terms.values()) or all(truncs.values())
        
        eval_rewards.append(ep_reward)
    
    print(f"Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    
    # Also evaluate stochastic
    print("\nStochastic Evaluation (20 episodes):")
    eval_rewards_stoch = []
    for ep in range(20):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            actions = {}
            for agent in env.agents:
                obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    action = trainer.networks[agent].get_action(obs_tensor, deterministic=False)
                actions[agent] = action.item()
            
            obs, rewards_dict, terms, truncs, _ = env.step(actions)
            ep_reward += sum(rewards_dict.values())
            done = all(terms.values()) or all(truncs.values())
        
        eval_rewards_stoch.append(ep_reward)
    
    print(f"Mean Reward: {np.mean(eval_rewards_stoch):.2f} ± {np.std(eval_rewards_stoch):.2f}")
    
    env.close()


if __name__ == '__main__':
    main()