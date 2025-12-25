"""
Independent PPO (IPPO) for Multi-Agent environments.
Each agent has its own actor-critic network trained independently.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    """Simple MLP actor-critic network."""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, act_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Smaller init for policy head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def forward(self, obs):
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()
        
        return action
    
    def evaluate_action(self, obs, action):
        """Get log prob, entropy, and value for given obs-action pair."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """Stores rollout data for PPO update."""
    
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def compute_returns(self, gamma=0.99, gae_lambda=0.95, last_value=0):
        """Compute GAE advantages and returns."""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def get_tensors(self, device='cpu'):
        """Convert to tensors for training."""
        return (
            torch.FloatTensor(np.array(self.obs)).to(device),
            torch.LongTensor(np.array(self.actions)).to(device),
            torch.FloatTensor(np.array(self.log_probs)).to(device),
        )


class IPPOTrainer:
    """Trains independent PPO agents."""
    
    def __init__(
        self,
        env,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        device='cpu',
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Get agent list
        self.agents = env.agents
        self.n_agents = len(self.agents)
        
        # Create one network and optimizer per agent
        self.networks = {}
        self.optimizers = {}
        self.buffers = {}
        
        for agent in self.agents:
            obs_dim = env.observation_space(agent).shape[0]
            act_dim = env.action_space(agent).n
            
            network = ActorCritic(obs_dim, act_dim, hidden_dim).to(device)
            self.networks[agent] = network
            self.optimizers[agent] = optim.Adam(network.parameters(), lr=lr, eps=1e-5)
            self.buffers[agent] = RolloutBuffer()
    
    def collect_rollout(self, n_steps=2048):
        """Collect experience from environment."""
        obs, _ = self.env.reset()
        
        episode_rewards = []
        current_ep_reward = 0
        
        for step in range(n_steps):
            actions = {}
            
            # Get actions from all agents
            for agent in self.agents:
                obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits, value = self.networks[agent](obs_tensor)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                actions[agent] = action.item()
                
                # Store in buffer
                self.buffers[agent].obs.append(obs[agent])
                self.buffers[agent].actions.append(action.item())
                self.buffers[agent].log_probs.append(log_prob.item())
                self.buffers[agent].values.append(value.item())
            
            # Step environment
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Store rewards and dones
            done = all(terminations.values()) or all(truncations.values())
            for agent in self.agents:
                self.buffers[agent].rewards.append(rewards[agent])
                self.buffers[agent].dones.append(float(done))
            
            current_ep_reward += sum(rewards.values())
            
            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0
                obs, _ = self.env.reset()
            else:
                obs = next_obs
        
        # Compute last values for GAE
        last_values = {}
        for agent in self.agents:
            obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.networks[agent](obs_tensor)
            last_values[agent] = value.item()
        
        return episode_rewards, last_values
    
    def update(self, last_values):
        """PPO update for all agents."""
        total_losses = {agent: 0 for agent in self.agents}
        
        for agent in self.agents:
            buffer = self.buffers[agent]
            
            # Compute advantages and returns
            advantages, returns = buffer.compute_returns(
                self.gamma, self.gae_lambda, last_values[agent]
            )
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Get tensors
            obs, actions, old_log_probs = buffer.get_tensors(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # PPO epochs
            n_samples = len(obs)
            indices = np.arange(n_samples)
            
            for epoch in range(self.n_epochs):
                np.random.shuffle(indices)
                
                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    batch_obs = obs[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    
                    # Evaluate actions
                    log_probs, entropy, values = self.networks[agent].evaluate_action(
                        batch_obs, batch_actions
                    )
                    
                    # Policy loss (clipped surrogate)
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = nn.functional.mse_loss(values, batch_returns)
                    
                    # Entropy bonus
                    entropy_loss = -entropy.mean()
                    
                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    
                    # Update
                    self.optimizers[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.networks[agent].parameters(), self.max_grad_norm)
                    self.optimizers[agent].step()
                    
                    total_losses[agent] += loss.item()
            
            # Clear buffer
            buffer.clear()
        
        return total_losses
    
    def train(self, total_timesteps=100000, rollout_steps=2048, log_interval=1):
        """Main training loop."""
        n_updates = total_timesteps // rollout_steps
        all_rewards = []
        best_mean_reward = -float('inf')
        
        for update in range(n_updates):
            # Collect rollout
            episode_rewards, last_values = self.collect_rollout(rollout_steps)
            all_rewards.extend(episode_rewards)
            
            # Update networks
            losses = self.update(last_values)
            
            # Logging
            if update % log_interval == 0 and len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                recent_mean = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
                
                if recent_mean > best_mean_reward:
                    best_mean_reward = recent_mean
                
                print(f"Update {update}/{n_updates} | "
                      f"Episodes: {len(episode_rewards)} | "
                      f"Mean Reward: {mean_reward:.2f} | "
                      f"Recent 100: {recent_mean:.2f} | "
                      f"Best: {best_mean_reward:.2f}")
        
        return all_rewards
    
    def save(self, path):
        """Save all agent networks."""
        state = {agent: self.networks[agent].state_dict() for agent in self.agents}
        torch.save(state, path)
        print(f"Saved checkpoint to {path}")
    
    def load(self, path):
        """Load all agent networks."""
        state = torch.load(path, map_location=self.device)
        for agent in self.agents:
            self.networks[agent].load_state_dict(state[agent])
        print(f"Loaded checkpoint from {path}")