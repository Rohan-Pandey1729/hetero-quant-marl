from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)
observations, infos = env.reset()

print(f"Agents: {env.agents}")
print(f"Num agents: {len(env.agents)}")
print(f"Observation shape: {env.observation_space('agent_0').shape}")
print(f"Action space: {env.action_space('agent_0').n} discrete actions")

# Quick rollout
total_reward = 0
for step in range(25):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    total_reward += sum(rewards.values())

print(f"Random policy reward: {total_reward:.2f}")
print("SUCCESS: MPE working!")
env.close()