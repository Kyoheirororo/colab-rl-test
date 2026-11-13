"""Minimal RL training for the Colab route test (WK-209).

Trains PPO on CartPole-v1 for a short run and prints an evaluation summary.
Designed to finish in a few minutes on a free Colab runtime.
"""
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0, seed=42)
model.learn(total_timesteps=20_000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"EVAL mean_reward={mean_reward:.1f} std={std_reward:.1f} episodes=20")
model.save("ppo_cartpole")
print("saved: ppo_cartpole.zip")
