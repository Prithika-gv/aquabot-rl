from stable_baselines3 import PPO
from aquabot_env import AquaBotEnv
import time

env = AquaBotEnv(render_mode='human')
model = PPO.load("./models/best_model")

obs, _ = env.reset()
total_reward = 0

for step in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    time.sleep(0.05)
    
    if terminated or truncated:
        print(f"Episode done! Collected: {info['collected']} | "
              f"Coverage: {info['coverage']*100:.0f}% | "
              f"Total reward: {total_reward:.1f}")
        break

env.close()