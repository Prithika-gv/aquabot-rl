from stable_baselines3 import PPO
from aquabot_env import AquaBotEnv
import csv, os

model = PPO.load("./models/best_model")
results = []
print("Running 20 evaluation episodes...\n")

for ep in range(20):
    env = AquaBotEnv()
    obs, _ = env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    results.append({
        'episode': ep + 1,
        'collected': info['collected'],
        'coverage_pct': round(info['coverage'] * 100, 1),
        'total_reward': round(total_reward, 2),
    })
    print(f"Ep {ep+1:2d} | Collected: {info['collected']}/8 | "
          f"Coverage: {info['coverage']*100:.0f}% | "
          f"Reward: {total_reward:.1f}")
    env.close()

os.makedirs('./results', exist_ok=True)
with open('./results/aquabot_eval.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

collected_vals = [r['collected'] for r in results]
coverage_vals  = [r['coverage_pct'] for r in results]
reward_vals    = [r['total_reward'] for r in results]

print(f"\n{'='*45}")
print(f"  AQUABOT EVALUATION SUMMARY (20 episodes)")
print(f"{'='*45}")
print(f"  Avg debris collected : {sum(collected_vals)/len(collected_vals):.2f} / 8")
print(f"  Avg coverage         : {sum(coverage_vals)/len(coverage_vals):.1f}%")
print(f"  Avg reward           : {sum(reward_vals)/len(reward_vals):.2f}")
print(f"  Best episode         : {max(collected_vals)} collected")
print(f"  Results saved to     : ./results/aquabot_eval.csv")
print(f"{'='*45}")