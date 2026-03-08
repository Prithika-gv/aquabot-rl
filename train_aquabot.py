from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from aquabot_env import AquaBotEnv
import os

print("🚤 AquaBot RL Training Starting...")

# Create vectorised environment (4 parallel envs for faster training)
env = make_vec_env(AquaBotEnv, n_envs=4)

# PPO model — small network, runs fast on CPU
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    policy_kwargs=dict(net_arch=[64, 64]),  # small = fast on laptop
    tensorboard_log="./aquabot_logs/"
)

# Save best model automatically
os.makedirs("./models/", exist_ok=True)
eval_env = AquaBotEnv()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./models/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Train — 200k steps takes ~10-15 mins on CPU laptop
model.learn(
    total_timesteps=500_000,
    callback=eval_callback,
    progress_bar=True
)

model.save("./models/aquabot_final")
print("✅ Training complete! Model saved to ./models/aquabot_final")