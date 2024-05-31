import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import config
from pathlib import Path

parser = argparse.ArgumentParser(description="Trains a CARLA agent")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--town", default="Town01", type=str, help="Name of the map in CARLA")
parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timestep to train for")
parser.add_argument("--reload_model", type=str, default="", help="Path to a model to reload")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--num_checkpoints", type=int, default=10, help="Checkpoint frequency")
parser.add_argument("--exp_name", default="pvp_carla", type=str, help="The name for this batch of experiments.")
parser.add_argument("--seed", default=0, type=int, help="The random seed.")
parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")

args = vars(parser.parse_args())
config.set_config('1')

from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from agent.env import CarlaEnv

from utilities.utils import get_time_str
from utilities.wandb_vallback import WandbCallback
from algorithm.td3.pvp_td3 import PVPTD3
from algorithm.sac.our_feature_extractor import OurFeaturesExtractor
from core_rl.rewards import reward_functions
from algorithm.haco import HACOReplayBuffer
from utilities.shared_control_monitor import SharedControlMonitor
from config import CONFIG

other_feat_dim = 1
experiment_batch_name = "{}_{}".format(args["exp_name"], 'birdview')
seed = args["seed"]
experiment_dir = Path("runs") / experiment_batch_name
trial_name = "{}_{}".format(experiment_batch_name, get_time_str())

use_wandb = args["wandb"]
project_name = args["wandb_project"]
team_name = args["wandb_team"]
trial_dir = experiment_dir / trial_name
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(trial_dir, exist_ok=True)
print(f"We start logging training data into {trial_dir}")

BEV_config = dict(
        size=[84, 84],
        pixels_per_meter=2,
        pixels_ahead_vehicle=16,
        cfg_type = "BeVWrapper",
    )

algo_config = dict(use_balance_sample=True,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,  # We run in reward-free manner!
            ),

            # PZH Note: Compared to MetaDrive, we use CNN as the feature extractor.
            # policy_kwargs=dict(net_arch=[256, 256]),
            policy_kwargs=dict(
                features_extractor_class=OurFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=256 + other_feat_dim),
                share_features_extractor=False,  # PZH: Using independent CNNs for actor and critics
                net_arch=[
                    256,
                ]
            ),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,  # We only conduct experiment less than 50K steps
            learning_starts=100,  # The number of steps before
            batch_size=128,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            verbose=2,
            device="auto",
        )

env = CarlaEnv(host=args["host"], port=args["port"], town=args["town"],
                fps=args["fps"], obs_sensor=CONFIG["obs_sensor"], obs_res=CONFIG["obs_res"], 
                    reward_fn=reward_functions[CONFIG["reward_fn"]], bev_config=BEV_config,
                    view_res=(1120, 560), action_smoothing=CONFIG["action_smoothing"],
                    allow_spectator=True, allow_render=args["no_render"])

env = Monitor(env=env, filename=str(trial_dir))
env = SharedControlMonitor(env=env, folder=trial_dir / "data", prefix=trial_name)

algo_config["env"] = env
experiment_dir = Path("runs") / experiment_batch_name
trial_dir = experiment_dir / trial_name

# ===== Setup the callbacks =====
save_freq = 500  # Number of steps per model checkpoint
callbacks = [
    CheckpointCallback(name_prefix="rl_model", verbose=1, save_freq=save_freq, save_path=str(trial_dir / "models"))
]

if use_wandb:
    callbacks.append(
        WandbCallback(
            trial_name=trial_name,
            exp_name=experiment_batch_name,
            team_name=team_name,
            project_name=project_name,
            config=config
        )
    )
callbacks = CallbackList(callbacks)
# ===== Setup the training algorithm =====
model = PVPTD3(**algo_config)
# ===== Launch training =====
model.learn(
    # training
    total_timesteps=50_000,
    callback=callbacks,
    reset_num_timesteps=True,
    # logging
    tb_log_name=experiment_batch_name,
    log_interval=1,
    save_buffer=False,
    load_buffer=False,
)