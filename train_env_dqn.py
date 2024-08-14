import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from uam_env.envs.uam_env import UAMEnv
from ray.tune.registry import register_env

def env_creator(env_config=None):
    return UAMEnv()  # return an env instance

env_name = "uam_env"
register_env(env_name, env_creator)

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
# Set the working directory in the Ray runtime environment
# ray.init(runtime_env={"working_dir": "."})
ray.init()

algo_name = "DQN"
if __name__ == "__main__":
    # Instantiate the environment to obtain observation and action spaces
    temp_env = UAMEnv()

    # Set the base PPO configuration
    base_config = (
        DQNConfig()
        .environment(env=env_name)
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        .rollouts(num_envs_per_worker=10)  # Number of environments per worker
        .rollouts(num_rollout_workers=3)  # Number of parallel workers
    )

    # Set the policy with observation and action spaces
    base_config = base_config.to_dict()  # Convert PPOConfig to a dictionary
    base_config['multiagent'] = {
        "policies": {
            "default_policy": (None, temp_env.observation_space, temp_env.action_space, {})
        }
    }
    #normalize the observation space
    base_config['observation_filter'] = "MeanStdFilter"

    # Set up the experiment
    cwd = os.getcwd()
    storage_path = os.path.join(cwd, "ray_results", env_name)
    exp_name = "tune_analyzing_results"
    
    tune.run(
        algo_name,
        name=algo_name+"_UAM",
        stop={"timesteps_total": 2500000},
        config=base_config,
        checkpoint_freq=20,
        checkpoint_at_end=True,
        storage_path=storage_path,
        log_to_file=True,
    )
