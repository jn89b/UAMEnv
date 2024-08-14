import os
import ray
import matplotlib.pyplot as plt
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from uam_env.envs.uam_env import UAMEnv
from ray.tune.registry import register_env
from uam_env.visualizer.visualizer import Visualizer


def env_creator(env_config=None):
    return UAMEnv()  # return an env instance

env_name = "uam_env"
register_env(env_name, env_creator)

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
ray.init(runtime_env={"working_dir": "."})
# ray.init()

if __name__ == "__main__":
    # Path to the checkpoint directory  # e.g., "ray_results/PPO_UAM/checkpoint_000050"
    cwd = os.getcwd()
    storage_path = os.path.join(cwd, "ray_results", env_name)
    print(storage_path)
    storage_path = os.path.join(storage_path, "PPO_UAM/PPO_trained_2")

    # Ensure that the checkpoint path is valid
    if not os.path.exists(storage_path):
        raise ValueError(f"Checkpoint path {storage_path} does not exist")


    storage_path = "ray_results/uam_env/PPO_UAM/PPO_trained_2/checkpoint_000005/checkpoint-5"
    storage_path = cwd + "/ray_results/uam_env/PPO_UAM/PPO_trained_2/checkpoint_000005"

    # Instantiate the environment to obtain observation and action spaces
    temp_env = UAMEnv()

    # Load the configuration used during training
    base_config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .rollouts(num_envs_per_worker=2)
        .rollouts(num_rollout_workers=3)
    )
    
    base_config = base_config.to_dict()
    base_config['multiagent'] = {
        "policies": {
            "default_policy": (None, temp_env.observation_space, temp_env.action_space, {})
        }
    }
    base_config['observation_filter'] = "MeanStdFilter"
    # Create the PPO agent
    agent = PPO(config=base_config)
    # Restore the agent from the checkpoint
    agent.restore(storage_path)
    print("Agent restored")
    # Now you can use the agent to run inferences in the environment
    env = UAMEnv()

    # Reset the environment
    obs,info = env.reset()

    # Run the agent in the environment
    num_success = 0
    num_trials = 2
    import time
    for i in range(num_trials):
        obs,info = env.reset()
        done = False
        while not done:
            start_time = time.time()
            action = agent.compute_single_action(obs)
            delta_time = time.time() - start_time
            print("Time taken: ", delta_time)
            obs, reward, done, truncated, info = env.step(action)
            # env.render()  # Render the environment (optional)
            if reward > 0 and done:
                num_success += 1

    print("Success rate: ", num_success/num_trials)

    # # Visualize the environment
    vis = Visualizer()
    fig, ax = vis.show_lanes_3d(
        env.corridors.lane_network, 
        uam_env=env, 
        plot_vehicles=True, zoom_in=False,
        show_crash=False)
    fig, ax, anim = vis.animate_vehicles(uam_env=env, show_crash=False)
    
    plt.legend()
    # anim.save('animation.gif', writer='imagemagick', fps=30)
    plt.show()
    