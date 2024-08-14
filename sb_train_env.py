from stable_baselines3 import PPO, A2C
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback


from uam_env.envs.uam_env import UAMEnv
model_name = "ppo_uam"
env = UAMEnv()
LOAD_MODEL = False
TOTAL_TIMESTEPS = 1250000#100000/2 #
CONTINUE_TRAINING = False
COMPARE_MODELS = False

checkpoint_callback = CheckpointCallback(save_freq=10000, 
                                        save_path='./models/'+model_name+'_4/',
                                        name_prefix=model_name)
check_env(env)

n_steps = 550 * 4
n_epochs = 10
batch_size = 100

if LOAD_MODEL and not CONTINUE_TRAINING:
    model = PPO.load(model_name)    
    model.set_env(env)
    
    # if COMPARE_MODELS:
    #     dumb_model = PPO.load(dumb_model_name)
    #     dumb_model.set_env(env)
    #     print("dumb model loaded")
    
    print("model loaded")
elif LOAD_MODEL and CONTINUE_TRAINING:
    model = PPO.load(model_name)
    model.set_env(env)
    print("model loaded and continuing training")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4,
                callback=checkpoint_callback)
    model.save(model_name)
    print("model saved")
else:
    #check env 
    # check_env(env)
    model = PPO("MultiInputPolicy", 
                env,
                n_epochs=n_epochs,
                ent_coef=0.001,
                seed=1, 
                verbose=1, tensorboard_log='tensorboard_logs/', 
                device='cuda')
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4, 
                callback=checkpoint_callback)
    model.save(model_name)
    print("model saved")