import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


from gym.wrappers import FrameStack, GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


import torch


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)

        return True


### Fix error
class PatchedJoypadSpace(JoypadSpace):
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.env.seed(seed)
        return self.env.reset()





def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    print("Action space before joypad: ", env.action_space)
    env = PatchedJoypadSpace(env, SIMPLE_MOVEMENT)
    print("Action space after joypad: ", env.action_space)
    print("observation space: ", env.observation_space.shape)
    print(SIMPLE_MOVEMENT)
    state = env.reset()
    print("stae: ", state.shape) ## state is pixel
    
    # done = True
    # timesteps = 1000000
    # for step in range(timesteps):
    #     if done:
    #         env.reset()
    #     state, reward, done, info = env.step(env.action_space.sample())
    #     env.render()
    # env.close()
    
    #### Preprocess environment
    # grayscale
    env = GrayScaleObservation(env, keep_dim=True) ## maintian keep_dim otherwise changed to 2D
    # wrap inside the dummy environment
    env = DummyVecEnv([lambda: env])
    # stack the frames
    env = VecFrameStack(env, 4, channels_order='last')

    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'
    TENSOR_LOG_DIR = './tensor_logs/'
    
    # callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    callback = EvalCallback(env, best_model_save_path=CHECKPOINT_DIR,
                            log_path=LOG_DIR, eval_freq=500, deterministic=True, render=False)
    
    model = PPO('CnnPolicy', env, verbose=1, learning_rate=0.000001, 
                device=device, n_steps=1024, tensorboard_log=TENSOR_LOG_DIR)

    model.learn(total_timesteps=1000000000, callback=callback)
    







if __name__ == '__main__':
    train()