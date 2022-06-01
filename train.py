#!pip install gym
#!pip install matplotlib
#!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#!pip install stable-baselines3[extra]
#!pip install jupyter-tensorboard
#!pip install vizdoom
#!pip freeze > requirements.txt

from vizdoom import *
import random
import time
import numpy as np
from matplotlib import pyplot as plt
from gym import Env
from gym.spaces import Discrete, Box
import cv2 as cv2
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO, DQN
import json

path = "/content/drive/MyDrive/RL/"
scenario = "deathmatch"

model = "DQN"

CHECKPOINT_DIR = path + 'train/'+model+'/' + scenario + "/"
LOG_DIR = path + 'logs/'+model+'/log_' + scenario + "/"

POSSIBLE_ACTIONS = 13

SCALE_HEIGHT = 240
SCALE_WIDTH = 320

CUTOUT_TOP = 70
CUTOUT_BOTTOM = 30
CUTOUT_WIDTH = int(round(SCALE_WIDTH * 0.15625,0)*2)

#for ppo
N_STEPS = 2048

REAL_HEIGHT = 70
REAL_WIDTH = 110

# Create Vizdoom OpenAI Gym Environment
class VizDoomGym(Env): 
    # Function that is called when we start the env
    def __init__(self, render=False, config='/content/drive/MyDrive/RL/github/ViZDoom/scenarios/' + scenario + '.cfg', observe=False): 
        # Inherit from Env
        super().__init__()
        self.KILLCOUNT            = 0 
        self.HEALTH               = 100 
        self.ARMOR                = 0
        self.medkits_taken = 0
        self.armorTaken = 0
        #self.startTime = time.time()
        #self.timeNow = time.time()
        
        # Setup the game 
        self.game = DoomGame()
        self.game.load_config(config)
        self.observe = observe
        # Render frame logic
        if render == False: 
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        # Start the game 
        self.game.init()
        
        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(REAL_HEIGHT,REAL_WIDTH ,1), dtype=np.uint8) 
        self.action_space = Discrete(POSSIBLE_ACTIONS)
        
    def step(self, action):


        actions = np.identity(POSSIBLE_ACTIONS)
        reward = self.game.make_action(actions[action], 3) 
        
        done = self.game.is_episode_finished()
        if done == True:
            #print("over")
            # Open a file with access mode 'a'
            try:
              with open('/content/drive/MyDrive/RL/json/DQN/KILLCOUNT.json', 'a') as fp:
                  fp.write(str(self.KILLCOUNT) + " ")
              with open('/content/drive/MyDrive/RL/json/DQN/HEALTH.json', 'a') as fp:
                  fp.write(str(self.medkits_taken) + " ")
              with open('/content/drive/MyDrive/RL/json/DQN/ARMOR.json', 'a') as fp:
                  fp.write(str(self.armorTaken) + " ")
            except:
              with open('/content/drive/MyDrive/RL/json/DQN/KILLCOUNT.json', 'w') as fp:
                  fp.write(str(self.KILLCOUNT) + " ")
              with open('/content/drive/MyDrive/RL/json/DQN/HEALTH.json', 'w') as fp:
                  fp.write(str(self.medkits_taken) + " ")
              with open('/content/drive/MyDrive/RL/json/DQN/ARMOR.json', 'w') as fp:
                  fp.write(str(self.armorTaken) + " ")
                
        # Get all the other stuff we need to retun 
        if self.game.get_state(): 
            state                = self.grayscale(self.game.get_state().screen_buffer)
            KILLCOUNT            = int(self.game.get_state().game_variables[0])
            HEALTH               = int(self.game.get_state().game_variables[1])
            ARMOR                = int(self.game.get_state().game_variables[2])
            
            health_delta = HEALTH - self.HEALTH 
            self.HEALTH = HEALTH
            killcount_delta = KILLCOUNT - self.KILLCOUNT
            self.KILLCOUNT = KILLCOUNT
            armor_delta = ARMOR - self.ARMOR
            self.ARMOR = ARMOR
            
            if armor_delta <= 0:
                armor_delta = 0
            if health_delta > 0:
                self.medkits_taken += 1
                health_factor = 10
            else:
                health_factor = 5
            if armor_delta > 0:
                self.armorTaken += 1
                
            #reward = reward * 100 + health_delta * health_factor + armor_delta * 50
            
            #print(reward)
            #info = KILLCOUNT, HEALTH, ARMOR, SELECTED_WEAPON, SELECTED_WEAPON_AMMO
            
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
            
            self.medkits_taken = 0
            self.armorTaken = 0
            self.KILLCOUNT            = 0 
            self.HEALTH               = 100 
            self.ARMOR                = 0
        
        info = {"KILLCOUNT":self.KILLCOUNT, "HEALTH":self.HEALTH, "ARMOR":self.ARMOR}
        
        if self.observe == True:
            time.sleep(0.02)
        
        return state, reward, done, info 
    def close(self):
        self.game.close()
    def render(self):
      pass
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation,0,-1), cv2.COLOR_BGR2GRAY)
        #crop = gray[CUTOUT_HEIGHT:HEIGHT, int(CUTOUT_WIDTH/2):WIDTH-int(CUTOUT_WIDTH/2)]
        state = cv2.resize(gray, (SCALE_WIDTH, SCALE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        state = state[ CUTOUT_TOP: SCALE_HEIGHT-CUTOUT_BOTTOM, int(CUTOUT_WIDTH/2):-int(CUTOUT_WIDTH/2) :]#-int(CUTOUT_WIDTH/2)]
        state = cv2.resize(state, (REAL_WIDTH, REAL_HEIGHT), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(state, (REAL_HEIGHT , REAL_WIDTH,1))
        #state = cv2.cvtColor(np.moveaxis(observation, 0,-1), cv2.COLOR_BGR2GRAY)
        return state
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)
        #return (state)

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1, time_step=0):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.time_step = time_step

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.logger.dump(self.n_calls)
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls + self.time_step))
            self.model.save(model_path)
            #self.model.save_replay_buffer("RB_"+model + "_" + scenario)
            policy = self.model.policy
            #policy.save("Policy"+model + "_" + scenario)
            
            policy.save("deathmatch")

callback = TrainAndLoggingCallback(check_freq=N_STEPS*3, save_path=CHECKPOINT_DIR, time_step=0)

CONTINUE = False
try:
    env.close()
    env = VizDoomGym(render=False, observe=False)
except:
    env = VizDoomGym(render=False, observe=False)

if CONTINUE == False:
  pass
  #model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=N_STEPS)
  model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, buffer_size=1000000, optimize_memory_usage=True)
  model.learn(total_timesteps=9000000, callback=callback, reset_num_timesteps=True, tb_log_name="deathmatch_no_shaping")
elif CONTINUE == True:
  try:
    del model
  except:
    pass
  
  #model = PPO('CnnPolicy', env,tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001,n_steps=N_STEPS)
  #model = model.load("/content/drive/MyDrive/RL/train/PPO/health_gathering_rerun3/best_model_1400832")
  #model.set_env(env)
  model = DQN.load("/content/drive/MyDrive/RL/train/PPO/health_gathering_rerun3/best_model_1400832", env=env)#, tensorboard_log = "/content/drive/MyDrive/RL/logs/DQN/log_basic_rerun/DQN_1/events.out.tfevents.1653850864.86b75d03021d.8120.0")
  model.set_env(env)
  model.learn(total_timesteps=9000000, callback=callback, reset_num_timesteps=False)#, tb_log_name=LOG_DIR)