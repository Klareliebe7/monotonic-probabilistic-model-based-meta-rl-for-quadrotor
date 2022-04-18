#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
@Author: Zhengkun Li
@Email: 1st.melchior@gmail.com
'''
import csv
import os,yaml
import numpy as np
from math import floor, ceil
from collections import namedtuple
import gym
import argparse
from random import choice
from quadrotorsim import QuadrotorSim
from plot_utils import plot_learning
from DDPG.ddpg_torch import Agent
from DDPG.utils import plot_learning_curve
from mpc.controllers import MPC
from mpc.models.dynamic_model import DynamicModel
from mpc.models.reward_model import RewardModel

NO_DISPLAY = False
try:
    from render import RenderWindow
except Exception:
    NO_DISPLAY = True


class Quadrotor(gym.Env):
    """
    Quadrotor environment.

    Args:
        mass(float): mass of the quad rotor
        dt (float): duration of single step (in seconds).
        nt (int): number of steps of single episode if no collision
            occurs.
        seed (int): seed to generate target velocity trajectory.
        task (str): name of the task setting. Currently, support
            `no_collision` and `approching`.
        map_file (None|str): path to txt map config file, default
            map is a 100x100 flatten floor.
        simulator_conf (None|str): path to simulator config xml file.
    """

    def __init__(self,
                 mass = 0.5,
                 dt=0.01,
                 nt=1000,
                 seed=0,
                 task='hovering_control',
                 map_file=None,
                 simulator_conf=None,
                 healthy_reward=1.0,
                 aproching_target =None,
                 env_name = 'Quadrotor_sim',
                 **kwargs):
        # TODO: other possible tasks: precision_landing
        assert task in ['approching', 'no_collision',
                        'hovering_control'], 'Invalid task setting'
        if simulator_conf is None:
            simulator_conf = os.path.join(os.path.dirname(__file__),
                                          'config.json')
        assert os.path.exists(simulator_conf), \
            'Simulator config file does not exist'
        print("Building "+env_name+"...")
        self.dt = dt
        self.nt = nt
        self.ct = 0
        self.task = task
        self.healthy_reward = healthy_reward
        self.simulator = QuadrotorSim()

        cfg_dict = self.simulator.get_config(simulator_conf,quality =mass)
        self.valid_range = cfg_dict['range']
        self.action_space = gym.spaces.Box(
            low=np.array([cfg_dict['action_space_low']] * 4, dtype='float32'),
            high=np.array(
                [cfg_dict['action_space_high']] * 4, dtype='float32'),
            shape=[4])

        self.body_velocity_keys = ['b_v_x', 'b_v_y', 'b_v_z']
        self.body_position_keys = ['b_x', 'b_y', 'b_z']
        self.accelerator_keys = ['acc_x', 'acc_y', 'acc_z']
        self.gyroscope_keys = ['gyro_x', 'gyro_y', 'gyro_z']#body_angular_velocity
        self.flight_pose_keys = ['pitch', 'roll', 'yaw']#
        self.barometer_keys = ['x', 'y', 'z']#global
        self.task_approching_keys = \
            ["x_destin","y_destin","z_destin"]

        obs_dim = len(self.body_velocity_keys) + \
            len(self.body_position_keys) + \
            len(self.accelerator_keys) + len(self.gyroscope_keys) + \
            len(self.flight_pose_keys) + len(self.barometer_keys)
        if self.task == 'approching':
            obs_dim += len(self.task_approching_keys)
        self.keys_order = self.body_velocity_keys + self.body_position_keys + \
            self.accelerator_keys + self.gyroscope_keys + \
            self.flight_pose_keys + self.barometer_keys

        if self.task == 'approching':
            self.keys_order.extend(self.task_approching_keys)
        self.observation_space = gym.Space(shape=[obs_dim], dtype='float32')
 
        self.state = {}
        self.viewer = None
        self.x_offset = self.y_offset = self.z_offset = 0
        self.pos_0 = np.array([0.0] * 3).astype(np.float32)

        if self.task == 'approching':
            if aproching_target is None:
                print("The aproching_target arg is needed ")
            self.waypoint_targets = \
                self.simulator.define_approching_task(
                    dt, nt, np.array([0.0,0.0,5.0]),aproching_target)
 
        self.map_matrix = Quadrotor.load_map(map_file)

        # Only for single quadrotor, also mark its start position
        y_offsets, x_offsets = np.where(self.map_matrix == -1)
        assert len(y_offsets) == 1
        self.y_offset = y_offsets[0]
        self.x_offset = x_offsets[0]
        self.z_offset = 5.  # TODO: setup a better init height
        self.map_matrix[self.y_offset, self.x_offset] = 0
        print(f"Mass = { self.simulator._quality} ")
        print(f"Inertia =  { self.simulator._inertia}")    
        print(f"env observastion space len:{obs_dim}")
        print(f"obs order = {self.keys_order}")
    def reset(self):
        self.simulator.reset()
        sensor_dict = self.simulator.get_sensor()   #### for mpc ###TODO
        state_dict = self.simulator.get_state()
        self._update_state(sensor_dict, state_dict)

        # Mark the initial position
        self.pos_0 = np.copy(self.simulator.global_position)

        return self._convert_state_to_ndarray()

    def step(self, action):
        
        self.ct += 1
        cmd = np.asarray(action, np.float32)

        old_pos = [self.simulator.global_position[0] + self.x_offset,
                   self.simulator.global_position[1] + self.y_offset,
                   self.simulator.global_position[2] + self.z_offset]

        self.simulator.step(cmd.tolist(), self.dt)
        sensor_dict = self.simulator.get_sensor()
        state_dict = self.simulator.get_state()
        self._update_state(sensor_dict, state_dict)

        new_pos = [self.simulator.global_position[0] + self.x_offset,
                   self.simulator.global_position[1] + self.y_offset,
                   self.simulator.global_position[2] + self.z_offset]

        if self.task in ['no_collision', 'hovering_control']:
            is_collision = self._check_collision(old_pos, new_pos)
            reward = self._get_reward(collision=is_collision)
            reset = False
            if is_collision:
                reset = True
                self.ct = 0
        elif self.task == 'approching':
            is_collision = self._check_collision(old_pos, new_pos)
            reward = self._get_reward(collision=is_collision)
            reset = False
            if is_collision:
                reset = True
                self.ct = 0
            waypoint_target = self.waypoint_targets[self.ct - 1]
            # body_waypoint_target = np.matmul(
                # self.simulator._coordination_converter_to_body,
                # waypoint_target)
            reward = self._get_reward(waypoint_target=waypoint_target)

        if self.ct == self.nt:
            reset = True
            self.ct = 0

        info = {k: self.state[k] for k in self.state.keys()}
        info['z'] += self.z_offset
        return self._convert_state_to_ndarray(), reward, reset, info

    def render(self, action,mode="human"):
        if(mode != "human"):
            raise NotImplementedError("Only human mode is supported")
        if self.viewer is None:
            if NO_DISPLAY:
                raise RuntimeError('[Error] Cannot connect to display screen.')
            self.viewer = RenderWindow(task=self.task,
                                       x_offset=self.x_offset,
                                       y_offset=self.y_offset,
                                       z_offset=self.z_offset)

        if 'z' not in self.state:
            # It's null state
            raise Exception('You are trying to render before calling reset()')

        state = self._get_state_for_viewer()
        # if self.task == 'approching':
            # self.viewer.view(
                # state, self.dt,
                # expected_velocity=self.velocity_targets[self.ct-1])
        # else:
        self.viewer.view(state, self.dt,action,normal_speed = False)

    def close(self):
        del self.simulator

    def _convert_state_to_ndarray(self):


        ndarray = []
        for k in self.keys_order:
            if k == 'z':
                ndarray.append(self.state[k] + self.z_offset)
            else:
                ndarray.append(self.state[k])

        ndarray = np.array(ndarray, dtype=np.float32)
        
        return ndarray
    def _get_reward_given_obs(self,obs,waypoint_target = None):
        """used in MBPO, negative value, higher means better performance"""
         
        if self.task == 'hovering_control':
            task_reward = 0.0  
            #sensor_dict = env.simulator.get_sensor()
            pitch,roll,yaw =    obs[-6] ,obs[-5],obs[-4] 
            velocity_norm = np.linalg.norm(
                obs[0:3])#rotation wont change the norm of the velocity
            angular_velocity_norm = np.linalg.norm(
                obs[9:12])
            # TODO: you may find better ones!
            alpha, beta = 1.0, 1.0
            hovering_range, in_range_r, out_range_r = 0.5, 10, -20
            """speed and angular_speed reward"""
            task_reward -= alpha * velocity_norm + beta * angular_velocity_norm
            # alpha*speed + beta*angular_speed
            """vertical dist reward"""

            z_move = abs(self.pos_0[2] - obs[-1])#init z - sensor.z ###TODO
 
            task_reward -= 10*(1-z_move**0.5)
            """pitch roll reward"""
            task_reward -= 5 - 5/3.1415*roll
            task_reward -= 5 - 5/3.1415*pitch
            # if z_move < hovering_range:
                # task_reward += in_range_r
            # else:
                # task_reward += max(out_range_r, hovering_range - z_move)

           
        elif self.task == 'approching':
            task_reward = 0.0  
            task_reward -=  self._get_distance(waypoint_target)
             
        return task_reward
    def _get_reward(self, collision=False, waypoint_target=(0.0, 0.0, 5.0)):
        """
        Reward function setting for different tasks. negative value
        """
        # Make sure energy cost always smaller than healthy reward,
        # to encourage longer running
        reward = - min(self.dt * self.simulator.power, self.healthy_reward)
        # if self.task == 'no_collision':
            # task_reward = 0.0 if collision else self.healthy_reward
            # reward += task_reward
        
        if self.task == 'hovering_control':
            task_reward = 0.0 if collision else self.healthy_reward
            #sensor_dict = env.simulator.get_sensor()

            # velocity_norm = np.linalg.norm(
                # self.simulator.global_velocity)
            # angular_velocity_norm = np.linalg.norm(
                # self.simulator.body_angular_velocity)
 
            """speed and angular_speed reward"""
            # alpha, beta = 1.0, 1.0
            # hovering_range, in_range_r, out_range_r = 0.5, 10, -20
            # task_reward -= alpha * velocity_norm + beta * angular_velocity_norm# alpha*speed + beta*angular_speed
            
            """vertical dist reward"""
            z_move = abs(self.pos_0[2] - self.state['z'])#init z - sensor.z
            task_reward -= 10*(1-z_move**0.5)
            """pitch roll reward"""
            #pitch,roll,yaw =    sensor_dict["pitch"] ,sensor_dict["roll"],sensor_dict["yaw"] 
            #task_reward -= 5 - 5/3.1415*roll
            #task_reward -= 5 - 5/3.1415*pitch
            # if z_move < hovering_range:
                # task_reward += in_range_r
            # else:
                # task_reward += max(out_range_r, hovering_range - z_move)

            reward += task_reward
        elif self.task == 'approching':
            task_reward = 0.0 if collision else self.healthy_reward
            task_reward -=  self._get_distance(waypoint_target)
            reward += task_reward
        return reward

    def _check_collision(self, old_pos, new_pos):
        # TODO: update to consider the body size of the quadrotor
        min_max = lambda x, y, i: \
            (int(floor(min(x[i], y[i]))), int(ceil(max(x[i], y[i]))))
        x_min, x_max = min_max(old_pos, new_pos, 0)
        y_min, y_max = min_max(old_pos, new_pos, 1)
        z_min, z_max = min_max(old_pos, new_pos, 2)

        taken_pos = self.map_matrix[y_min:y_max+1, x_min:x_max+1]
        if z_min < np.any(taken_pos) or z_max < np.any(taken_pos):
            return True
        else:
            return False

    def _update_state(self, sensor, state):
        for k, v in sensor.items():
            self.state[k] = v

        for k, v in state.items():
            self.state[k] = v

        if self.task == 'approching':
            t = min(self.ct, self.nt-1)
            next_waypoint = self.waypoint_targets[t]
            self.state['x_destin'] = next_waypoint[0]
            self.state['y_destin'] = next_waypoint[1]
            self.state['z_destin'] = next_waypoint[2]

    def _get_velocity_diff(self, velocity_target):
        vt_x, vt_y, vt_z = velocity_target
        diff = abs(vt_x - self.state['b_v_x']) + \
            abs(vt_y - self.state['b_v_y']) + \
            abs(vt_z - self.state['b_v_z'])
        return diff
    def _get_distance(self, waypoint_target):
        r_x, r_y, r_z = waypoint_target
        diff = (r_x - self.state['x'])**2 + \
            (r_y - self.state['y'])**2 + \
            (r_z - self.state['z'])**2
        return diff**0.5
        
        
    def _get_angel(self ):
        r_x, r_y, r_z = waypoint_target
        diff = (r_x - self.state['x'])**2 + \
            (r_y - self.state['y'])**2 + \
            (r_z - self.state['z'])**2
        return diff**0.5
    def _get_state_for_viewer(self):
        state = {k: v for k, v in self.state.items()}
        state['x'] = self.simulator.global_position[0]
        state['y'] = self.simulator.global_position[1]
        state['z'] = self.simulator.global_position[2]
        state['g_v_x'] = self.simulator.global_velocity[0]
        state['g_v_y'] = self.simulator.global_velocity[1]
        state['g_v_z'] = self.simulator.global_velocity[2]
        return state
    def get_coming_waypoint_targets(self,number_points = None):
        if number_points:
            return self.waypoint_targets[self.ct:self.ct + number_points]
        else:
            return self.waypoint_targets[self.ct]
    @staticmethod
    def load_map(map_file):
        if map_file is None:
            flatten_map = np.zeros([100, 100], dtype=np.int32)
            flatten_map[50, 50] = -1
            return flatten_map

        map_lists = []
        with open(map_file, 'r') as f:
            for line in f.readlines():
                map_lists.append([int(i) for i in line.split(' ')])

        return np.array(map_lists)
 
def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
 
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)
 
 
   
 
'''
if __name__ == '__main__':
    import sys
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_reward_model', type=int, default=0)
    parser.add_argument('--train_reward_model', type=int, default=0)
    parser.add_argument('--render', type=int, default=1)
    args = parser.parse_args()
    config = load_config('./config/config.yml')
    
    """
    Env setting
    """
    print("Setting environment............")
    task = "hovering_control"
    env = Quadrotor(task=task, nt=1000)
    env.reset()
    #env.render(np.array([0,0,0,0]))
    reset = False
    step = 1
    total_reward = 0
    ts = time.time()
    
    """
    Ensemble Dynamic Model setting
    """
    ensemble = False
    nn_config = config['NN_config']
    print(f"Setting dynamic model............ \n with NN config:{nn_config}............")
    
    mpc_config = config['mpc_config']
    reward_config = config['reward_config']
    state_dim = env.observation_space.shape[0]# finished
    action_dim = env.action_space.shape[0]#finished env 73
    action_low, action_high = env.action_space.low, env.action_space.high
    print("obs dim, act dim: ", state_dim, action_dim)
    print("act low high: ", action_low, action_high)
    nn_config["model_config"]["state_dim"] = state_dim
    nn_config["model_config"]["action_dim"] = action_dim
    reward_config["model_config"]["state_dim"] = state_dim
    reward_config["model_config"]["action_dim"] = action_dim
    optimizer_name = mpc_config["optimizer"]
    mpc_config[optimizer_name]["action_low"] = action_low
    mpc_config[optimizer_name]["action_high"] = action_high
    mpc_config[optimizer_name]["action_dim"] = action_dim

    model1 = DynamicModel(NN_config=nn_config,model_name ="model_1")#
    if ensemble:
        model2 = DynamicModel(NN_config=nn_config,model_name ="model_2")
        model3 = DynamicModel(NN_config=nn_config,model_name ="model_3")
        model_list = [model1,model2,model3]
    # if args.use_reward_model:
        # reward_config["model_config"]["load_model"] = True
        # reward_model = RewardModel(reward_config=reward_config)
    # else:
    reward_model = None

    """
    initial MPC controller
    """
    print(f"initiallizing MPC with {optimizer_name} optimizer ...")
    mpc_controller = MPC(mpc_config=mpc_config, reward_model=reward_model)

    if args.train_reward_model:
        reward_model = RewardModel(reward_config=reward_config)
    """
    DDPG setting
    """
    print(f"Setting DDPG............")
    agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=256, fc2_dims=256, 
                    n_actions=env.action_space.shape[0])
    """
    NN pretrain
    """
    # ###TODO: add meta learning here, need to make several different env with different mass and inertia 
    pretrain_episodes = 1
    print(f"NN pretraining with {pretrain_episodes} episodes............")
    if True:
        for epi in range(pretrain_episodes):
            print(f"running pretraining episode {epi}/{pretrain_episodes}...")
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                # action = np.array([5.,5.,5.,5.])
                obs_next, reward, done, state_next = env.step(action)
                if not ensemble:
                    model1.add_data_point([0, obs, action, obs_next - obs])
                else:
                    choice(model_list).add_data_point([0, obs, action, obs_next - obs])
                obs = obs_next    
                env.render(action)
        if not ensemble:
            model1.fit()   
        else:
            model1.fit()  
            model2.fit()  
            model3.fit()  
    else:
   
        model1.load_model(os.getcwd()+"/dynaminc_model_path/model_1/ep35_l_0.00316.pth")   # 500 episodes training
        if ensemble:
            model2.load_model("")  
            model3.load_model("")  
        
    """
    learn policy and dynamic model recursivly
    """  
    if True:
        print(f"learn policy and dynamic model recursivly")
        test_episode = 100
        rollout = 5*10
        sample_ratio = 1/5
        MBPO_ACTIVATE = True
        reward_ave_his,reward_var_his = [],[] 
        #test_epoch = 10
        # for ep in range(test_epoch):
            # print('epoch: ', ep)
        
        for epi in range(test_episode):
            print(f"Running policy learning episode {epi+1}/{test_episode}....")
            obs = env.reset()
            acc_reward, reset = 0, False

            i = 0   
            reward_his = []
            model1.reset_dataset()
            if ensemble:
                model2.reset_dataset()
                model3.reset_dataset()
            while not reset:
                i+= 1
                #action = np.array([mpc_controller.act(model=model, state=obs)])
                action = agent.choose_action(obs)#DDPG
                action =  np.clip((action+np.array([1.1,1.1,1.1,1.1]))*7.45,0.1,15)
                obs_next, reward, reset, _ = env.step(action)
                agent.remember(obs, action, reward, obs_next, reset)#DDPG
                reward_his.append(reward)
 
                if MBPO_ACTIVATE:
                    agent.remember_last_play(obs, action, reward, obs_next, reset,global_state_=env.simulator._save_state())#MBPO
                agent.learn()#DDPG
                
                if not ensemble:
                    model1.add_data_point([0, obs, action, obs_next - obs])
                else:
                    choice(model_list).add_data_point([0, obs, action, obs_next - obs])
                obs = obs_next
                acc_reward += reward    
                
                # if args.render:
                if epi%5 == 0:
                    env.render(action)
                    
                    
                #sensor_dict = env.simulator.get_sensor()
                #state_dict = env.simulator.get_state()
                # print('---------- step %s ----------' % i)
                # print('observastion:', obs_next)
                # print(f"global_stsate:{env.simulator._save_state()}")
                # print(f"sensor_dict:{sensor_dict}")
                # print(f"state_dict:{state_dict}")              
                #print(f'test_episode:{epi},reward:{ reward:.2f}')
            reward_ave_last_epi =np.mean(reward_his, axis=0)
            if (np.array(reward_ave_his)<reward_ave_last_epi).all():

                agent.save_models(name = os.path.join(os.getcwd(),"policy_model_path",  f"ep{epi+1}_reward_{reward_ave_last_epi:.1f}_"))
                
            reward_ave_his.append(reward_ave_last_epi)#plotting
            reward_var_his.append(np.std(reward_his, axis=0))#plotting
            """ MBPO"""
            if MBPO_ACTIVATE:
                # got sample_ratio*i samples from trajectory
                states, actions, rewards, states_, dones,global_states = agent.last_memory.sample_buffer(int(sample_ratio*i),with_global_state = True)
                # for each sample start a simulation
                for ii in range(int(sample_ratio*i)):
                    obs, action, reward, state_, reset,global_state = states[ii], actions[ii], rewards[ii], states_[ii], dones[ii],global_states[ii]
                    if not ensemble:
                        obs_next = model1.predict(obs, action) + obs 
                    else:
                        obs_next1= model1.predict(obs, action) + obs
                        obs_next2 = model2.predict(obs, action) + obs
                        obs_next3 = model3.predict(obs, action) + obs
                        obs_next = (obs_next1 +obs_next2 + obs_next3)/3
                    obs = np.array(obs_next, dtype=np.float32)
                    for j  in range(rollout):
                        action = agent.choose_action(obs)#DDPG
                        action =  np.clip((action+np.array([1.1,1.1,1.1,1.1]))*7.45,0.1,15)
                        reward = env._get_reward_given_obs(obs)
                        if not ensemble:
                            obs_next = model1.predict(obs, action) + obs 
                        else:
                            obs_next1= model1.predict(obs, action) + obs
                            obs_next2 = model2.predict(obs, action) + obs
                            obs_next3 = model3.predict(obs, action) + obs
                            obs_next = (obs_next1 +obs_next2 + obs_next3)/3
                        agent.remember(obs, action, reward, obs_next, reset)#DDPG
                        agent.learn()#DDPG
                        obs = obs_next
                agent.reset_last_play()
            """ MBPO"""        
            
            #env.close()
            print('total reward: ', acc_reward)
            te = time.time()
            print('time cost: ', te - ts)
            model1.fit()  
            if ensemble:
                model2.fit()
                model3.fit()
        x = [i+1 for i in range(len(reward_ave_his))]
        plot_learning(x, reward_ave_his,reward_var_his , "Policy_learning_reward")

    """
    Action planning by mpc
    """  
    if False:
 
        print(f"Using MPC with {optimizer_name} and updating dynamic model in every episode...")
        
        test_epoch = 20
        sample_ratio = 1/5
        test_episode = 10
        # for ep in range(test_epoch):
            # print('epoch: ', ep)
            
        for epi in range(test_episode):
            print(f"Running MPC planning episode {epi+1}/{test_episode}....")
            obs = env.reset()
            acc_reward, reset = 0, False
            mpc_controller.reset()
            i = 0   
            model1.reset_dataset()
            if ensemble:
                model2.reset_dataset()
                model3.reset_dataset()
            while not reset:
                i+= 1
                action = np.array([mpc_controller.act(model=model1, state=obs)])
                #print(f"action={action.reshape(-1)}")
                obs_next, reward, reset, _ = env.step(action.reshape(-1))

                if not ensemble:
                    model1.add_data_point([0, obs, action, obs_next - obs])
                else:
                    choice(model_list).add_data_point([0, obs, action, obs_next - obs])
                obs = obs_next
                acc_reward += reward    
                sensor_dict = env.simulator.get_sensor()
                state_dict = env.simulator.get_state()
                if args.render:
                    env.render(action.reshape(-1))
                # print('---------- step %s ----------' % i)
                # print('observastion:', obs_next)
                # print(f"global_stsate:{env.simulator._save_state()}")
                # print(f"sensor_dict:{sensor_dict}")
                # print(f"state_dict:{state_dict}")              
                #print(f'test_episode:{epi},reward:{ reward:.2f}')
   
            
            #env.close()
            print('total reward: ', acc_reward)
            te = time.time()
            print('time cost: ', te - ts)
            model1.fit()  
            if ensemble:
                model2.fit()
                model3.fit()

'''
