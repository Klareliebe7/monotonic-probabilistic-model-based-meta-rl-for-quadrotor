'''
@Author: Zhengkun Li
@Email: 1st.melchior@gmail.com
'''

import numpy as np
import copy

from mpc.optimizers import RandomOptimizer, CEMOptimizer


class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, mpc_config, reward_model = None):
        # mpc_config = config["mpc_config"]
        self.type = mpc_config["optimizer"]
        conf = mpc_config[self.type]
        self.horizon = conf["horizon"]
        self.gamma = conf["gamma"]
        self.action_low = np.array(conf["action_low"]) # array (dim,)
        self.action_high = np.array(conf["action_high"]) # array (dim,)
        self.action_dim = conf["action_dim"]
        self.popsize = conf["popsize"]

        self.particle = conf["particle"]
        self.mission = 'hovering_control'
        self.init_mean = np.array([conf["init_mean"]] * self.horizon)
        self.init_var = np.array([conf["init_var"]] * self.horizon)

        if len(self.action_low) == 1: # auto fill in other dims
            self.action_low = np.tile(self.action_low, [self.action_dim])
            self.action_high = np.tile(self.action_high, [self.action_dim])
        #print(f"action_low:{self.action_low} TESTESTESTESTESET")
        self.optimizer = MPC.optimizers[self.type](sol_dim=self.horizon*self.action_dim,popsize=self.popsize,upper_bound=np.array(conf["action_high"]),lower_bound=np.array(conf["action_low"]),max_iters=conf["max_iters"],num_elites=conf["num_elites"],epsilon=conf["epsilon"],alpha=conf["alpha"])
        #print(f"========upper_bound======== {np.array(conf['action_high'])}")
     
        if reward_model is not None:
            self.reward_model = reward_model
            self.optimizer.setup(self.cost_function)
        else:
            self.optimizer.setup(self.quadrotor_cost_function) # default cost function ###TODO 这里作修改
        self.reset()

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        ##print('set init mean to 0')
        self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [1])
 
        self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [1])

    def act(self, model, state,waypoints_horizon = None):
        '''
        :param state: model, (numpy array) current state
        :return: (float) optimal action
        '''
        self.model = model
        self.state = state

        soln, var = self.optimizer.obtain_solution(self.prev_sol, self.init_var,waypoints_horizon=waypoints_horizon)
        #print(f"slon = {soln.shape}")
        #print(f"slon[self.action_dim:] = {soln[self.action_dim:,:].shape}")
        #print(f"np.zeros(self.action_dim) = {np.zeros((self.action_dim,4)).shape}")
        if self.type == "CEM":
            self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:,:], np.zeros((self.action_dim,4))])
        else:
            pass
        action = soln[0,:] 
        #print(f"acrion = {action.shape}")       
        return np.array(action)

    def cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))

        costs = np.zeros(self.popsize*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize*self.particle, axis=0)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)

            state_next = self.model.predict(state, action) + state

            cost = -self.reward_model.predict(state_next, action)  # compute cost
            cost = cost.reshape(costs.shape)
            costs += cost * self.gamma**t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def quadrotor_cost_function(self, actions, waypoints_target =None):
        """USED IN cem.py line 143 costs = self.cost_function(samples)
        
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        if self.mission == 'hovering_control':
        
            # TODO: may be able to change to tensor like pets
            #print(f'action nefore reshape{actions.shape}')
            #actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
            #print(f'action after reshape{actions.shape}')
            actions = np.tile(actions, (self.particle, 1, 1))
            #print(f'action after tile{actions.shape}')
            costs = np.zeros(self.popsize*self.particle)
            #print(f"cost {costs.shape}")
            state = np.repeat(self.state.reshape(1, -1), self.popsize*self.particle, axis=0)
            #print(f"state {state.shape}")
     
            for t in range(self.horizon):
                action = actions[:, t, :]  # numpy array (batch_size x action dim)
                #print(f'action slice{action.shape}')
                state_next = self.model.predict(state , action,axis_ = 1 ) + state
                #print(f'state_next= {state_next.shape}')
                #cost = self.cartpole_cost(state_next, action)  # compute cost
                cost =self._get_reward_given_obs( state_next,action,waypoint_target = waypoint_target )
                #print(f'cost={cost.shape}')
                costs += cost * self.gamma**t
                state = copy.deepcopy(state_next)

            # average between particles
            costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
            return costs
        elif self.mission == 'approching':
            #print(f'action nefore reshape{actions.shape}')
            #actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
            #print(f'action after reshape{actions.shape}')
            actions = np.tile(actions, (self.particle, 1, 1))
            #print(f'action after tile{actions.shape}')
            costs = np.zeros(self.popsize*self.particle)
            #print(f"cost {costs.shape}")
            state = np.repeat(self.state.reshape(1, -1), self.popsize*self.particle, axis=0)
            #print(f"state {state.shape}")
            
            for t in range(self.horizon):
                #print(np.array(waypoints_target[t]))
                waypoint_target_horizon = np.repeat(np.array(waypoints_target[t]).reshape(1, -1), self.popsize*self.particle, axis=0)
                #print(f"waypoint_target_horizon={waypoint_target_horizon } should be (3*1000)")
                action = actions[:, t, :]  # numpy array (batch_size x action dim)
                #print(f'action slice{action.shape}')
                state_next = self.model.predict(state , action,axis_ = 1 ) + state
                #print(f'state_next= {state_next.shape}')
                #cost = self.cartpole_cost(state_next, action)  # compute cost
                cost =self._get_reward_given_obs( state_next,action,waypoint_target = waypoint_target_horizon )
                #print(f'cost={cost }')
                costs += cost * self.gamma**t
                state = copy.deepcopy(state_next)

            # average between particles
            costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
            return costs
    '''
    def quadrotor_cost(self, state, action, target_point,env_cost=False  ):
        """
        Calculate the quadrotor env cost given the state

        Parameters:
        ----------
            @param numpy array - state : size should be (batch_size x state dim)
            @param numpy array - action : size should be (batch_size x action dim)
            @param numpy array - target_point : size should be (3)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
 
        target_point = np.tile(target_point,(len(state[:,0]),1))#make target_point size of batch_size*3 ###TODO 
        x_target,y_target,z_target = target_point[:, 0], target_point[:, 1], target_point[:, 2]
        x, y, z = state[:, 3], state[:, 4], state[:, 5]
        x_dot, y_dot, z_dot = state[:, 0], state[:, 1], state[:, 2]
        acc_x, acc_y, acc_z = state[:, 6], state[:,7], state[:, 8]
        pitch ,  roll ,  yaw = state[:, 9], state[:,10], state[:, 11]
 
        #action = action.squeeze()
        a0,a1,a2,a3 = action[:, 0], action[:, 1], action[:, 2],action[:, 3]
        #print(f"action is :{action}")
        
        dist = get_distance(x,y,z,x_target,y_target,z_target)
        speed =  x_dot **2 + y_dot**2+ z_dot**2
        acc =  acc_x**2+ acc_y**2+ acc_z**2
        reward = np.exp(-dist ) -speed - acc
        
        # length = 0.5 # pole length
        # x_tip_error = x - length*sin_theta
        # y_tip_error = length - length*cos_theta
        # reward = np.exp(-(x_tip_error**2 + y_tip_error**2)/length**2)

        self.action_cost = True
      
        if self.action_cost:
            reward += -0.01 * a0**2-0.01 * a1**2-0.01 * a2**2-0.01 * a3**2

       

        cost = -reward

        return cost
    '''
    def get_distance(x,y,z,x_target,y_target,z_target):
        
        return np.sqrt((x-x_target)**2+(y-y_target)**2+(z-z_target)**2)
    def _get_reward_given_obs(self,obs,action ,waypoint_target = None):
        """
        Calculate the quadrotor env cost given the state and actions
        Smaller -> better
        Parameters:
        ----------
            @param numpy array - obs : size should be (batch_size x state dim)
            @param numpy array - action : size should be (batch_size x action dim)
 
        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
      
        #print(f"np.full((1000), 5)={np.full((1000), 5).shape}")
        #print(f"obs[:,-1]={obs[:,-1].shape}")
        if self.mission == 'hovering_control':
            cost = np.full((1000), 0.5)
            #a0,a1,a2,a3 = action[:, 0], action[:, 1], action[:, 2],action[:, 3]
            pitch,roll =    abs(obs[:,-5] ),abs(obs[:,-4])
            '''
            velocity_norm = np.linalg.norm(
                obs[:,0:3],axis = 1)#rotation wont change the norm of the velocity
            angular_velocity_norm = np.linalg.norm(
                obs[:,9:12],axis = 1)
            cost += 5 - 5/3.1415*roll
            cost += 5 - 5/3.1415*pitch
            alpha, beta = 1.0, 1.0
            hovering_range, in_range_r, out_range_r = 0.5, 10, -20
            """speed and angular_speed reward"""
            cost -= alpha * velocity_norm + beta * angular_velocity_norm
            # alpha*speed + beta*angular_speed
            """vertical dist reward"""
            '''
            z_move = abs(np.full((1000), 5) - obs[:,-1])#init z - sensor.z
            #print(f"z_move = {z_move}")
            #print(f"z_move = {z_move.shape }")
            cost +=   z_move  +roll**2 + pitch**2
            # if z_move < hovering_range:
                # cost += in_range_r
            # else:
                # cost += max(out_range_r, hovering_range - z_move)
            #print(f"z cost:{z_move[0]:.2f} direction cost;{roll[0]+pitch[0]:.2f}")
            return  cost
        elif self.mission == "approching":
            cost = np.full((1000), 0.5)
            pitch,roll =    abs(obs[:,-9] ),abs(obs[:,-8])
            x,y,z = obs[:,-6],obs[:,-5],obs[:,-4]
            xr,yr,zr = waypoint_target[:,0],waypoint_target[:,1],waypoint_target[:,2]
            cost = (x-xr)**2 + (y-yr)**2 + (z-zr)**2
            print(f"x= {x[0]}                                             y={y[0]}                                      z={z[0]}")
            print(f"xr= {xr[0]}                                             yr={yr[0]}                                      zr={zr[0]}")
            print(f"pos cost = {cost[0]}")
            atti_cost = roll + pitch 
            print(f"roll{roll[0]} pitch{pitch[0]}")
            cost += atti_cost  
            print(f"atti cost = {atti_cost[0]}")
            return cost