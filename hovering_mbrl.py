if __name__ == '__main__':
    import sys
    import time
    from env import *
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

