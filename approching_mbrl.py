if __name__ == '__main__':
    import sys
    import time
    import torch
    from copy import deepcopy,copy
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
    task = "approching"
    env = Quadrotor(task=task, nt=1000,aproching_target = np.array([0.,0.,5.]),env_name = "env")
    env.reset()
    #env.render(np.array([0,0,0,0]))
    reset = False
    step = 1
    total_reward = 0
    ts = time.time()
    # for maml
    env_1 = Quadrotor(mass = 0.25,task=task, nt=1000,aproching_target = np.array([5.,5.,10.]),env_name = "env_1")
    env_1.reset()
    env_2 = Quadrotor(mass = 0.75,task=task, nt=1000,aproching_target = np.array([5.,5.,10.]),env_name = "env_2")
    env_2.reset()    

    """
    Ensemble Dynamic Model setting
    """
    ensemble = False
    nn_config = config['NN_config']
    print(f"Setting dynamic model with NN config:{nn_config}............")
    
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
 
    reward_model = None
#
    """
    initial MPC controller
    """
    print(f"initiallizing MPC with {optimizer_name} optimizer ...")
    mpc_controller = MPC(mpc_config=mpc_config )
    mpc_controller.mission = 'approching'
    if args.train_reward_model:
        reward_model = RewardModel(reward_config=reward_config)
    """
    DDPG setting
    """
    print(f"Setting DDPG............")
    print(f'states_dims = {env.observation_space.shape}')
    print(f'actions_dims = {env.action_space.shape[0]}')
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape, tau=0.001,  batch_size=64, fc1_dims=256, fc2_dims=256, n_actions=env.action_space.shape[0])
    """
    NN pretrain
    """
 
    # ###TODO: add meta learning here, need to make several different env with different mass and inertia 
    pretrain_episodes = 1
    print(f"NN pretraining with {pretrain_episodes} episodes............")
    if False:
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

        
 
    """
    Generating trajectories of quad with different mass for MAML
    """
    mass_order = ["0.25","0.5","0.75"]
    env_dict = {"0.25" :env_1, "0.5" :env, "0.75" :env_2,}
    pretrain_episodes = 1000
    traj_25, traj_50, traj_75 = [],[],[]
    traj_dict = {"0.25" :traj_25, "0.5" :traj_50, "0.75" :traj_75,}
    print(f"Genrating trajectories of quad with different mass  ...")
    if False:
        for epi in range(pretrain_episodes):
            for task_number,mass_index in enumerate(mass_order):
                print(f"Running  episode {epi}/{pretrain_episodes} for env mass {mass_index}...")
                obs = env_dict[mass_index].reset()
                done = False
                while not done:
                    action = env_dict[mass_index].action_space.sample()
                    obs_next, reward, done, state_next = env_dict[mass_index].step(action)
                    #model1.add_data_point([0, obs, action, obs_next - obs])
                    traj_dict[mass_index].append([task_number, obs, action, obs_next - obs])
                    obs = obs_next    
                    #env_dict[mass_index].render(action)
        print("saving  trajectories...")
        np.save("./quad_traj/0.25/traj_25.npy", traj_25)
        np.save("./quad_traj/0.50/traj_50.npy", traj_50)
        np.save("./quad_traj/0.75/traj_75.npy", traj_75)


    '''
    MAML
    '''
    if False:
        print("loading trajectories...")
        traj_25 = np.load( "./quad_traj/0.25/traj_25.npy",allow_pickle = True) 
        traj_50 = np.load( "./quad_traj/0.50/traj_50.npy",allow_pickle = True) 
        traj_75 = np.load( "./quad_traj/0.75/traj_75.npy",allow_pickle = True)   
        traj_dict = {"0.25" :traj_25, "0.5" :traj_50, "0.75" :traj_75,}    
        # building meta learning dynamic model
        model_meta = DynamicModel(NN_config=nn_config,model_name ="model_meta")
        model_meta.n_epochs = 1 #train once 
        model_meta.validate_freq = 1#vali every train
        model_meta.validation_ratio = 0.1# alive ratio reduce from normal 0.2
        ori_phi  =deepcopy(model_meta.model.state_dict()  )
        meta_gradient_zero = deepcopy(model_meta.model.state_dict()  )
        for i in ori_phi:
            meta_gradient_zero[i] = torch.zeros_like(ori_phi[i])
        maml_epoches = 50
        tasks = 3
        beta = 0.0001
        for epoch in range(maml_epoches):
            print(f"=============================epoch {epoch+1}=============================")
            loss_sum = 0.0
            theta_list = []
            loss_list = []
            test_loss_list = []
            zero = deepcopy(meta_gradient_zero)
            print("Genrating thea...  ")
            for i,mass_index in enumerate(mass_order):
                model_meta.model.load_state_dict(ori_phi)
                for traj_fragment in traj_dict[mass_index ][ :-1000]:
                    model_meta.add_data_point(traj_fragment)
                model_meta.fit()   
                model_meta.reset_dataset()
                theta_list.append(deepcopy(model_meta.model.state_dict() ) )
            print("Genrating phi...")
            for j,mass_index in enumerate(mass_order):
                model_meta.model.load_state_dict(theta_list[j])
                for traj_fragment in traj_dict[mass_index][ :-1000]:
                    model_meta.add_data_point(traj_fragment)
                loss_this_epoch = model_meta.fit()   
                loss_list.append(loss_this_epoch)
                model_meta.reset_dataset()
                theta_task = model_meta.model.state_dict()
                ##test
                test_loader,_ = model_meta.make_dataset(traj_dict[mass_index][ -1000:])
                result_test = model_meta.validate_model(test_loader)
                test_loss_list.append(result_test)
                for i in theta_task:
                    zero[i] += theta_task[i]-theta_list[j][i]
            print(f"test loss in 3 tasks:{test_loss_list}")
            for i in zero:
                zero[i] +=  ori_phi[i]
            ori_phi = zero
            torch.save(ori_phi, f"dynaminc_model_path/{model_meta.model_name}/epoch{epoch}_loss{np.mean(loss_list):.4f}_testloss{np.mean(test_loss_list):.4f}.pth")

   
    '''
    Load the model
    '''
    print("loading the pretrained meta learning weight of dynamic model")
    model1.load_model(os.getcwd()+"/dynaminc_model_path/model_meta/epoch4_loss0.0026_testloss0.0008.pth")       
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
                states, actions, rewards, states_, dones,batch_nrs = agent.last_memory.sample_buffer(int(sample_ratio*i),with_batch_nr = True)
                # for each sample start a simulation
                for ii in range(int(sample_ratio*i)):
                    obs, action, reward, state_, reset,batch_nr = states[ii], actions[ii], rewards[ii], states_[ii], dones[ii],batch_nrs[ii]
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
                        waypoint_index = np.clip(batch_nr+j,0,len(env.waypoint_targets))###TODO MAYBE PROBLEMATIC
                        waypoint_target_ = env.waypoint_targets[waypoint_index]
                        reward = env._get_reward_given_obs(obs,waypoint_target=waypoint_target_)
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
            while not reset: ####TODO ADD ENV RETURN WAYPOINT NEXT
                i+= 1
 
                waypoints_horizon = env.get_coming_waypoint_targets(mpc_config['CEM']['horizon'])#mpc horizon
                action = np.array([mpc_controller.act(model=model1, state=obs,waypoints_horizon = waypoints_horizon)])
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

