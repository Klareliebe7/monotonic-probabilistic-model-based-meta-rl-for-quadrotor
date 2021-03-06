'''
@Author: Zhengkun Li
@Email: 1st.melchior@gmail.com
'''
import os
from  mpc.models.base import Model, MLPRegression
import numpy as np
import torch
import torch.nn as nn
from time import time
def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

def CPU(var):
    return var.cpu().detach()

class DynamicModel(Model):
    def __init__(self, NN_config,model_name = None):
        super().__init__()
        if model_name:
            self.model_name = model_name    
        self.validate_result_log = []
        model_config = NN_config["model_config"]
        training_config = NN_config["training_config"]
        
        self.state_dim = model_config["state_dim"] 
        self.action_dim = model_config["action_dim"] 
        self.input_dim = self.state_dim+self.action_dim# 

        self.n_epochs = training_config["n_epochs"]
        self.lr = training_config["learning_rate"]
        self.batch_size = training_config["batch_size"]
        
        self.save_model_flag = training_config["save_model_flag"]
        self.save_model_path = training_config["save_model_path"]
        
        self.validation_flag = training_config["validation_flag"]
        self.validate_freq = training_config["validation_freq"]
        self.validation_ratio = training_config["validation_ratio"]

        if model_config["load_model"]:
            self.model = CUDA(torch.load(model_config["model_path"]))
        else:
            self.model = CUDA(MLPRegression(self.input_dim, self.state_dim, model_config["hidden_sizes"]))

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.dataset = []

    def process_dataset(self, dataset):
        # dataset format: list of [task_idx, state, action, next_state-state]
        data_list = []
        for data in dataset:
            s = data[1] # state
            a = data[2] # action
            label = data[3] # here label means the (next state - state) [state dim]
            # print(f"s = {s.shape}")
            # print(f"a = {a.shape}")
            data = np.concatenate((s, a.reshape(-1)), axis=0) # [state dim + action dim]
             
            data_torch = CUDA(torch.Tensor(data))
            label_torch = CUDA(torch.Tensor(label))
            data_list.append([data_torch, label_torch])
        return data_list

    def predict(self, s, a,axis_= 0):
        # convert to torch format
        data = np.concatenate((s, a), axis=axis_)####TODO!!! NOT SAME IN BOTH SCENARIOS
        inputs = CUDA(torch.tensor(data).float())
        
         
        #inputs = torch.cat((s, a), axis=1)
        with torch.no_grad():
            delta_state = self.model(inputs)
            delta_state = CPU(delta_state).numpy()
        return delta_state
    
    def add_data_point(self, data):
        # data format: [task_idx, state, action, next_state-state]
        self.dataset.append(data)
        
    def reset_dataset(self, new_dataset = None):
        # dataset format: list of [task_idx, state, action, next_state-state]
        if new_dataset is not None:
            self.dataset = new_dataset
        else:
            self.dataset = []
            
    def make_dataset(self, dataset, make_test_set = False):
        # dataset format: list of [task_idx, state, action, next_state-state]
        num_data = len(dataset)
        data_list = self.process_dataset(dataset)
            
        if make_test_set:
            indices = list(range(num_data))
            split = int(np.floor(self.validation_ratio * num_data))
            np.random.shuffle(indices)
            train_idx, test_idx = indices[split:], indices[:split]
            train_set = [data_list[idx] for idx in train_idx]
            test_set = [data_list[idx] for idx in test_idx]
            train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
            test_loader = None
            if len(test_set):
                test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=self.batch_size)
        else:
            train_loader = torch.utils.data.DataLoader(data_list, shuffle=True, batch_size=self.batch_size)
            test_loader = None
        return train_loader, test_loader

    def fit(self, dataset=None, logger = True):
        debug = False
        if debug:
            print(f"the first 5 weights in the model during this fit:{list(self.model.state_dict().values())[0][0][:5]}")
             
        if dataset is not None:
            train_loader, test_loader = self.make_dataset(dataset, make_test_set=self.validation_flag)
        else: # use its own accumulated data
            train_loader, test_loader = self.make_dataset(self.dataset, make_test_set=self.validation_flag)
        
        for epoch in range(self.n_epochs):
            loss_this_epoch = []
            for datas, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_this_epoch.append(loss.item())
            
            if self.save_model_flag:
                torch.save(self.model, self.save_model_path)
                
            if self.validation_flag and (epoch+1) % self.validate_freq == 0:
                loss_test = 11111111
                if test_loader is not None:
                    loss_test = self.validate_model(test_loader)
                loss_train = self.validate_model(train_loader)
                if logger:
                    print(f"model:{self.model_name},training epoch [{epoch+1}/{self.n_epochs}],loss train: {loss_train:.5f}, loss test  {loss_test:.5f}")
                
                if (np.array(self.validate_result_log)>loss_test).all() and False:
 
                    self.save_model(os.path.join(os.getcwd(),"dynaminc_model_path/",self.model_name , f"ep{epoch+1}_l_{loss_test:.5f}.pth"))
                self.validate_result_log.append(loss_test)   
        if debug:
            print(f"the first 5 weights in the model after this fit:{list(self.model.state_dict().values())[0][0][:5]}")
            
        return np.mean(loss_this_epoch)

    def validate_model(self, testloader):
        loss_list = []
        for datas, labels in testloader:
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss_list.append(loss.item())
            result = np.mean(loss_list)
            
        return result

    def reset_model(self):
        def weight_reset(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.model.apply(weight_reset)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))