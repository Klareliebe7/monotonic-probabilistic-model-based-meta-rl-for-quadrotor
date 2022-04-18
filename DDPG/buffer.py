import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.global_state_memory = [0 for _ in range(max_size)]
    def store_transition(self, state, action, reward, state_, done,global_state = None):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        if global_state:
            self.global_state_memory[index] = global_state
        self.mem_cntr += 1

    def sample_buffer(self, batch_size,with_global_state = False,with_batch_nr = False):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        if with_global_state:
            global_state = []
            for ii in batch:
                global_state.append(self.global_state_memory[ii])
            return states, actions, rewards, states_, dones,global_state
        if with_batch_nr :
            return states, actions, rewards, states_, dones,batch
        return states, actions, rewards, states_, dones

    def reset(self):
            self.mem_cntr = 0
            self.state_memory = np.zeros((self.mem_size, * self.input_shape))
            self.new_state_memory = np.zeros((self.mem_size, * self.input_shape))
            self.action_memory = np.zeros((self.mem_size,  self.n_actions))
            self.reward_memory = np.zeros(self.mem_size)
            self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
            self.global_state_memory = [0 for _ in range(self.mem_size)]