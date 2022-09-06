import pickle
import os
from random import shuffle 

#Based upon https://github.com/tmoer/alphazero_singleplayer/blob/master/helpers.py
class Database:
    def __init__(self, args):
        #All measured in epochs
        self.init_max_size = args.db_size # Initial size of the database, in generatations
        self.max_size = args.db_max_size # Max size of the training database, in generatations
        self.size_increase_rate = args.db_rate # Number of generatations to perform before incrementing size of Database
        self.clear()
    
    def clear(self):
        self.history = []
        self.generation_index = 0 # Number of generatationsh of updates so far
        self.current_size = 0 # Current size of the database, in generatations
        self.current_max_size = self.init_max_size # Current max size of the database, in generatations
    
    def store(self, generation_history):
        if self.current_size < self.current_max_size:
            self.history.append(generation_history)
            self.current_size += 1
        else:
            self.history.pop(0)
            self.history.append(generation_history)
        
        if self.current_max_size<self.max_size and self.generation_index >= self.init_max_size:
            if  ((self.generation_index - self.init_max_size) % self.size_increase_rate) == 0:
                    self.current_max_size+=1
        
        self.generation_index += 1
        
    def get_history(self):
        whole_history = []
        for gen_history in self.history:
            whole_history.extend(gen_history)
        shuffle(whole_history)
        return whole_history 

    def save_checkpoint(self, folder='checkpoints',iteration=0,generation=0):
        filename=f'hist_{iteration}_gen_{generation}.pkl'
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f)

    def load_checkpoint(self, folder='checkpoints',iteration=0,generation=0):
        filename=f'hist_{iteration}_gen_{generation}.pkl'
        filepath = os.path.join(folder, filename)

        with open(filepath, 'rb') as f:
            self.history = pickle.load(f)

        self.generation_index = generation
        self.current_size = len(self.history)
        
        self.current_max_size = self.init_max_size
        for i in range(self.init_max_size, self.generation_index):
            if self.current_max_size==self.max_size:
                break
            if  ((i - self.init_max_size) % self.size_increase_rate) == 0:
                self.current_max_size+=1