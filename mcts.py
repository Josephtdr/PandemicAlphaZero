#Adapted from https://github.com/tmoer/alphazero_singleplayer/blob/master/alphazero.py
#and https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
import numpy as np
import logging 

log = logging.getLogger(__name__)
class State():
    ''' State object '''
    def __init__(self, obs, reward, done, parent_action, available_actions, model):
        ''' Initialize a new state '''
        self.obs = obs # state
        self.terminal = done # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0 # number of visits
        
        if not done:
            priors, self.V = model.predict(obs, available_actions)
            self.child_actions = [Action(a,priors[a],parent_state=self) for a in np.where(available_actions)[0]]
        else:
            self.V = reward
    
    def select(self,c_puct=4):
        """
        Applies the puct formula to select the child action to explore
        Returns:
            the child to exlore
        """
        PUCT = np.array(
            [child_action.Q + #Q value
             c_puct * child_action.prior * (np.sqrt(self.n)/(child_action.n + 1)) # U value
                for child_action in self.child_actions], dtype=np.float64) 
        winner = np.argmax(PUCT)
        return self.child_actions[winner]

    def add_dirichlet_noise(self, alpha, x=0.75):
        dirichlet_noise = np.random.dirichlet(np.repeat(alpha, len(self.child_actions)))

        for action, noise in zip(self.child_actions, dirichlet_noise):
            action.prior = (action.prior * x) + ((1-x) * noise)

    ###Updates visit count on backward pass
    def update(self):
        self.n += 1

class Action():
    ''' Action object '''
    def __init__(self, index, prior, parent_state):
        self.index = index
        self.prior = prior #probabililty of visit according to network
        self.parent_state = parent_state    
        self.W = 0.0
        self.n = 0 # number of visits
        self.Q = 0
                
    def add_child_state(self, obs, available_actions, reward, done, model):
        self.child_state = State(
            obs=obs, reward=reward, done=done, parent_action=self,
            available_actions=available_actions, model=model
            )
        return self.child_state
        
    def update(self,value):
        self.n += 1
        self.W += value
        self.Q = self.W/self.n

class MCTS():
    ''' MCTS object '''
    def __init__(self,root,root_obs,model,args,training=True):
        self.root = None
        self.root_obs = root_obs
        self.model = model
        
        self.training = training
        self.n_mcts = args.n_mcts # Number of traces to perform per step
        self.temperature = args.temperature
        self.c_puct = args.c_puct
        self.dir_alpha = args.dir_alpha
    
    """
    Returns the MCTS estimate for the policy
    """
    def get_mcts_policy(self, env):
        ''' Perform the MCTS search from the root '''
        mask = env.available_actions()
        if self.root is None:
            self.root = State(
                obs=self.root_obs, reward=0.0, done=False, parent_action=None,
                available_actions=mask, model=self.model
            ) # Initialize new root
        else:
            self.root.parent_action = None # Set current node as root
        if self.root.terminal:
            raise(ValueError("Can't do tree search from a terminal state"))

        if self.training: 
            self.root.add_dirichlet_noise(self.dir_alpha) # Add noise to aid exploration

        # print(f'mask pre copy: {np.where(mask)[0]}')
        while(self.root.n < self.n_mcts):
            self.search(env.copy()) # copy original Env to rollout from

        counts = np.array([action.n for action in self.root.child_actions])
        
        pi = np.zeros(env.action_dimensions)
        if self.temperature==0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            idx = self.root.child_actions[best_a].index
            pi[idx] = 1
        else:
            counts = (counts / np.max(counts))**self.temperature
            probs = np.abs(counts/np.sum(counts))
            idxs = [a.index for a in self.root.child_actions]
            pi[idxs] = probs

        return pi, mask

    # Original Paper https://link.springer.com/article/10.1007/s00521-021-05928-5#Sec14
    def get_A0GB_target(self):
        # log.debug('Calculating A0GB target ... ')
        state=self.root
        # Travese tree greedily until leaf or terminal state found
        while not state.terminal and hasattr(state,'child_actions'):
            counts = np.array([a.n if hasattr(a,'child_state') else 0 for a in state.child_actions])
            if any(c for c in counts):
                best_as = np.where(counts == np.max(counts))[0]
                best_a = np.random.choice(best_as)
                state = state.child_actions[best_a].child_state
            else:
                break
        # log.debug('Calculated successfully ... ')
        return state.V

    def search(self,env):
        state = self.root # reset to root for new trace

        while not state.terminal: 
            action = state.select(c_puct=self.c_puct)
            new_obs,reward,done,_ = env.step(action.index)

            if hasattr(action,'child_state'):
                state = action.child_state # select
                continue
            else:
                state = action.add_child_state(
                    obs=new_obs,reward=reward,done=done,
                    available_actions=env.available_actions(),model=self.model) # expand & evaluate
                break

        # Back-up
        value = state.V
        while state.parent_action is not None: # loop back-up until root is reached
            action = state.parent_action
            action.update(value)
            state = action.parent_state
            state.update()
    
    def forward(self, action, next_obs):
        ''' Move the root forward '''
        child_action = [a for a in self.root.child_actions if a.index==action][0]
        if not hasattr(child_action,'child_state'):
            self.root = None
            self.root_obs = next_obs
        else:
            self.root = child_action.child_state

if __name__=="__main__":
    lst = np.array([1,2,3])

    noise = np.random.dirichlet(np.repeat(1, len(lst)))

    print(noise)
