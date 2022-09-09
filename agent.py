import wandb
import logging
import numpy as np
from mcts import MCTS
from database import Database

log = logging.getLogger(__name__)
#Adapted from https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py
class AZ_Agent():
    def __init__(self, env, model, args, training=True, practice=True):
        self.env = env
        self.model = model
        self.args = args
        self.training = training
        if training:
            self.training_history = Database(args)  # history of examples

    def executeEpisode(self, seed=None, model=None):
        """
        This function executes one episode of self-play.
        If in training mode, as the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        If in training mode, Returns:
            trainExamples: a list of examples of the form (state, pi, v)
                           pi is the MCTS informed policy vector,
                           v the A0GB value 
        """
        episode_info = {
            'DISCARD' : 0, 'DRIVE_FERRY' : 0, 'DIRECT_FLIGHT' : 0, 
            'CHARTER_FLIGHT' : 0, 'SHUTTLE_FLIGHT' : 0, 'GIVE_KNOWLEDGE' : 0, 
            'RECEIVE_KNOWLEDGE' : 0, 'BUILD_RESEARCHSTATION' : 0, 
            'TREAT_DISEASE' : 0, 'DISCOVER_CURE' : 0,  
        }

        train_examples = []
        if model is None:
            model = self.model

        obs= self.env.reset(seed=seed)
        done = False

        mcts = MCTS(root=None, root_obs=obs, model=model, args=self.args,training=self.training)

        while not done:
            pi, mask = mcts.get_mcts_policy(self.env)
            if self.training:
                v = mcts.get_A0GB_target()
                train_examples.append([mcts.root.obs, pi, v, mask])
        
            action = np.random.choice(len(pi),p=pi)

            log.debug(f'Advancing real game with action #{action}, task {seed}...')
            obs, v, done, info = self.env.step(action)
            mcts.forward(action, obs)
            
            episode_info[info['action_type'].name] += 1
        
        del info['action_type']
        info.update(episode_info)

        if self.training:
            return train_examples, info
        else:
            return v, info

    def executeEpisode_NOMCTS(self, seed=None, model=None):
        episode_info = {
            'DISCARD' : 0, 'DRIVE_FERRY' : 0, 'DIRECT_FLIGHT' : 0, 
            'CHARTER_FLIGHT' : 0, 'SHUTTLE_FLIGHT' : 0, 'GIVE_KNOWLEDGE' : 0, 
            'RECEIVE_KNOWLEDGE' : 0, 'BUILD_RESEARCHSTATION' : 0, 
            'TREAT_DISEASE' : 0, 'DISCOVER_CURE' : 0,  
        }
        env = self.env.copy()
        obs= env.reset(seed=seed)
        done = False

        while not done:
            mask = env.available_actions()
            priors, _ = model.predict(obs, mask)

            posACTS = np.where(priors==np.max(priors))[0]
            action = np.random.choice(posACTS)

            obs, v, done, info = env.step(action)
            episode_info[info['action_type'].name] += 1
        
        del info['action_type']
        info.update(episode_info)

        return v, info

    def learn(self, i, generational_info):
        
            log.info(f'Saving Iter {self.args.model_iter}, Generation #{i} history ... ')
            self.training_history.save_checkpoint(iteration=self.args.model_iter, generation=i)

            log.info('Beginning Learning ...')
            pi_loss, v_loss = self.model.train(self.training_history.get_history())

            log.info(f'Logging generation #{i} data ... ')
            generational_info['pi_loss'] = pi_loss
            generational_info['v_loss'] = v_loss
            if self.args.external_log:
                wandb.log(generational_info)

            log.info(f'Saving Iter {self.args.model_iter}, Generation #{i} model, avg loss {pi_loss+v_loss} ... ')
            self.model.save_checkpoint(iteration=self.args.model_iter, generation=i)

if __name__=="__main__":
    pass