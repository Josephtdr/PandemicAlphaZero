import wandb
import logging
import numpy as np
from tqdm import tqdm
import concurrent.futures
from itertools import repeat
from collections import Counter
from mcts import MCTS
from database import Database

log = logging.getLogger(__name__)
#Based upon https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py
class AZ_Agent():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, env, model, args, training=True, practice=True):
        self.env = env
        self.model = model
        self.args = args
        self.training = training
        if training:
            self.training_history = Database(args)  # history of examples
        # wandb.init(
        # project="pandemic-ppo-raw",
        # config = {
        #     "training": training,
        #     "Evaluating": not training,
        #     "practice": practice,
        #     "learning_rate": "unknown",
        #     "architecture": "recNN",
        #     "batch_size": "unknown"
        # })

    def executeEpisode(self, seed=None, model=None):
        """
        This function executes one episode of self-play.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        Returns:
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
            return train_examples, info  # [x.append(v) for x in train_examples]
        else:
            return v, info

    # def learn_worker(self, i, seed):
    #     try:
    #         log.debug(f'Generating episode {i} ... ')
    #         train_examples, info = self.executeEpisode(seed=seed)
    #         log.debug(f'Episode {i} finished')
    #         return train_examples, info
    #     except Exception as e:
    #         log.warning(f'Failed task {seed} :(, \n with warning: \n{e}\n')
    #         return None

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

    # def ev_worker(self, model, seed):
    #     log.info(f'Evaluating seed {seed}')
    #     try:
    #         v, info = self.executeEpisode(model=model,seed=seed)
    #         log.debug(f"Finished Evaluating seed {seed}, obatained result {v}")
    #         return info
    #     except Exception as e:
    #         log.warning(f'Failed task {seed} :(, \n with warning {e}')
    #         return e
        

    # def evaluate(self, model, seeds):
    #     self.training = False
    #     external_log = False

    #     if external_log:
    #         wandb.init(
    #         project="Pandemic_AZ",
    #         config = {"Evaluating": not self.training,
    #             "training": self.training,
    #             "Evaluating": not self.training,
    #             "practice": True,
    #         })

    #     with concurrent.futures.ProcessPoolExecutor(self.args.n_processes) as executor:
    #         # results = list(tqdm(executor.map(self.ev_worker, repeat(model), seeds), total=len(seeds)))
    #         results = executor.map(self.ev_worker, repeat(model), seeds)
    #         for result in results:
    #             if external_log:
    #                 wandb.log(result)
    #         log.info(f'Processing Finished')


    #     return 
    #     def crawl(q,process):
    #         while not q.empty():
    #             seed = q.get()
    #             try:
    #                 _, info = self.executeEpisode(model=model,seed=seed,process=process)
    #                 log.debug(f"Finished Evaluating seed {seed}")
    #             except:
    #                 log.warning(f'Process {process} failed task {seed} :(')
    #             # wandb.log(info)
        
    #     q = multiprocessing.Queue()
    #     num_processes = 2
    #     for seed in seeds:
    #         q.put(seed)

    #     processes = []

    #     for i in range(num_processes):
    #         log.debug(f'Starting process #{i} ...')
    #         process = multiprocessing.Process(target=crawl, args=(q,i,))
    #         process.start()
    #         processes.append(process)

    #     for process in processes:
    #         process.join()
    #     # q.join()

    #     log.info(f'All tasks completed')
    #     return
    #     return
    #     V = 0
    #     t = tqdm(seeds, desc=f'Evaluating model ')
    #     for seed in t:


    #         v, info = self.executeEpisode(model=model,seed=seed)
    #         wandb.log(info)

    #         if v > 0:
    #             V += v
    #     log.info(f"Finished Evaluation, obtained winrate {V/len(seeds)}")
    #     print(f"Finished Evaluation, obtained winrate {V/len(seeds)}")

if __name__=="__main__":
    pass