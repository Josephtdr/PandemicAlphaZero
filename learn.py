import logging
import argparse
from sys import stdout
from agent import AZ_Agent
from model import PandemicModel
from Game import Game
from torch import cuda
import numpy as np
import wandb
from collections import Counter
from concurrent.futures import as_completed, ProcessPoolExecutor

log = logging.getLogger(__name__)

def generate_ep(i, seed):
    log.info(f'Generating episode {i} with seed {seed} ... ')
    
    train_examples, info = agent.executeEpisode(seed=seed)
    log.info(f'Episode {i} from seed {seed} finished!')
    return train_examples, info
    # except Exception as e:
    #     log.warning(f'Failed task {seed} :(, \n with warning: \n{e}\n')
    #     return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_iter', type=int, default=0, help='Iteration for naming')
    parser.add_argument('--load_gen', type=int, default=None, help='Generations for loaded model')
    #Network ARGS
    parser.add_argument('--n_hidden_units', type=int, default=1024, help='Number of units per hidden layers in NN')    
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--max_lr', type=float, default=0.1, help='Max Learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9, help='Momentum during learning')
    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate')
    parser.add_argument('--cuda',type=bool,default=cuda.is_available())
    #Learning Args
    parser.add_argument('--n_generations', type=int, default=1, help='Number of generatations of training and learning to perform')
    parser.add_argument('--n_ep', type=int, default=500, help='Number of episodes to generate per generation')
    #Training Args
    parser.add_argument('--n_epoch', type=int, default=20, help='Number of batches of learning (or evlauation)  to perform to perform per generation')
    parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size')
    #MCTS ARGS
    parser.add_argument('--n_mcts', type=int, default=800, help='Number of MCTS traces per step')
    parser.add_argument('--temperature', type=int, default=1, help='Used in normalising the counts during policy creation in MCTS')
    parser.add_argument('--c_puct', type=int, default=4, help='Exploration constant used within PUCT formula')
    parser.add_argument('--dir_alpha', type=int, default=.8, help='Dirichlet noise perameter alpha')
    #Database ARGS
    parser.add_argument('--db_size', type=int, default=5, help='Initial Size of the training database, in generatations')
    parser.add_argument('--db_max_size', type=int, default=20, help='Max Size of the training database, in generatations')
    parser.add_argument('--db_rate', type=int, default=2, help='Number of generatations to perform before incrementing size of Database')
    #other
    parser.add_argument( '-log','--loglevel',default='info',choices=logging._nameToLevel.keys(),
                     help='Provide logging level. Example --loglevel debug, default=warning' )
    parser.add_argument('--n_processes', default=1,type=int, help='Number of processes to use during data generation' )
    parser.add_argument('--external_log',default=1,choices=[0,1], help='Log to wandb')
    args = parser.parse_args()

    logging.basicConfig(stream=stdout,level=args.loglevel.upper(),
        format='%(name)s - %(levelname)s - %(asctime)s %(message)s', datefmt='%I:%M:%S %p')

    if args.external_log:
        wandb.init(
        project="Pandemic_AZ",
        config = {
            "training": False,
            "Evaluating": True,
            "practice": True,
        })

    log.info('\nStarting new Attempt ...')

    log.info('Loading %s...', Game.__name__)
    env = Game()

    log.info('Loading %s...', PandemicModel.__name__)
    model = PandemicModel(env, args)

    start_gen = 0
    if args.load_gen is not None:
        start_gen = args.load_gen+1
        log.info('Loading checkpoint "%s/%s"...', args.model_iter, args.load_gen)
        model.load_checkpoint(iteration=args.model_iter, generation=args.load_gen)
    else:
        log.warning('Not loading a checkpoint!')

    agent = AZ_Agent(env, model, args)

    if args.load_gen is not None:
        log.info("Loading 'trainExamples' from file...")
        agent.training_history.load_checkpoint(iteration=args.model_iter, generation=args.load_gen)

 
    log.info('Starting the learning process ...')

    for i in range(start_gen, start_gen+args.n_generations):
        log.info(f'Starting Generation #{i} ... ')
        generation_train_examples = []
        generational_info = {}

        seeds = np.random.randint(1, 1000001, size=(args.n_ep,))

        with ProcessPoolExecutor(args.n_processes) as executor:
            futures = [executor.submit(generate_ep, i, seed) for i, seed in enumerate(seeds)]  
            for future in as_completed(futures):
                train_examples, info = future.result()
                generation_train_examples.extend(train_examples)
                generational_info = dict(Counter(generational_info) + Counter(info))
     
        log.info('Finished collating Episode Data.')

        for value in generational_info.values():
            value /= args.n_ep

        agent.training_history.store(generation_train_examples)

        agent.learn(i, generational_info)