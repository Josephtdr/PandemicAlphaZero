from concurrent.futures import as_completed, ProcessPoolExecutor
import logging
import argparse
from sys import stdout
from agent import AZ_Agent
from model import PandemicModel
from Game import Game
import numpy as np
import wandb

log = logging.getLogger(__name__)

def evaluate_model(i, seed):
    log.info(f'Evaluating ep {i}, seed {seed}')
    try:
        v, info = agent.executeEpisode(model=model,seed=seed)
        log.info(f"Finished Evaluating seed {seed}, obatained result {'WON' if v==1 else 'LOST'}")
        return info
    except Exception as e:
        log.warning(f'Failed task {seed} :(, \n with warning {e}')
        return e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_iter', type=int, default=0, help='Iteration for naming')
    parser.add_argument('--load_gen', type=int, nargs="*", default=[], help='Generations for loaded model')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--max_lr', type=float, default=0.1, help='Max Learning rate')
    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate')
    parser.add_argument('--cuda',type=bool,default=False, help='Warning code not fully setup to run with cuda, unsupported')
    parser.add_argument('--n_ep', type=int, default=500, help='Number of episodes to generate per generation')
    parser.add_argument('--n_mcts', type=int, default=800, help='Number of MCTS traces per step')
    parser.add_argument('--temperature', type=int, default=1, help='Used in normalising the counts during policy creation in MCTS')
    parser.add_argument('--c_puct', type=int, default=4, help='Exploration constant used within PUCT formula')
    parser.add_argument('--dir_alpha', type=int, default=.8, help='Dirichlet noise perameter alpha')
    parser.add_argument( '-log','--loglevel',default='info',choices=logging._nameToLevel.keys(), help='Provide logging level. Example --loglevel debug, default=warning' )
    parser.add_argument('--external_log',default=0,type=int,choices=[0,1], help='Log to wandb')
    parser.add_argument('--n_processes', default=1,type=int, help='Number of processes to use during data generation' )
    parser.add_argument('--n_hidden_units', type=int, default=1024, help='Number of units per hidden layers in NN')    
    args = parser.parse_args()

    logging.basicConfig(stream=stdout,level=args.loglevel.upper(),
        format='%(name)s - %(levelname)s - %(asctime)s %(message)s', datefmt='%I:%M:%S %p')

    log.info('\nStarting Evaluation Attempt ...')
    log.info('Loading %s...', Game.__name__)
    env = Game()
    log.info('initialising Agent...')
    agent = AZ_Agent(env, None, args, training=False)

    log.info(f'Initialising model {PandemicModel.__name__}')
    model = PandemicModel(env, args)

    if not args.load_gen:
        log.warning('Not loading any checkpoints, exiting!')
        exit()

    seeds = np.random.randint(1, 1000001, size=(args.n_ep,))
    
    for load_gen in args.load_gen:
        if args.external_log:
            run = wandb.init(
                project="Pandemic_AZ",
                reinit=True,
                name=f'fn_Generation_{load_gen}',
                config = {"training": False}
            )
        log.info(f'Loading checkpoint Iter:{args.model_iter}, Generation:{load_gen} ...')
        model.load_checkpoint(iteration=args.model_iter, generation=load_gen)

        log.info('Starting the evaluation process ...')

        with ProcessPoolExecutor(args.n_processes) as executor:
            futures = [executor.submit(evaluate_model, i, seed) for i, seed in enumerate(seeds)]  
            for future in as_completed(futures):
                result = future.result()
                if args.external_log:
                    run.log(result)
        log.info(f'Processing Finished')
        if args.external_log:
            run.finish()