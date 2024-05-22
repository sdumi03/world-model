import argparse
import cma
import torch
import numpy as np

from torch.multiprocessing import Process, Queue

from os import makedirs
from os.path import join, exists
from time import sleep
from tqdm import tqdm

from models import controller
from utils import misc


def evaluate(param_queue, resul_queue, solutions, results, rollouts=100):
    """ Give current controller evaluation.

    Evaluation is minus the cumulated reward averaged over rollout runs.

    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts

    :returns: minus averaged cumulated reward
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        param_queue.put((s_id, best_guess))

    print("Evaluating...")
    # for _ in tqdm(range(rollouts)):
    for _ in range(rollouts):
        while resul_queue.empty(): sleep(.1)

        restimates.append(resul_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)


def slave_routine(param_queue, resul_queue, empty_queue, trained_dir, time_limit):
    """ Thread routine.

    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.

    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.

    As soon as e_queue is non empty, the thread terminate.

    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).

    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        r_gen = misc.RolloutGenerator(trained_dir, device, time_limit)

        while empty_queue.empty():
            if param_queue.empty(): sleep(.1)
            else:
                s_id, params = param_queue.get()
                resul_queue.put((s_id, r_gen.rollout(params)))
                # print('finished')


def main(args):
    # Multiprocessing variables
    num_workers = min(args.max_workers, args.n_samples * args.population_size)
    time_limit = 10
    print('Num workers:', num_workers)

    # Create ctrl dir if non exitent
    trained_dir = join('trained', args.env)
    ctrl_dir = join(trained_dir, 'controller')
    makedirs(ctrl_dir, exist_ok=True)

    param_queue = Queue()
    resul_queue = Queue()
    empty_queue = Queue()

    for _ in range(num_workers):
        Process(
            target=slave_routine,
            args=(
                param_queue,
                resul_queue,
                empty_queue,
                trained_dir,
                time_limit
            )
        ).start()
                                                   #misc.R_SIZE
    ctrl = controller.MODEL(misc.LATENT_SIZE, misc.RECURRENT_SIZE, misc.ACTION_SIZE)

    # define current best and load parameters
    cur_best = None
    ctrl_file = join(ctrl_dir, 'best.tar')
    if exists(ctrl_file):
        state = torch.load(ctrl_file, map_location={'cuda': 'cpu'})
        cur_best = - state['reward']
        ctrl.load_state_dict(state['state_dict'])
        print("Previous best was {}...".format(-cur_best))

    parameters = ctrl.parameters()
    evolution_strategy = cma.CMAEvolutionStrategy(
        misc.flatten_parameters(parameters),
        0.1,
        {'popsize': args.population_size}
    )

    epoch = 0
    log_step = 4
    # print('BLABLABLA', - cur_best, cur_best, args.target_return)

    # while not evolution_strategy.stop():
    while True:
        if evolution_strategy.stop(): break

        if cur_best is not None and - cur_best > args.target_return:
            print("Already better than target, breaking...")
            break

        result_list = [0] * args.population_size
        solutions = evolution_strategy.ask()

        # push parameters to queue
        for solution_id, solution in enumerate(solutions):
            for _ in range(args.n_samples):
                param_queue.put((solution_id, solution))

        # retrieve results
        if args.display: pbar = tqdm(total=args.population_size * args.n_samples)

        for _ in range(args.population_size * args.n_samples):
            while resul_queue.empty(): sleep(.1)

            result_solution_id, result = resul_queue.get()
            result_list[result_solution_id] += result / args.n_samples

            if args.display: pbar.update(1)
        if args.display: pbar.close()

        evolution_strategy.tell(solutions, result_list)
        evolution_strategy.disp()

        # evaluation and saving
        # if epoch % log_step == log_step - 1:
        if epoch % log_step == 0:
            best_params, best, std_best = evaluate(param_queue, resul_queue, solutions, result_list)

            print(f"Current evaluation: {best}")

            if not cur_best or cur_best > best:
                cur_best = best
                print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                misc.load_parameters(best_params, ctrl)
                torch.save({
                    'epoch': epoch,
                    'reward': - cur_best,
                    'state_dict': ctrl.state_dict()
                    }, join(ctrl_dir, 'best.tar'))

            if - best > args.target_return:
                print(f"Terminating controller training with value {best}...")
                break

        epoch += 1
        print()

    evolution_strategy.result_pretty()
    empty_queue.put('EOP')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', type=str, help='Enviroment for training'
    )
    parser.add_argument(
        '--n-samples', type=int, help='Number of samples used to obtain return estimate'
    )
    parser.add_argument(
        '--population-size', type=int, help='Population size'
    )
    parser.add_argument(
        '--target-return', type=float, help='Stops once the return gets above target_return'
    )
    parser.add_argument(
        '--display', action='store_true', help='Use progress bars if specified'
    )
    parser.add_argument(
        '--max-workers', type=int, default=32, help='Maximum number of workers'
    )
    args = parser.parse_args()

    main(args)