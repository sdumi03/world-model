import argparse
from multiprocessing import Pool
from os import makedirs
from os.path import join, exists
from importlib import import_module


def thread_generator(thread_args):
    ix, generator, rollout_per_thread = thread_args

    thread_dir = join('datasets', generator, 'thread_{ix}')
    makedirs(thread_dir, exist_ok=True)

    generator_path = join('generators', generator)
    assert exists(generator_path), 'The generator does not exists...'

    g = import_module('generators.{generator}')
    g.generate_dataset(rollout_per_thread, thread_dir)

    print(f"Generated Dir: {thread_dir}, Rollouts: {rollout_per_thread}")

    return True


def main(args):
    rollout_per_thread = args.rollouts // args.threads
    assert rollout_per_thread > 0, 'The number of rollouts should be > 0'

    thread_args = [(ix, args.generator, rollout_per_thread) 
                   for ix in range(args.threads)]

    with Pool(args.threads) as thread:
        thread.map(thread_generator, thread_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help='Number of rollouts')
    parser.add_argument('--threads', type=int, help='Number of threads')
    parser.add_argument('--generator', type=str, help='Generator of rollouts')
    args = parser.parse_args()

    main(args)