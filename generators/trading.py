import numpy as np
from os.path import join, exists
from envs import trading


def generate_dataset(rollout_per_thread, thread_dir):
    env = trading.Env()

    for i in range(rollout_per_thread):
        env.reset()

        action_rollout  = []
        state_rollout   = []
        reward_rollout  = []
        done_rollout    = []

        while True:
            action = np.random.randint(env.action_space)

            state, reward, done, _, _ = env.step(action)

            action_rollout  += [action]
            state_rollout   += [state]
            reward_rollout  += [reward]
            done_rollout    += [done]

            if done:
                np.savez(
                    join(thread_dir, f"rollout_{i}"),
                    actions = np.array(action_rollout),
                    states  = np.array(state_rollout),
                    rewards = np.array(reward_rollout),
                    dones   = np.array(done_rollout)
                )

                break
