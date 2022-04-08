import numpy as np
import torch
import os
import time
import argparse

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.blockstdout import stdout_redirected
from a2c_ppo_acktr.test_data_manager import TestDataManager

parser = argparse.ArgumentParser(description='RL')
# parser.add_argument(
#     '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--run-id',
    type=int,
    default=0,
    help='which training to evaluate'
)
parser.add_argument(
    '--n-episodes',
    type=int,
    default=100,
    help='number of test episodes'
)
parser.add_argument(
    '--n-processes',
    type=int,
    default=1,
    help='number of env processes'
)
parser.add_argument(
    '--ig-algos',
    default='rl_model ig_greedy',  # ig_mcts
    help='which if algorithms to evaluate'
)
args = parser.parse_args()


def main():
    log_dir = os.getcwd() + '/data'
    run_id = args.run_id if args.run_id > 0 else utils.get_latest_run_id(log_dir)
    run_path = os.path.join(log_dir, "log_{}".format(run_id))
    save_path = os.path.join(run_path, '/test')


    utils.cleanup_log_dir(save_path)

    args.env_name = 'CollisionAvoidance-v0'
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, save_path, device='cpu', allow_early_resets=False)

    # Setting to save plot trajectories
    for i in range(args.num_processes):
        plot_save_dir = save_path + '/figures_test/figs_env' + str(i) + '/'
        envs.env_method('set_plot_save_dir', plot_save_dir, indices=i)
        envs.env_method('set_n_env', args.n_processes, i, False, indices=i)
        if i != 0:
            envs.env_method('set_plot_env', False, indices=i)

    def get_latest_model_file(run_path):
        p = [item for item in os.listdir(os.path.join(run_path, '/ppo')) if item.endswith('.pt')]

        p.sort()
        return p[-1]

    # Load Network Model
    actor_critic, obs_rms = torch.load(get_latest_model_file(run_path), map_location='cpu')

    # Init LSTM reset masks and hidden states
    masks = torch.zeros(args.n_processes, 1)
    recurrent_hidden_states = torch.zeros(
        args.n_processes, actor_critic.recurrent_hidden_state_size)

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    # Environment Settings
    envs.env_method('set_use_expert_action', 1, False, '', False, 0.0, False)
    envs.env_method('set_n_obstacles', 2)

    # Get IG algorithms to compare
    ig_algs = args.ig_algos.split()
    n_algs = len(ig_algs)

    # Init Test Data Storage Object
    test_data = TestDataManager(args.n_processes, n_algs, args.n_episodes, ig_algs, save_path)

    for alg_id in range(n_algs):

        ig_alg = ig_algs[alg_id]

        for i in range(args.n_processes):
            # eps_num = env.get_attr("episode_number", indices=i)
            if ig_alg == 'rl_model':
                envs.env_method('set_use_expert_action', alg_id, False, "", indices=i)
            else:
                envs.env_method('set_use_expert_action', alg_id, True, ig_alg, indices=i)

        with stdout_redirected():
            obs = envs.reset()

        test_data.reset()

        while test_data.get_finished_episodes_number(alg_id) < args.n_episodes:
            start = time.time()
            # Query Policy
            with torch.no_grad():
                _, action, _, recurrent_hidden_states = actor_critic.act(obs, recurrent_hidden_states, masks,
                                                                         deterministic=True)
            end = time.time()
            nn_runtime = end - start
            # episode_nn_processing_times.append((end - start) / n_envs)
            # actions.append(action)

            # Obs reward and next obs
            scaled_action = 4.0 * action
            with stdout_redirected():
                obs[:], reward, dones, infos = envs.step(scaled_action)

            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in dones],
                dtype=torch.float32)

            # Save data
            test_data.step(reward, infos, dones, ig_alg, alg_id, nn_runtime)

            eps_number = test_data.get_finished_episodes_number(alg_id)
            if eps_number % 10 == 0:
                print('Episode ' + str(eps_number) + ' completed with algorithm' + ig_alg)

        # Collect data from this algo
        test_data.algo_postprocessing(alg_id)

    # Collect all data and print yaml file
    test_data.postprocessing()

    # Save episode rewards as csv files
    test_data.save_eps_rewards()


if __name__ == "__main__":
    main()
