import numpy as np
import yaml

import os

import matplotlib.pyplot as plt


class TestDataManager:

    def __init__(self, n_envs, n_algs, n_eps, algos, save_path):

        self.n_envs = n_envs
        self.n_algs = n_algs
        self.n_eps = n_eps

        self.algos = algos

        self.save_path = save_path

        self.total_avg_nn_timings = np.zeros((n_algs, 1))

        self.total_n_infeasible = np.zeros((n_algs, 1))
        self.total_n_deadlocked = np.zeros((n_algs, 1))
        self.total_n_collisions = np.zeros((n_algs, 1))
        self.total_n_finished = np.zeros((n_algs, 1))
        self.total_n_timeout = np.zeros((n_algs, 1))

        self.max_step_reward = np.zeros((n_algs, 1))

        self.eps_rewards = [[] for i in range(n_algs)]
        self.eps_steps = [[] for i in range(n_algs)]
        self.eps_env_ids = [[] for i in range(n_algs)]
        self.eps_status = [[] for i in range(n_algs)]
        self.eps_free_cells = [[] for i in range(n_algs)]
        self.eps_seed = [[] for i in range(n_algs)]

        self.n_eps_envs = np.zeros(n_envs)

        self.reset()

    def reset(self):
        self.rewards = [[] for i in range(self.n_envs)]
        self.n_coverage_finished = 0
        self.n_infeasible = 0
        self.n_collisions = 0
        self.n_deadlocked = 0
        self.n_timeout = 0
        self.actions = []
        # self.episode_step = 0

        self.episode_nn_processing_times = []
        # self.episode_mpc_processing_times = []

    def step(self, reward, infos, dones, ig_alg, alg_id, nn_runtime):
        if ig_alg != 'rl_model':
            ig_expert_runtime = sum([infos[i]["ig_expert_runtime"] for i in range(self.n_envs)]) / self.n_envs
            self.episode_nn_processing_times.append(ig_expert_runtime)

        self.n_infeasible += sum([infos[i]["is_infeasible"] for i in range(self.n_envs)])
        self.max_step_reward[alg_id] = max(np.max(reward), self.max_step_reward[alg_id])

        for i in range(self.n_envs):

            self.rewards[i].append(np.squeeze(reward[i]))

            if dones[i].any():

                scenario_seed = infos[i]['scenario_seed']

                if alg_id == 0 or scenario_seed in self.eps_seed[0]:
                    self.eps_seed[alg_id].append(scenario_seed)

                    if infos[i]["in_collision"]:
                        self.n_collisions += 1
                    if infos[i]["finished_coverage"]:
                        self.n_coverage_finished += 1
                    if infos[i]["deadlocked"]:
                        self.n_deadlocked += 1
                    if infos[i]["ran_out_of_time"]:
                        self.n_timeout += 1

                    self.eps_steps[alg_id].append(infos[i]["step_num"])
                    self.eps_free_cells[alg_id].append(infos[i]["n_free_cells"])
                    self.eps_rewards[alg_id].append(np.sum(self.rewards[i]))
                    self.eps_status[alg_id].append(0 if infos[i]["ran_out_of_time"] or infos[i]["in_collision"] else 1)
                    self.n_eps_envs[i] = infos[i]["n_episodes"]
                    self.eps_env_ids[alg_id].append([i, infos[i]["n_episodes"]])
                self.rewards[i] = []

    def get_finished_episodes_number(self, alg_id):
        return len(self.eps_rewards[alg_id])

    def algo_postprocessing(self, alg_id):

        self.total_avg_nn_timings[alg_id] = np.mean(self.episode_nn_processing_times)

        self.total_n_collisions[alg_id] = self.n_collisions
        self.total_n_infeasible[alg_id] = self.n_infeasible
        self.total_n_deadlocked[alg_id] = self.n_deadlocked
        self.total_n_finished[alg_id] = self.n_coverage_finished
        self.total_n_timeout[alg_id] = self.n_timeout

        self.eps_steps[alg_id] = self.eps_steps[alg_id][:self.n_eps]
        self.eps_free_cells[alg_id] = self.eps_free_cells[alg_id][:self.n_eps]
        self.eps_rewards[alg_id] = self.eps_rewards[alg_id][:self.n_eps]
        self.eps_status[alg_id] = self.eps_status[alg_id][:self.n_eps]
        self.eps_env_ids[alg_id] = self.eps_env_ids[alg_id][:self.n_eps]
        self.eps_seed[alg_id] = self.eps_seed[alg_id][:self.n_eps]

    def postprocessing(self):

        avg_rewards = np.mean(np.asarray(self.eps_rewards).transpose(), axis=0)
        min_rewards = np.min(np.asarray(self.eps_rewards).transpose(), axis=0)
        max_rewards = np.max(np.asarray(self.eps_rewards).transpose(), axis=0)
        std_rewards = np.std(np.asarray(self.eps_rewards).transpose(), axis=0)

        avg_steps = np.mean(np.asarray(self.eps_steps).transpose(), axis=0)

        results_dict = {
            'ig_algs': self.algos,
            'avg_rewards': avg_rewards.tolist(),
            'min_rewards': min_rewards.tolist(),
            'max_rewards': max_rewards.tolist(),
            'std_rewards': std_rewards.tolist(),
            'avg_steps': avg_steps.tolist(),
            'n_finished': self.total_n_finished.tolist(),
            'n_timeout': self.total_n_timeout.tolist(),
            'n_collisions': self.total_n_collisions.tolist(),
            'n_infeasible': self.total_n_infeasible.tolist(),
            'n_deadlocked': self.total_n_deadlocked.tolist(),
            'max_step_reward': self.max_step_reward.tolist(),
            'avg_timings': self.total_avg_nn_timings.tolist()
        }
        with open(os.path.join(self.save_path, '/results.yml'), 'w') as f:
            yaml.dump(results_dict, f)

    def _print_summary(self, results_dict):
        pass

    def save_eps_rewards(self):

        for i in range(self.n_algs):
            ig_alg = self.algos[i]
            output = np.c_[np.asarray(self.eps_rewards[i]), np.asarray(self.eps_steps[i]), np.asarray(self.eps_free_cells[i]),
                           np.asarray(self.eps_status[i]), np.asarray(self.eps_env_ids[i]), np.asarray(self.eps_seed[i])]
            np.savetxt(os.path.join(self.save_path, '/eps_' + ig_alg + '.csv'), output, delimiter=",")
        np.savetxt(os.path.join(self.save_path, '/rewards.csv'), self.eps_rewards, delimiter=",")

    def rewards_plot(self):

        fig = plt.figure()
        plt.rc('font', size=10)
        fig.set_size_inches(8, 8)
        ax = fig.add_subplot(111)
        ax.violinplot(self.eps_rewards, showmeans=True, showextrema=True)
        # ax.set_xlabel('timesteps')
        ax.set_ylabel('episode rewards')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(self.algos)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.legend()
        fig.tight_layout()

        # dateObj = datetime.now()
        # timestamp = dateObj.strftime("%Y%m%d_%H%M%S")
        fig.savefig(os.path.join(self.save_path, '/rewards.png'))