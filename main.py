import copy
import glob
import os
import time
from collections import deque

import numpy as np
import torch


from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model_ig import Policy_IG
from a2c_ppo_acktr.storage import RolloutStorage
# from evaluation import evaluate

from a2c_ppo_acktr.color_print import *

import gym_collision_avoidance

from a2c_ppo_acktr.blockstdout import stdout_redirected


def main():
    args = get_args()

    # --- Setup Torch ----------------------

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # --- Setup Envs and Paths ------------------

    log_dir = os.getcwd() + '/data'
    save_path = os.path.join(log_dir, "log_{}".format(utils.get_latest_run_id(log_dir) + 1))
    args.save_dir = save_path

    eval_log_dir = save_path + "/eval"
    utils.cleanup_log_dir(save_path)
    utils.cleanup_log_dir(eval_log_dir)

    print("Log path: {}".format(save_path))

    args.env_name = 'CollisionAvoidance-v0'
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, save_path, device, False)

    # Save plot trajectories
    for i in range(args.num_processes):
        plot_save_dir = save_path + '/figures_train/figs_env' + str(i) + '/'
        envs.env_method('set_plot_save_dir', plot_save_dir, indices=i)
        envs.env_method('set_n_env', args.num_processes, i, False, indices=i)
        if i != 0:
            envs.env_method('set_plot_env', False, indices=i)

    tensorboard_writer = None
    if args.tensorboard_logdir is not None or save_path is not None:
        from tensorboardX import SummaryWriter
        import datetime
        ts_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(save_path + '/tb/', ts_str))

    # ----------------------------------------------------------------------------------------------------

    # Init Network Model
    actor_critic = Policy_IG(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy, 'device': device, 'fix_cnn': False})
    actor_critic.to(device)

    # Init Algorithm
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    # Init Batch Buffer
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    # Environment Settings
    envs.env_method('set_use_expert_action', 1, False, '', False, 0.0, False)
    envs.env_method('set_n_obstacles', 2)

    # with stdout_redirected():
    obs = envs.reset()

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    epinfos_reset = {
        'reward': 0.0,
        'ig_reward': 0.0,
        'n_episodes': 0,
        'n_infeasible': 0,
        'n_timeout': 0,
        'n_collision': 0,
        'n_deadlocked': 0,
        'n_steps_avg': 0
    }

    episode_rewards = deque(maxlen=10)

    reward_counter = torch.zeros((args.num_processes, 1))
    reward_ig_counter = torch.zeros((args.num_processes))

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        epinfos = copy.copy(epinfos_reset)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obs reward and next obs
            with stdout_redirected():
                obs, reward, done, infos = envs.step(action)

            # for info in infos:
            #     if 'episode' in info.keys():
            #         episode_rewards.append(info['episode']['r'])

            # Save Episode Infos
            reward_counter += reward
            reward_ig_counter += torch.Tensor([info["ig_reward"] for info in infos])
            done_idc = np.array(done.nonzero()).squeeze(axis=0)
            if done_idc.size > 0:
                for i in done_idc.tolist():
                    epinfos['n_episodes'] += 1

                    epinfos['reward'] += reward_counter[i]
                    episode_rewards.append(reward_counter[i].item())
                    reward_counter[i] = 0.0

                    epinfos['ig_reward'] += reward_ig_counter[i]
                    reward_ig_counter[i] = 0.0

                    epinfos['n_steps_avg'] += infos[i]['step_num']
                    epinfos['n_timeout'] += infos[i]["ran_out_of_time"]
                    epinfos['n_collision'] += infos[i]["in_collision"]
                    epinfos['n_deadlocked'] += infos[i]["deadlocked"]
                    epinfos['n_infeasible'] += infos[i]["is_infeasible"]

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, approx_kl, clipfrac, returns, epochs = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + '_' + str(j) + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            if epinfos['n_episodes'] > 0:
                for key in epinfos.keys():
                    if key == "n_episodes":
                        continue
                    epinfos[key] = epinfos[key] / epinfos['n_episodes']

            print_cyan(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))
            # time.sleep(2)

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("rewards/mean_reward", np.mean(episode_rewards), total_num_steps)
                tensorboard_writer.add_scalar("rewards/median_reward", np.median(episode_rewards), total_num_steps)
                tensorboard_writer.add_scalar("rewards/min_reward", np.min(episode_rewards), total_num_steps)
                tensorboard_writer.add_scalar("rewards/max_reward", np.max(episode_rewards), total_num_steps)

                tensorboard_writer.add_scalar("train/dist_entropy", dist_entropy, total_num_steps)
                tensorboard_writer.add_scalar("train/value_loss", value_loss, total_num_steps)
                tensorboard_writer.add_scalar("train/policy_loss", action_loss, total_num_steps)
                tensorboard_writer.add_scalar("train/approx_kl", approx_kl, total_num_steps)
                tensorboard_writer.add_scalar("train/clipfrac", clipfrac, total_num_steps)
                tensorboard_writer.add_scalar("train/returns", returns, total_num_steps)
                tensorboard_writer.add_scalar("train/epochs", epochs, total_num_steps)

                if epinfos['n_episodes'] > 0:
                    for key in epinfos.keys():
                        tensorboard_writer.add_scalar('rollouts/' + key, epinfos[key], total_num_steps)
                else:
                    tensorboard_writer.add_scalar('rollouts/n_episodes', epinfos['n_episodes'], total_num_steps)

        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     obs_rms = utils.get_vec_normalize(envs).obs_rms
        #     evaluate(actor_critic, obs_rms, args.env_name, args.seed,
        #              args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
