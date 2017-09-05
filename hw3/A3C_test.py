import os
import os.path as osp
import threading
import multiprocessing
import time
import random
import scipy.signal
import itertools
import numpy as np
import tensorflow as tf
import gym
from gym import wrappers

import logz
from A3C_model import *
from dqn_utils import *
from atari_wrappers import *

#dir for summary(tenorboard)
SAVE_DIR = '/tmp/A3C_train_8'
GLOBAL_TIMESTEPS = 0
GLOBAL_EPISODES = 0

def discount(rew_t, V_T, gamma):
    x = np.array(list(rew_t) + [V_T])
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def update_network(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    update_fn = []
    for from_var, to_var in zip(sorted(from_vars, key=lambda v: v.name),
                                sorted(to_vars, key=lambda v: v.name)):
        update_fn.append(to_var.assign(from_var))
    return update_fn

class Worker(object):
    def __init__(self, name, env, coord, global_network, logdir=None):
        self.name = name
        self.env = env
        self.coord = coord
        self.global_network = global_network
        self.input_shape = env.observation_space.shape
        self.output_dim = env.action_space.n
        self.local_network = A3C_Network(self.input_shape, self.output_dim, name=name, logdir=logdir)
        self.update_local_ops = update_network('global', self.name)

        self.episodes = 0
        
    def work(self, sess, max_timesteps, min_timesteps_per_batch, gamma=0.99):
        global GLOBAL_TIMESTEPS
        global GLOBAL_EPISODES
        mean_episode_reward = -float('inf')
        best_mean_episode_reward = -float('inf')
        log_worker = (self.name == 'thread_0')

        #schedule learning rate
        num_iterations = float(max_timesteps)
        lr_multiplier = 1.0*1
        lr_schedule = PiecewiseSchedule([
                                            (0,                   1e-4 * lr_multiplier),
                                            (num_iterations / 10, 1e-4 * lr_multiplier),
                                            (num_iterations / 2,  5e-5 * lr_multiplier),
                                        ],
                                        outside_value=5e-5 * lr_multiplier)
        ###############
        # RUN ENV     #
        ###############
        with sess.as_default(), sess.graph.as_default():
            # Init local network
            Model = self.local_network
            env = self.env
            sess.run(self.update_local_ops)

            # Run each episode recurrently
            while not self.coord.should_stop():
                GLOBAL_EPISODES += 1
                self.episodes += 1
                # Stopping criterion
                if GLOBAL_TIMESTEPS >= max_timesteps:
                    coord.request_stop()
                    break
                if GLOBAL_EPISODES % 10 ==0:
                    print('----------num_episode:{}----------'.format(GLOBAL_EPISODES))
                
                # Init env
                ob = env.reset()
                done = False
                obs, acs, rewards, vpred_t = [], [], [], []
                current_episode_reward = 0
                rnn_state = Model.state_init
                self.batch_rnn_state = rnn_state

                # Run env, collect one episode
                while not done:
                    obs.append(ob)
                    ac,v,rnn_state = sess.run([Model.sample_one_ac,Model.Vs[0],Model.state_out], 
                        feed_dict={Model.inputs:[ob],
                        Model.state_in[0]:rnn_state[0],
                        Model.state_in[1]:rnn_state[1]})
                    #ac = sess.run(Model.sample_one_ac, feed_dict={Model.inputs: ob[None,:]})
                    acs.append(ac)
                    vpred_t.append(v)
                    ob, rew, done, _ = env.step(ac)
                    rewards.append(rew)
                    current_episode_reward += rew
                    GLOBAL_TIMESTEPS += 1
                    if GLOBAL_TIMESTEPS%1000==0: print('{}, {}, {}'.format(GLOBAL_TIMESTEPS, return_t[:4], vpred_t[:4]))
                    
                    ###############
                    # TRAIN MODEL #
                    ###############
                    # If done or buffer is full then get to train
                    if done or len(rewards)==min_timesteps_per_batch:
                        # Estimate advantage function
                        #vpred_t = sess.run(Model.Vs, feed_dict={Model.inputs: np.array(obs)})
                        v_T = sess.run(Model.Vs, feed_dict={Model.inputs: np.array(ob[None,:]), 
                                                            Model.state_in[0]:rnn_state[0],
                                                            Model.state_in[1]:rnn_state[1]})
                        return_t = discount(np.array(rewards), v_T[0]*(1-done), gamma)[:-1]
                        #return_t = discount(np.array(rewards), 0, gamma)[:-1]
                        feed_dict = {Model.inputs: np.array(obs),
                            Model.action_samples: np.array(acs),
                            Model.V_ts: np.array(vpred_t),
                            Model.Returns: return_t,
                            Model.learning_rate: lr_schedule.value(GLOBAL_TIMESTEPS),
                            Model.state_in[0]:self.batch_rnn_state[0],
                            Model.state_in[1]:self.batch_rnn_state[1]}
                        
                        # Compute gradient
                        _, old_action_Ps, self.batch_rnn_state = sess.run([Model.apply_grads, Model.act_Ps, Model.state_out], feed_dict=feed_dict)
                        # Update local network
                        sess.run(self.update_local_ops)

                        if not done:
                            obs, acs, rewards, vpred_t = [], [], [], []
                ###################
                # LOG and SUMMARY #
                ###################
                # Summary and save
                if self.episodes % 3 == 0 and log_worker:
                    print('\n add summary:', GLOBAL_TIMESTEPS, 'total episode:', GLOBAL_EPISODES, 'worker episode:', self.episodes)
                    print(len(rewards), return_t)
                    global_feed_dict = {self.global_network.inputs: np.array(obs),
                        self.global_network.action_samples: np.array(acs),
                        self.global_network.V_ts: vpred_t,
                        self.global_network.Returns: return_t,
                        self.global_network.state_in[0]:self.batch_rnn_state[0],
                        self.global_network.state_in[1]:self.batch_rnn_state[1]}
                    summary_str = sess.run(self.global_network.summary_op, feed_dict=global_feed_dict)
                    self.global_network.summary_writer.add_summary(summary_str, GLOBAL_TIMESTEPS)
                    summary_str = sess.run(Model.summary_op, feed_dict=feed_dict)
                    Model.summary_writer.add_summary(summary_str, GLOBAL_TIMESTEPS)
                    # Computer KL and entropy
                    kl, ent = sess.run([Model.KL, Model.entropy], feed_dict={Model.inputs:np.array(obs), 
                                                                                Model.state_in[0]:self.batch_rnn_state[0],
                                                                                Model.state_in[1]:self.batch_rnn_state[1],
                                                                                Model.old_act_Ps:old_action_Ps})
                    # Log diagnostics
                    episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
                    if len(episode_rewards) > 0:
                        mean_episode_reward = np.mean(episode_rewards[-100:])
                    if len(episode_rewards) > 100:
                        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
                    logz.log_tabular("EpRewCurrent", current_episode_reward)
                    logz.log_tabular("EpRewMean", mean_episode_reward)
                    logz.log_tabular("EpBestRewMean", best_mean_episode_reward)
                    logz.log_tabular("KLOldNew", kl)
                    logz.log_tabular("Entropy", ent)
                    logz.log_tabular("learning_rate", lr_schedule.value(GLOBAL_TIMESTEPS))
                    logz.log_tabular("TimestepsSoFar", GLOBAL_TIMESTEPS)
                    # If you're overfitting, EVAfter will be way larger than EVBefore.
                    # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
                    logz.dump_tabular()

                    
def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    #set seed
    tf.set_random_seed(seed) 
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    env = wrap_deepmind(env)

    return env

def main_A3C():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    seed = 0 # Use a seed of zero (you may want to randomize the seed!)

    #logdir for logz which is used to plot learning curve
    logdir = '/tmp/log/A3C_Pong_seed0'
    if tf.gfile.Exists(logdir):
        tf.gfile.DeleteRecursively(logdir)
    if tf.gfile.Exists(SAVE_DIR):
        tf.gfile.DeleteRecursively(SAVE_DIR)

    logz.configure_output_dir(logdir)

    tf.reset_default_graph()
    coord = tf.train.Coordinator()
    num_workers = min(multiprocessing.cpu_count(), 4)
    print('--------{} threads are used--------'.format(num_workers))
    workers = []
    env_list = []
    worker_threads = []

    try:
        #initialize workers
        for i in range(num_workers):
            env = get_env(task, seed)
            env_list.append(env)

            if i == 0:
                expt_dir = '/tmp/hw3_vid_dir2_A3C_test/'
                env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
                input_shape = env.observation_space.shape
                output_dim = env.action_space.n
                global_network = A3C_Network(input_shape, output_dim, name='global', logdir=SAVE_DIR)
                
            worker_save_path = SAVE_DIR+str(i)
            if tf.gfile.Exists(worker_save_path):
                tf.gfile.DeleteRecursively(worker_save_path)
            print('Creat worker: thread_{}'.format(i))
            workers.append(Worker('thread_{}'.format(i), env, coord, global_network, worker_save_path))

        # Run training
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True 
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        for worker in workers:
            worker_work = lambda: worker.work(sess, task.max_timesteps, 30)
            t = threading.Thread(target=worker_work)
            print('Start worker: {}'.format(worker.name))
            t.start()
            time.sleep(0.5)
            worker_threads.append(worker_work)

        print("Ctrl + C to close")
        coord.wait_for_stop()

        print('Save the checkpoint')
        checkpoint_path = os.path.join(SAVE_DIR, 'model.ckpt')
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, checkpoint_path)

    except KeyboardInterrupt:
        coord.request_stop()

    finally:

        coord.request_stop()
        coord.join(worker_threads)

        print("Closing Environments")
        for env in env_list:
            env.close()

        print("Closing Session")
        sess.close()

if __name__ == "__main__":
    main_A3C()

