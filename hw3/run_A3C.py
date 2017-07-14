import os
import sys
import gym
import gym.spaces
import scipy.signal
import argparse
import time
import itertools
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

import A3C_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/A3C_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('save_dir', '/tmp/A3C_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_iters', 100000,
                            """Number of batches to run.""")

def Return_compute(rewards, V_T, gamma=0.95):
    x = np.array(rewards + [V_T])
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def A3C_learn(env, train_iters=FLAGS.max_iters, num_timesteps=20, gamma=0.95):
    
    input_shape = env.observation_space.shape
    action_num = env.action_space.n
    with tf.Graph().as_default() as g:
        ###############
        # RUN ENV     #
        ###############
        obs_t = env.reset()
        done = False

        model = A3C_model.A3C_Network(input_shape, action_num)

        summary_op = tf.summary.merge_all()

        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True 
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            summary_writer = tf.summary.FileWriter(FLAGS.save_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables())

            for global_step in range(train_iters):
                observations = []
                actions = []
                rewards = []
                V_values = []
                if done:
                    obs_t = env.reset()
                    done = False
            
                for t in range(num_timesteps):
                    ### 1. Check stopping criterion
                    if done:
                        break
                    observations.append(obs_t)
                    ###1. Build the graph
                    act_P, V_t_ph = (model.act_Ps[0,:], model.Vs[0])
                    act_t_ph = tf.contrib.distributions.Categorical(probs=act_P).sample()

                    ### 2. Run and collect the data: step the env and store the transition
                    act_t, V_t = sess.run([act_t_ph, V_t_ph], feed_dict={model.inputs: obs_t[None,:]})

                    obs_t, rew_t, done, _ = env.step(act_t)
                    actions.append(act_t)
                    rewards.append(rew_t)
                    V_values.append(V_t)

                ### 3. Train the network.
                Returns = Return_compute(rewards, V_values[-1], gamma)[:-1]
                
                feed_dict = {model.inputs: np.array(observations),
                            model.action_samples: np.array(actions),
                            model.V_ts: np.array(V_values),
                            model.Returns: Returns}
                ###apply the gradient
                _, summary_str = sess.run([model.apply_gradients_op, summary_op], feed_dict=feed_dict)
                print(global_step)
                
                if global_step % 100 == 0:
                    print(global_step, train_iters)
                    summary_writer.add_summary(summary_str, global_step)

                if global_step % 1000 == 0 or (global_step + 1) == train_iters:
                    checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument("--max_iters", default=FLAGS.max_iters, type=int)
    args = parser.parse_args()
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    FLAGS.save_dir = os.path.join(FLAGS.train_dir, args.envname)
    if tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_dir)
    tf.gfile.MakeDirs(FLAGS.save_dir)
    
    env = gym.make(args.envname)

    A3C_learn(env, args.max_iters)

if __name__=='__main__':
    main()
