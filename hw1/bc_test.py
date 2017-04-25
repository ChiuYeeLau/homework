import tensorflow as tf
import numpy as np
import tf_util
import gym
import bc_eval
import load_policy

def test(args):
    #this function is used to 1.generate expert data for DAgger method
    #2.test the behavioral cloning or DAgger result.

    with tf.Graph().as_default() as g:
        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(args.expert_policy_file)
        print('loaded and built')
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        returns = []
        observations = []
        actions = []
        obs = env.reset()
        action = env.action_space.sample()
        print(action.shape)
        actionspace = action.shape[0]

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            if args.mode != 'expert':
                observ, logits = bc_eval.perdict(sess, obs[None,:].shape, actionspace, args.envname, args.mode)
            for i in range(args.num_rollouts):
                if args.render:
                    print('iter', i)
                elif i%10==0:
                    print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                
                while not done:
                    expert_action = policy_fn(obs[None,:])
                    if args.mode == 'expert':
                        action_tmp = expert_action
                    else:
                        action_tmp = np.array(sess.run(logits, feed_dict={observ: obs[None,:]}))
                        action_tmp = np.reshape(action_tmp, action.shape)
                    observations.append(obs)
                    actions.append(expert_action)
                    obs, r, done, _ = env.step(action_tmp)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    #if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        import os
        import pickle
        
        s = os.path.join('/tmp/DAgger_data', '%s.pkl' % args.envname)
        if os.path.exists(s):
            with open(s, 'rb') as f:
                d = pickle.load(f)
            pre_observations = list(d['observations'])
            pre_actions = list(d['actions'])
            observations += pre_observations
            actions += pre_actions
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        print(np.array(observations).shape)
        print(np.array(actions).shape)
        with open(s,'wb') as f:
            pickle.dump(expert_data, f)
            print('save expert_data into file: %s' % s)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('mode', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    
    test(args)

if __name__ == '__main__':
    main()