import tensorflow as tf
import numpy as np
import os
from six.moves import xrange
import bc_train
import bc_test
import argparse
from easydict import EasyDict as edict

train_path = '/tmp/DAgger'
max_iter = 10
def da_train(envname):
    #train using DAgger method
    
    train_dir = os.path.join(train_path, envname)
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    if tf.gfile.Exists('/tmp/DAgger_data'):
        tf.gfile.DeleteRecursively('/tmp/DAgger_data')
    tf.gfile.MakeDirs('/tmp/DAgger_data')

    expert_policy_file = os.path.join('./experts', '%s.pkl' % envname)
    args_dict = {'expert_policy_file': expert_policy_file,
            'envname': envname,
            'mode': 'expert',
            'render': False,
            'max_timesteps': 1000,
            'num_rollouts': 100}
    args = edict(args_dict)
    #get some basic data using 'expert' mode
    bc_test.test(args)
    args.mode = 'DAgger'
    total_step = bc_train.train(envname, 'DAgger', once=True)
    #DAgger method
    for i in xrange(max_iter):
        print('train_iter:', i)
        bc_test.test(args)
        total_step = bc_train.train(envname, 'DAgger', once=False, total_step=total_step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    args = parser.parse_args()
    
    if not tf.gfile.Exists(train_path):
        tf.gfile.MakeDirs(train_path)
    print('Begin to train %s using DAgger' % args.envname)
    da_train(args.envname)

if __name__ == '__main__':
    main()
