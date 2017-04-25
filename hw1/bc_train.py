import numpy as np
import sys
import os
import argparse
from six.moves import xrange
import tensorflow as tf
import behavioral_cloning as bc

BATCH_SIZE = 128
max_steps = 10000
def train(envname, mode, once=False, total_step=0):
    #train the NN for behavioral cloning or Dagger.
    #the first time train the Dagger, restore the checkpoint of behavioral cloning.
    #thus, DAgger is based on the behavioral cloning at the first time and let once=True at the first time.
    train_path = os.path.join('/tmp', mode)
    train_dir = os.path.join(train_path, envname)
    if tf.gfile.Exists(train_dir) and mode=='behavioral_cloning':
        tf.gfile.DeleteRecursively(train_dir)
    if not tf.gfile.Exists(train_dir):
        tf.gfile.MakeDirs(train_dir)
    

    with tf.Graph().as_default():
        
        global_step = tf.Variable(0, trainable=False)

        observations, actions = bc.input(envname, mode)
        obs_batch, actions_batch = bc.input_batch(observations, actions, BATCH_SIZE)
        actions_batch = tf.reshape(actions_batch, [BATCH_SIZE, -1])
        actionspace = actions_batch.get_shape()[1].value

        linear = bc.inference(obs_batch, actionspace, mode='train')

        loss = bc.loss(linear, actions_batch)

        train_op = bc.train(loss, global_step)

        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False))
        sess.run(init)

        if mode == 'DAgger' and once==False:
            ckpt = tf.train.get_checkpoint_state(train_dir)
            print('reuse the checkpoint in %s' % train_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
        elif once==True:
            ckpt = tf.train.get_checkpoint_state(os.path.join('/tmp/behavioral_cloning',envname))
            print('reuse the checkpoint in %s' % os.path.join('/tmp/behavioral_cloning',envname))
            saver.restore(sess, ckpt.model_checkpoint_path)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        threads = tf.train.start_queue_runners(sess=sess)

        for step in xrange(max_steps):
            _, loss_value = sess.run([train_op, loss])
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            virtual_step = step + total_step
            if step % 100 == 0:
                print(virtual_step, max_steps)
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, virtual_step)

            if step % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=virtual_step)
        total_step += max_steps
    return total_step



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    args = parser.parse_args()
    
    train_path = '/tmp/behavioral_cloning'
    if not tf.gfile.Exists(train_path):
        tf.gfile.MakeDirs(train_path)
    print('Begin to train %s using behavioral cloning' % args.envname)
    train(args.envname, mode='behavioral_cloning')

if __name__ == '__main__':
    main()