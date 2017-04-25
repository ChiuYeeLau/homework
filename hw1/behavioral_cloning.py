import pickle
import tensorflow as tf
import numpy as np
import os 

MOVING_AVERAGE_DECAY = 0.9
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100000
BATCH_SIZE = 128
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  

def input(envname, mode):
    if mode=='DAgger':
        filename = os.path.join('/tmp', 'DAgger_data','%s.pkl' % envname)
    else:
        filename = os.path.join('.', 'expert_data','%s.pkl' % envname)
    if os.path.exists(filename):
        print('data file of %s is found in %s' % (envname, filename))
    else:
        raise ValueError("the data file of %s is not found" % envname)
    with open(filename, 'rb') as f:
        d = pickle.load(f)
    observations = d['observations']
    actions = d['actions']
    return observations, actions

def input_batch(observations, actions, batch_size=BATCH_SIZE):
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    obs_batch, actions_batch = tf.train.shuffle_batch(
        [observations, actions], batch_size=batch_size, capacity=capacity,
         min_after_dequeue=min_after_dequeue, enqueue_many=True)
    obs_batch = tf.cast(obs_batch, tf.float32)
    actions_batch = tf.cast(actions_batch, tf.float32)
    return obs_batch, actions_batch


def inference(observations, actionspace, mode):
    feature = observations.get_shape()[1].value
    with tf.variable_scope('hide1') as scope:
        hide1 = tf.contrib.layers.fully_connected(inputs=observations, num_outputs=10, activation_fn=tf.nn.relu,
        #normalizer_fn=tf.contrib.layers.batch_norm,
        #normalizer_params={'decay': 0.9, 'is_training': mode == 'train', 'scale': True},
        weights_initializer=tf.truncated_normal_initializer(stddev=1.0/(feature**0.5), dtype=tf.float32),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.004))

    with tf.variable_scope('hide2') as scope:
        hide2 = tf.contrib.layers.fully_connected(inputs=hide1, num_outputs=10, activation_fn=tf.nn.relu, 
        #normalizer_fn=tf.contrib.layers.batch_norm,
        #normalizer_params={'decay': 0.9, 'is_training': mode == 'train', 'scale': True},
        weights_initializer=tf.truncated_normal_initializer(stddev=1.0/(10**0.5), dtype=tf.float32),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.004))

    with tf.variable_scope('linear') as scope:
        linear = tf.contrib.layers.fully_connected(inputs=hide2, num_outputs=actionspace, activation_fn=None, 
        weights_initializer=tf.truncated_normal_initializer(stddev=1.0/(10**0.5), dtype=tf.float32))

    return linear

def loss(linear, actions):
    l2_loss = tf.nn.l2_loss(linear - actions) / BATCH_SIZE
    tf.add_to_collection('losses', l2_loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('total_loss', total_loss)
    return total_loss

def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    apply_gradients_op = tf.contrib.layers.optimize_loss(
        loss=total_loss,
        global_step=global_step,
        learning_rate=0.1,
        optimizer='Adam')

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
  
    with tf.control_dependencies([apply_gradients_op]):
        train_op = tf.no_op(name='train')

    return train_op