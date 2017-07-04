import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

class A3C_Network():
    def __init__(self, ob_shape, action_num, factor=0.5, beta=0.01):
        #inference graph
        self.inputs = tf.placeholder(dtype=tf.uint8, shape=[None]+list(ob_shape), name='inputs')
        self.inputs_float = tf.cast(self.inputs, tf.float32)/255.0
        with tf.variable_scope('conv'):
            out = layers.convolution2d(inputs=self.inputs_float, num_outputs=32, kernel_size=5, 
                                        stride=1, padding="SAME", activation_fn=tf.nn.relu)
            out = layers.max_pool2d(inputs=out, kernel_size=2, stride=2, padding='VALID')
            out = layers.convolution2d(inputs=out, num_outputs=64, kernel_size=5, 
                                        stride=1, padding="SAME", activation_fn=tf.nn.relu)
            out = layers.max_pool2d(inputs=out, kernel_size=2, stride=2, padding='VALID')
            out = layers.convolution2d(inputs=out, num_outputs=64, kernel_size=3, 
                                        stride=1, padding="SAME", activation_fn=tf.nn.relu)
            out = layers.max_pool2d(inputs=out, kernel_size=2, stride=2, padding='VALID')
            out = layers.flatten(inputs=out)
        with tf.variable_scope('fc'):
            fc_out = layers.fully_connected(inputs=out, num_outputs=256, activation_fn=tf.nn.relu)
        with tf.variable_scope('action_prob'):
            linear = layers.fully_connected(inputs=fc_out, num_outputs=action_num, activation_fn=None)
            self.act_Ps = tf.nn.softmax(linear)
        with tf.variable_scope('V-function'):
            self.Vs = tf.reshape(layers.fully_connected(inputs=fc_out, num_outputs=1, activation_fn=None), shape=[-1])
        
        #loss computing graph
        self.action_samples = tf.placeholder(dtype=tf.int32, shape=[None])
        self.Returns = tf.placeholder(dtype=tf.float32, shape=[None])
        self.V_ts = tf.placeholder(dtype=tf.float32, shape=[None])

        Advantage = self.Returns - self.V_ts

        sample_P = tf.reduce_sum(self.act_Ps * tf.one_hot(self.action_samples, action_num), 1)
        policy_loss = - tf.reduce_sum(tf.log(sample_P + 1e-6) * Advantage, name='policy_loss')
        tf.add_to_collection('losses', policy_loss)

        value_loss = tf.multiply(factor, tf.nn.l2_loss(self.Vs - self.Returns), name='value_loss')
        tf.add_to_collection('losses', value_loss)

        entropy_loss = tf.multiply(beta, tf.reduce_sum(self.act_Ps * tf.log(self.act_Ps)), name='entropy_loss')
        tf.add_to_collection('losses', entropy_loss)

        losses = tf.get_collection('losses')

        self.total_loss = tf.add_n(losses, name='total_loss')

        for l in losses+[self.total_loss]:
            tf.summary.scalar(l.op.name, l)

        #A3C_train_step
        lr = tf.constant(0.001)
        tf.summary.scalar('learning_rate', lr)

        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        grads = opt.compute_gradients(self.total_loss)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        self.apply_gradients_op = opt.apply_gradients(grads)


        

