import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1, name='sample_once')

class A3C_Network2(object):
    def __init__(self, ob_shape, action_num, factor=0.5, beta=0.01):
        #inference graph
        self.inputs = tf.placeholder(dtype=tf.uint8, shape=[None]+list(ob_shape), name='inputs')
        self.inputs_float = tf.cast(self.inputs, tf.float32)/255.0
        with tf.variable_scope('conv'):
            out = layers.convolution2d(inputs=self.inputs_float, num_outputs=16, kernel_size=8, 
                                        stride=4, padding="SAME", activation_fn=tf.nn.relu)
            out = layers.convolution2d(inputs=out, num_outputs=32, kernel_size=4, 
                                        stride=2, padding="SAME", activation_fn=tf.nn.relu)
            out = layers.flatten(inputs=out)
        with tf.variable_scope('fc'):
            fc_out = layers.fully_connected(inputs=out, num_outputs=256, activation_fn=tf.nn.relu)
        with tf.variable_scope('action_prob'):
            linear = layers.fully_connected(inputs=fc_out, num_outputs=action_num, activation_fn=None)
            self.act_Ps = tf.nn.softmax(linear)
        with tf.variable_scope('V-function'):
            self.Vs = tf.reshape(layers.fully_connected(inputs=fc_out, num_outputs=1, activation_fn=None), shape=[-1])

        #sample one action(tensor)
        self.sample_one_ac = categorical_sample_logits(self.act_Ps)[0]
        
        #loss computing graph
        self.action_samples = tf.placeholder(dtype=tf.int32, shape=[None])
        self.Returns = tf.placeholder(dtype=tf.float32, shape=[None])
        self.V_ts = tf.placeholder(dtype=tf.float32, shape=[None])
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        Advantage = self.Returns - self.V_ts

        sample_P = tf.reduce_sum(self.act_Ps * tf.one_hot(self.action_samples, action_num), axis=1)
        policy_loss = tf.reduce_sum(-tf.log(sample_P + 1e-7) * Advantage, name='policy_loss')
        tf.add_to_collection('losses', policy_loss)

        value_loss = tf.multiply(factor, tf.nn.l2_loss(self.Vs - self.Returns), name='value_loss')
        tf.add_to_collection('losses', value_loss)

        entropy_loss = tf.multiply(beta, tf.reduce_sum(self.act_Ps * tf.log(self.act_Ps + 1e-7)), name='entropy_loss')
        tf.add_to_collection('losses', entropy_loss)

        losses = tf.get_collection('losses')

        self.total_loss = tf.add_n(losses, name='total_loss')

        for l in losses+[self.total_loss]:
            tf.summary.scalar(l.op.name, l)

        #compute KL and entropy
        self.act_dist = tf.contrib.distributions.Categorical(probs=self.act_Ps)
        self.old_act_Ps = tf.placeholder(shape=[None, action_num], name='oldpolicy', dtype=tf.float32)
        old_act_dist = tf.contrib.distributions.Categorical(probs=self.old_act_Ps)

        self.KL = tf.reduce_mean(tf.contrib.distributions.kl_divergence(self.act_dist, old_act_dist))
        self.entropy = tf.reduce_mean(self.act_dist.entropy())

        #A3C_train_step
        #lr = tf.constant(0.001)
        tf.summary.scalar('learning_rate', self.learning_rate)

        opt = tf.train.AdamOptimizer(self.learning_rate)
        grads = opt.compute_gradients(self.total_loss)

        for var in tf.trainable_variables():
            if var == float('nan'):
                print(var, 'is nan!!!')
            tf.summary.histogram(var.op.name, var)
        
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        self.apply_gradients_op = opt.apply_gradients(grads)


class A3C_Network(object):
    def __init__(self, input_shape, output_dim, name='global', logdir=None):
        """Network structure is defined here
        Args:
            name (str): The name of scope
            input_shape (list): The shape of input image [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): directory to save summaries
                TODO: create a summary op
        """
        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.uint8, shape=[None, *input_shape], name="states")
            self.states = tf.cast(self.inputs, tf.float32)/255.0
            self.action_samples = tf.placeholder(tf.int32, shape=[None], name="actions")
            self.Returns = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.V_ts = tf.placeholder(tf.float32, shape=[None], name="v_preds")
            self.advantage = self.Returns - self.V_ts
            self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")

            action_onehot = tf.one_hot(self.action_samples, output_dim, name="action_onehot")
            net = self.states

            with tf.variable_scope("layer1"):
                net = tf.layers.conv2d(net,
                                       filters=16,
                                       kernel_size=(8, 8),
                                       strides=(4, 4),
                                       name="conv")
                net = tf.nn.relu(net, name="relu")

            with tf.variable_scope("layer2"):
                net = tf.layers.conv2d(net,
                                       filters=32,
                                       kernel_size=(4, 4),
                                       strides=(2, 2),
                                       name="conv")
                net = tf.nn.relu(net, name="relu")

            with tf.variable_scope("fc1"):
                net = tf.contrib.layers.flatten(net)
                net = tf.layers.dense(net, 256, name='dense')
                net = tf.nn.relu(net, name='relu')

            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(net, [0])
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            net = tf.reshape(lstm_outputs, [-1, 256])

            # actor network
            actions = tf.layers.dense(net, output_dim, name="final_fc")
            self.act_Ps = tf.nn.softmax(actions, name="action_prob")
            single_action_prob = tf.reduce_sum(self.act_Ps * action_onehot, axis=1)

            self.sample_one_ac = categorical_sample_logits(self.act_Ps)[0]

            entropy = - self.act_Ps * tf.log(self.act_Ps + 1e-7)
            entropy = tf.reduce_sum(entropy, axis=1)

            log_action_prob = tf.log(single_action_prob + 1e-7)
            maximize_objective = log_action_prob * self.advantage + entropy * 0.005
            self.actor_loss = - tf.reduce_mean(maximize_objective)

            # value network
            self.Vs = tf.reshape(tf.layers.dense(net, 1, name="values"), shape=[-1])
            self.value_loss = tf.nn.l2_loss(self.Vs - self.Returns)

            self.total_loss =  self.value_loss * 0.5 + self.actor_loss
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.gradients = tf.gradients(self.total_loss, var_list)

        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads = self.optimizer.apply_gradients(zip(grads, global_vars))
        
        '''
        self.gradients_placeholders = []

        for grad, var in self.gradients:
            if grad is not None:
                self.gradients_placeholders.append((tf.placeholder(var.dtype, shape=var.get_shape()), var))
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)
        '''
        if logdir:
            loss_summary = tf.summary.scalar(name+"total_loss", self.total_loss)
            actor_loss_summary = tf.summary.scalar(name+"actor_loss", self.actor_loss)
            value_loss_summary = tf.summary.scalar(name+"value_loss", self.value_loss)
            value_summary = tf.summary.histogram(name+"values", self.Vs)
            Returns_summary = tf.summary.histogram(name+"Returns", self.Returns)
            vpreds_summary = tf.summary.histogram(name+"vpreds", self.V_ts)
            summarys = [loss_summary, actor_loss_summary, value_loss_summary, value_summary, Returns_summary, vpreds_summary]

            for var in var_list:
                if var == float('nan'):
                    print(var, 'is nan!!!')
                summarys.append(tf.summary.histogram(var.op.name, var))
            '''
            for grad, var in self.optimizer.compute_gradients(self.total_loss):
                if grad is not None:
                    summarys.append(tf.summary.histogram(var.op.name + '/gradients', grad))
            '''
            self.summary_op = tf.summary.merge(summarys)
            self.summary_writer = tf.summary.FileWriter(logdir)
        
        #compute KL and entropy
        self.old_act_Ps = tf.placeholder(shape=[None, output_dim], name='oldpolicy', dtype=tf.float32)

        self.KL = tf.reduce_sum(self.old_act_Ps * (tf.log(self.old_act_Ps) - tf.log(self.act_Ps))) / tf.to_float(tf.shape(self.inputs)[0])
        self.entropy = tf.reduce_sum( - self.act_Ps * tf.log(self.act_Ps)) / tf.to_float(tf.shape(self.inputs)[0])