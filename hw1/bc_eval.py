from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import os
import numpy as np
import tensorflow as tf

import behavioral_cloning as bc


def perdict(sess, shape, actionspace, envname, mode):
    checkpoint_dir = os.path.join('/tmp', mode, envname)
    observ = tf.placeholder(tf.float32, shape=shape, name='observ')

    logits = bc.inference(observ, actionspace, mode='eval')

    saver = tf.train.Saver(tf.trainable_variables())

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print('load the checkpoint: %s' % checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    return observ, logits
