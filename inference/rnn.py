# Building the MDN-RNN model

import numpy as np
import tensorflow as tf

class MDNRNN(object):
    # Init all params and vars of MDNRNN class
    # hps is a dictionary of hyperparameters
    def __init__(self, hps, reuse=False, gpu_mode=False):
        self.hps = hps

        with tf.variable_scope('mdn_rnn', reuse=reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self.g = tf.Graph()
                    with self.g.as_default():
                        self.build_model()

            else:
                tf.logging.info('Model using gpu.')
                self.g = tf.Graph()
                with self.g.as_default():
                    self.build_model()

        self.__init_session()

    def build_model(self):
        # Building the RNN
        RNN = "RNN"
        self.num_mixture = self.hps.num_mixture
        ACTION_DIMENSIONS = 3
        KMIX = self.num_mixture # number of guassian mixtures in the MDN
        INWIDTH = self.hps.input_seq_width
        OUTWIDTH = self.hps.output_seq_width
        LENGTH = self.hps.max_seq_len
        if self.hps.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Implement LSTM cell inside the RNN with optional dropout
        cell_fn =  tf.contrib.rnn.LayerNormBasicLSTMCell
        use_layer_norm = False if self.hps.use_layer_norm == 0 else True
        # Recurrent Dropout: This type of dropout is applied to the recurrent connections of the RNN.
        # It drops some of the connections between the hidden states of the RNN across time steps
        # This helps to prevent the model from overfitting to the temporal dependencies in the training data.
        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
        if use_recurrent_dropout:
            cell = cell_fn(self.hps.rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else:
            cell = cell_fn(self.hps.rnn_size, layer_norm=use_layer_norm)

        # Input Dropout: This type of dropout is applied to the input layer of the RNN.
        # It drops some of the input units, which helps the model to learn more robust features.
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True
        if use_input_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)

        # Output Dropout: This type of dropout is applied to the output layer of the RNN.
        # It drops some of the output units, which helps to prevent the model from overfitting to the specific outputs during training.
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True
        if use_output_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)

        self.cell = cell

        self.sequence_lengths = LENGTH
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, INWIDTH])
        self.output_x = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, OUTWIDTH]) 

        # Create dynamic RNN
        actual_input_x = self.input_x # copy to prevent modifying the input tensor
        self.initial_state = cell.zero_state(batch_size=self.hps.batch_size, dtype=tf.float32) # Initial state of the RNN filled with zeros

        # create tensor biases and weights
        NOUT = OUTWIDTH * KMIX * ACTION_DIMENSIONS # number of biases
        with tf.variable_scope(RNN):
            output_w = tf.get_variable('output_w', shape=[self.hps.rnn_size, NOUT]) # dimension of the output weights (2d tensor)
            output_b = tf.get_variable('output_b', [NOUT]) # dimension of the output biases (1d tensor)
        
        # ouput shape is a 3d tensor with shape [batch_size, max_seq_len (number of timesteps), encoded_elements_per_timestep (rnn_size)]
        output, last_State = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=actual_input_x,
                                               initial_state=self.initial_state,
                                               dtype=tf.float32,
                                               swap_memory=True,
                                               scope=RNN)
        
        #  NOTE: RNN output produces a deterministic prediction of the next state,
        #  whereas the Mixture Density Network (MDN) layer produces a distribution of possible next states (stochastic prediction)


        #  Building the MDN layer
        #  The MDN layer is a fully connected layer that takes the output of the RNN and produces the stochastic prediction of the next state.
        #  Uses Guassian Mixture based on KMIX number of mixtures

        #  Flatten the output tensor to a 2D tensor so that it may be multiplied by the output weights
        #  -1 means "unspecified" dimension (using first 2 dimensions of input tensor) and rnn_size is the number of encoded elements per timestep
        #  For instance, if output WAS a 3d with [100, 1000, 256], it BECOMES a 2d tensor with [100000, 256]
        output = tf.reshape(output, [-1, self.hps.rnn_size]) 

        # Multiply the output tensor by the output weights and add the output biases
        output = tf.nn.xw_plus_b(x=output, weights=output_w, biases=output_b)

        # Reshape the output of the MDN layer to a different 2d tensor
        # Divde the output tensor into (KMIX number of mixtures multiplied by ACTION_DIMENSIONS) elements and multiply the result by the first column of the tensor
        output = tf.reshape(output, [-1, KMIX * ACTION_DIMENSIONS])
        self.final_state = last_State

        # function inside the function
        def get_mdn_coef(output):
            #  split the output into 3 equal parts
            logmix, mean, logstd = tf.split(output, 3, 1)
            logmix -= tf.reduce_logsumexp(logmix, 1, keepdims=True)
            return logmix, mean, logstd
        
        out_logmix, out_mean, out_logstd = get_mdn_coef(output)
        self.out_logmix = out_logmix
        self.out_mean = out_mean
        self.out_logstd = out_logstd

        # Implement the training operation
        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        def tf_lognormal(y, mean, logstd):
            return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI
        def get_lossfun(logmix, mean, logstd, y):
            v = logmix + tf_lognormal(y, mean, logstd)
            v = tf.rdeduce_logsumexp(v, 1, keepdims=True)
            return -tf.reduce_mean(v)
        flat_target_data = tf.reshape(self.output_x,[-1, 1])

        # get the actual loss
        lossfunc = get_lossfun(out_logmix, out_mean, out_logstd, flat_target_data)
        self.cost = tf.reduce_mean(lossfunc)
        