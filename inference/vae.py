import numpy as np
import tensorflow as tf

# Building the VAE model within a class

class ConvVAE(object):
    # Init all params and vars of ConvVAE class
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False, gpu_mode=False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse

        with tf.variable_scope('conv_vae', reuse=self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self._build_graph()
            else:
                tf.logging.info('Model using gpu.')
                self._build_graph()

        self.__init_session()

        def _build_graph(self):
            self.g = tf.Graph()
            with self.g.as_default():
                # Input image 64x64x3
                self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
                
                # Building the encoder part of the VAE
                #  relu conv 32x4, 64x4, 128x4, 256x4
                h = tf.layers.conv2d(self.x, 32, 4, strides=2, activation=tf.nn.relu, name='enc_conv1')
                h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name='enc_conv2')
                h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name='enc_conv3')
                h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name='enc_conv4')
                # extrude into vector
                #  If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. 
                # In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
                h = tf.reshape(h, [-1, 2*2*256])

                #  Build the "V" part of the VAE (the stochastic variation component)
                #  z_size is the size of the output vector of the variational part of the encoder
                # fc stands for fully connected layer
                self.mu = tf.layers.dense(h, self.z_size, name='enc_fc_mu') # mean "μ"
                self.logvar = tf.layers.dense(h, self.z_size, name='enc_fc_logvar') # log variance
                self.sigma = tf.exp(self.logvar / 2.0) # standard deviation (sigma) "σ"
                self.epsilon = tf.random_normal([self.batch_size, self.z_size]) # random normal distribution (epsilon) "ε"  N(0,1)
                self.z = self.mu + self.sigma * self.epsilon # z = μ + σε

                # Build the decoder part of the VAE
                h = tf.layers.dense(self.z, self.z_size, name='dec_fc')
                # extrude into vector of 1x1x1024
                h = tf.reshape(h, [-1, 1, 1, 1024])

                # Invert convolutions (begin deconvolution)
                h = tf.layers.cov2d_transpose(h, 128, 5, stride=2, activation=tf.nn.relu, name='dec_deconv1')
                h = tf.layers.cov2d_transpose(h, 64, 5, stride=2, activation=tf.nn.relu, name='dec_deconv2')
                h = tf.layers.cov2d_transpose(h, 32, 6, stride=2, activation=tf.nn.relu, name='dec_deconv3')
                # Obtain final output image 64x64x3
                self.y = tf.layers.cov2d_transpose(h, 3, 6, stride=2, activation=tf.nn.sigmoid, name='dec_deconv4')

                # Implement the training operations
                if self.is_training:
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)

                    # Reconstruction loss
                    self.r_loss = tf.reduce_sum(tf.square(self.x - self.y), reduction_indices=[1,2,3])
                    self.r_loss = tf.reduce_mean(self.r_loss)

                    # KL divergence loss
                    self.kl_loss = -0.5 * tf.reduce_sum((1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)), reduction_indices=1)
                    self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
                    self.kl_loss = tf.reduce_mean(self.kl_loss)

                    self.loss = self.r_loss + self.kl_loss

                    # Learning Rate
                    self.lr = tf.Variable(self.learning_rate, trainable=False)
                    self.optimizer = tf.train.AdamOptimizer(self.lr)

                    # Compute gradients
                    grads = self.optimizer.compute_gradients(self.loss)

                    self.train_op = self.optimizer.apply_gradient(grads, global_step=self.global_step, name='train_step')
                
                # Initialize all variables
                self.init = tf.global_variables_initializer()
                