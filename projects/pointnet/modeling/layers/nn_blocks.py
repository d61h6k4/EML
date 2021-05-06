# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper layers from PointNet paper."""

from typing import Dict, Optional, Tuple

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='pointnet')
class MLP(tf.keras.layers.Layer):
    """MLP class from PointNet paper implemented as Conv2d -> BN -> Activation."""

    def __init__(self,
                 num_outputs_channel: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: str = 'SAME',
                 kernel_initializer: str = 'GlorotNormal',
                 kernel_regularizer: tf.keras.regularizers.Regularizer = None,
                 bias_regularizer: tf.keras.regularizers.Regularizer = None,
                 activation: str = 'relu',
                 use_sync_bn: bool = False,
                 norm_momentum: float = 0.99,
                 norm_epsilon: float = 0.001,
                 **kwargs):
        """Constructor.

        Args:
          num_outputs_channel: An `int` of outputs channel. Usually called filter in Conv2d.
          kernel_size: A `Tuple` of 2 integers, kernel size of convolution.
          stride: A `tuple` of 2 integers, stride of convolution.
          padding: A `str` of padding of convolution.
          kernel_initializer: A `str` of name of kernel initializer. Default is Xavier uniform.
          kernel_regularizer: A keras `Regularizer` for convolution's kernel.
          bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
          activation: A `srr` of name of activation function.
          use_sync_bn: If True, use synchronized batch normalization.
          norm_momentum: A `float` of normalization momentum for the moving average.
          norm_epsilon: A `float` added to variance to avoid dividing by zero.
        """
        super().__init__(**kwargs)

        self._num_outputs_channel = num_outputs_channel
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activation = activation
        self._use_sync_bn = use_sync_bn
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon

        if self._use_sync_bn:
            self._norm = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            self._norm = tf.keras.layers.BatchNormalization

        if tf.keras.backend.image_data_format() == 'channels_last':
            self._bn_axis = -1
        else:
            self._bn_axis = 1

        self._activation_fn = tf.keras.layers.Activation(self._activation)

    def build(self, input_shape: tf.TensorShape):
        """Build the layer"""
        self._conv1 = tf.keras.layers.Conv2D(filters=self._num_outputs_channel,
                                             kernel_size=self._kernel_size,
                                             strides=self._stride,
                                             padding=self._padding,
                                             use_bias=False,
                                             kernel_initializer=self._kernel_initializer,
                                             kernel_regularizer=self._kernel_regularizer,
                                             bias_regularizer=self._bias_regularizer)
        self._norm1 = self._norm(axis=self._bn_axis, momentum=self._norm_momentum, epsilon=self._norm_epsilon)
        super().build(input_shape)

    def get_config(self) -> Dict:
        config = {
            'num_outputs_channel': self._num_outputs_channel,
            'kernel_size': self._kernel_size,
            'stride': self._stride,
            'padding': self._padding,
            'kernel_initializer': self._kernel_initializer,
            'kernel_regularizer': self._kernel_regularizer,
            'bias_regularizer': self._bias_regularizer,
            'activation': self._activation,
            'use_sync_bn': self._use_sync_bn,
            'norm_momentum': self._norm_momentum,
            'norm_epsilon': self._norm_epsilon
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self,
             inputs: tf.types.experimental.TensorLike,
             training: Optional[bool] = None) -> tf.types.experimental.TensorLike:
        x = self._conv1(inputs)
        x = self._norm1(x, training=training)
        return self._activation_fn(x)


@tf.keras.utils.register_keras_serializable(package='pointnet')
class FC(tf.keras.layers.Layer):
    """FC class from PointNet paper implemented as Dense -> BN -> Activation."""

    def __init__(self,
                 num_outputs: int,
                 kernel_initializer: str = 'GlorotNormal',
                 kernel_regularizer: tf.keras.regularizers.Regularizer = None,
                 bias_regularizer: tf.keras.regularizers.Regularizer = None,
                 activation: str = 'relu',
                 use_sync_bn: bool = False,
                 norm_momentum: float = 0.99,
                 norm_epsilon: float = 0.001,
                 **kwargs):
        """Constructor.

        Args:
          num_outputs: An `int` of outputs channel. 
          kernel_initializer: A `str` of name of kernel initializer. Default is Xavier uniform.
          kernel_regularizer: A keras `Regularizer` for Dense's kernel.
          bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Dense.
          activation: A `srr` of name of activation function.
          use_sync_bn: If True, use synchronized batch normalization.
          norm_momentum: A `float` of normalization momentum for the moving average.
          norm_epsilon: A `float` added to variance to avoid dividing by zero.
        """
        super().__init__(**kwargs)

        self._num_outputs = num_outputs
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activation = activation
        self._use_sync_bn = use_sync_bn
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon

        if self._use_sync_bn:
            self._norm = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            self._norm = tf.keras.layers.BatchNormalization

        if tf.keras.backend.image_data_format() == 'channels_last':
            self._bn_axis = -1
        else:
            self._bn_axis = 1

        self._activation_fn = tf.keras.layers.Activation(self._activation)

    def build(self, input_shape: tf.TensorShape):
        """Build the layer"""
        self._dense1 = tf.keras.layers.Dense(units=self._num_outputs,
                                             activation=None,
                                             kernel_initializer=self._kernel_initializer,
                                             kernel_regularizer=self._kernel_regularizer,
                                             bias_regularizer=self._bias_regularizer)
        self._norm1 = self._norm(axis=self._bn_axis, momentum=self._norm_momentum, epsilon=self._norm_epsilon)
        super().build(input_shape)

    def get_config(self) -> Dict:
        config = {
            'num_outputs': self._num_outputs,
            'kernel_initializer': self._kernel_initializer,
            'kernel_regularizer': self._kernel_regularizer,
            'bias_regularizer': self._bias_regularizer,
            'activation': self._activation,
            'use_sync_bn': self._use_sync_bn,
            'norm_momentum': self._norm_momentum,
            'norm_epsilon': self._norm_epsilon
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self,
             inputs: tf.types.experimental.TensorLike,
             training: Optional[bool] = None) -> tf.types.experimental.TensorLike:
        x = self._dense1(inputs)
        x = self._norm1(x, training=training)
        return self._activation_fn(x)
