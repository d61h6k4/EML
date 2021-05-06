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
"""T-Net from PointNet paper."""

import tensorflow as tf

from typing import Dict
from projects.pointnet.modeling.layers.nn_blocks import FC, MLP


@tf.keras.utils.register_keras_serializable(package="pointnet")
class TransformNetModel(tf.keras.Model):
    """T-Net model class."""

    def __init__(self,
                 input_specs: tf.keras.layers.InputSpec,
                 k: int,
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
            input_specs: A `tf.keras.layers.InputSpec` of the input tensor.
            k: An `int` of output dimension size. (Output: lxk)
            kernel_initializer: A `str` of name of kernel initializer. Default is Xavier uniform.
            kernel_regularizer: A keras `Regularizer` for convolution's kernel.
            bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
            activation: A `srr` of name of activation function.
            use_sync_bn: If True, use synchronized batch normalization.
            norm_momentum: A `float` of normalization momentum for the moving average.
            norm_epsilon: A `float` added to variance to avoid dividing by zero.
        """
        self._input_specs = input_specs
        self._k = k
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activation = activation
        self._use_sync_bn = use_sync_bn
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon

        assert self._input_specs.shape[1] is not None
        assert self._input_specs.shape[2] is not None

        num_points: int = self._input_specs.shape[1]
        points_dim: int = self._input_specs.shape[2]
        assert num_points == 1024
        assert points_dim == 3
        inputs = tf.keras.Input(shape=self._input_specs.shape[1:])

        x = MLP(num_outputs_channel=64,
                kernel_size=(1, points_dim),
                padding='VALID',
                stride=(1, 1),
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activation=activation,
                use_sync_bn=use_sync_bn,
                norm_momentum=norm_momentum,
                norm_epsilon=norm_epsilon)(inputs)
        x = MLP(num_outputs_channel=128,
                kernel_size=(1, 1),
                padding='VALID',
                stride=(1, 1),
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activation=activation,
                use_sync_bn=use_sync_bn,
                norm_momentum=norm_momentum,
                norm_epsilon=norm_epsilon)(x)
        x = MLP(num_outputs_channel=1024,
                kernel_size=(1, 1),
                padding='VALID',
                stride=(1, 1),
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activation=activation,
                use_sync_bn=use_sync_bn,
                norm_momentum=norm_momentum,
                norm_epsilon=norm_epsilon)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(num_points, 1), strides=(2, 2), padding='VALID')(x)
        x = tf.keras.layers.Reshape(target_shape=(-1,))(x)
        x = FC(512,
               kernel_initializer=kernel_initializer,
               activation=activation,
               use_sync_bn=use_sync_bn,
               norm_momentum=norm_momentum,
               norm_epsilon=norm_epsilon)(x)
        x = FC(256,
               kernel_initializer=kernel_initializer,
               activation=activation,
               use_sync_bn=use_sync_bn,
               norm_momentum=norm_momentum,
               norm_epsilon=norm_epsilon)(x)
        x = tf.keras.layers.Dense(points_dim * self._k,
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=tf.keras.initializers.Constant(
                                      value=tf.reshape(tf.eye(num_rows=points_dim, num_columns=self._k), [-1])))(x)
        x = tf.keras.layers.Reshape(target_shape=(points_dim, self._k))(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

    def get_config(self) -> Dict:
        config = {
            'input_specs': self._input_specs,
            'k': self._k,
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
