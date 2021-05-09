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
"""PointNet."""

import tensorflow as tf

from typing import Mapping
from projects.pointnet.modeling.layers.nn_blocks import FC, MLP


@tf.keras.utils.register_keras_serializable(package='pointnet')
class PointNetModel(tf.keras.Model):
    """The PointNet model class."""

    def __init__(self,
                 input_specs: tf.keras.layers.InputSpec,
                 classes_num: int,
                 input_tnet: tf.keras.Model,
                 feature_tnet: tf.keras.Model,
                 kernel_initializer: str = 'GlorotNormal',
                 kernel_regularizer: tf.keras.regularizers.Regularizer = None,
                 bias_regularizer: tf.keras.regularizers.Regularizer = None,
                 activation: str = 'relu',
                 use_sync_bn: bool = False,
                 norm_momentum: float = 0.99,
                 norm_epsilon: float = 0.001,
                 **kwargs: object):
        """Constructor.

        Args:
            input_specs: A `tf.keras.layers.InputSpec` of the input tensor.
            classes_num: An `int` of number of classes in classification.
            kernel_initializer: A `str` of name of kernel initializer. Default is Xavier uniform.
            kernel_regularizer: A keras `Regularizer` for convolution's kernel.
            bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
            activation: A `srr` of name of activation function.
            use_sync_bn: If True, use synchronized batch normalization.
            norm_momentum: A `float` of normalization momentum for the moving average.
            norm_epsilon: A `float` added to variance to avoid dividing by zero.
        """
        # this attribute allows to set
        # attributes of the class before calling super
        # !WARNING: it disables autotracking of attributes
        self._self_setattr_tracking = False

        self._input_specs = input_specs
        self._classes_num = classes_num
        self._input_tnet = input_tnet
        self._feature_tnet = feature_tnet
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

        point_cloud = tf.keras.Input(shape=self._input_specs.shape[1:])
        endpoints = {}

        as_grey_image = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1))(point_cloud)
        project_matrix = input_tnet(as_grey_image)
        projected_point_cloud = tf.keras.layers.Lambda(lambda tm: tf.matmul(tm[0], tm[1]))(
            (point_cloud, project_matrix))

        projected_point_cloud_as_grey_image = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, -1))(
            projected_point_cloud)
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
                norm_epsilon=norm_epsilon)(projected_point_cloud_as_grey_image)
        x = MLP(num_outputs_channel=64,
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

        feature_project_matrix = feature_tnet(x)
        endpoints['feature_project_matrix'] = feature_project_matrix

        local_features = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=[2]))(x)
        local_features = tf.keras.layers.Lambda(lambda tm: tf.matmul(tm[0], tm[1]))(
            (local_features, feature_project_matrix))
        local_features = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, 2))(local_features)

        x = MLP(num_outputs_channel=64,
                kernel_size=(1, 1),
                padding='VALID',
                stride=(1, 1),
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activation=activation,
                use_sync_bn=use_sync_bn,
                norm_momentum=norm_momentum,
                norm_epsilon=norm_epsilon)(local_features)
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
        global_features = tf.keras.layers.MaxPool2D(pool_size=(num_points, 1), strides=(2, 2), padding='VALID')(x)

        # PointNet head
        x = tf.keras.layers.Reshape(target_shape=(-1,))(global_features)
        x = FC(512,
               kernel_initializer=kernel_initializer,
               activation=activation,
               use_sync_bn=use_sync_bn,
               norm_momentum=norm_momentum,
               norm_epsilon=norm_epsilon)(x)
        x = tf.keras.layers.Dropout(0.7)(x)
        x = FC(256,
               kernel_initializer=kernel_initializer,
               activation=activation,
               use_sync_bn=use_sync_bn,
               norm_momentum=norm_momentum,
               norm_epsilon=norm_epsilon)(x)
        x = tf.keras.layers.Dropout(0.7)(x)
        classes_score = FC(classes_num,
                           kernel_initializer=kernel_initializer,
                           activation=activation,
                           use_sync_bn=use_sync_bn,
                           norm_momentum=norm_momentum,
                           norm_epsilon=norm_epsilon)(x)
        endpoints['classes_score'] = classes_score

        super().__init__(inputs=point_cloud, outputs=endpoints, **kwargs)

    def get_config(self) -> Mapping[str, object]:
        config = {
            'input_specs': self._input_specs,
            'classes_num': self._classes_num,
            'input_tnet': self._input_tnet,
            'feature_tnet': self._feature_tnet,
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

    @property
    def checkpoint_items(self) -> Mapping[str, tf.keras.Model]:
        """Returns a dictionary of items to be additionally checkpointed."""
        return dict(input_tnet=self.input_tnet, feature_tnet=self.feature_tnet)

    @property
    def input_tnet(self) -> tf.keras.Model:
        return self._input_tnet

    @property
    def feature_tnet(self) -> tf.keras.Model:
        return self._feature_tnet
