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
"""
Implementaiton of PoinNet.
title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation}
author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J}
"""

from typing import Optional, Sequence, Union, Mapping

import jax

import jax.numpy as jnp
import haiku as hk


class MLP(hk.Module):
    """
    MLP from PoinNet paper.

    Conv2D -> BN -> Activation.
    """

    def __init__(self,
                 output_channels: int,
                 kernel_shape: Union[int, Sequence[int]],
                 bn_config: Mapping[str, Union[float, bool]],
                 stride: Union[int, Sequence[int]] = 1,
                 padding: str = 'SAME',
                 name: Optional[str] = None):
        """Constructor.
        """
        super().__init__(name=name)

        bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault("decay_rate", 0.999)
        self._bn_config = bn_config
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._padding = padding

    def __call__(self, inputs: jnp.ndarray, is_training: bool, test_local_stats: bool = False) -> jnp.ndarray:
        x = inputs

        x = hk.Conv2D(output_channels=self._output_channels,
                      kernel_shape=self._kernel_shape,
                      stride=self._stride,
                      with_bias=False,
                      padding=self._padding,
                      name="conv2d_0")(x)
        x = hk.BatchNorm(name="batchnorm_0", **self._bn_config)(x, is_training, test_local_stats=test_local_stats)
        x = jax.nn.relu(x)

        return x


class FC(hk.Module):
    """
    FC from PoinNet paper.

    Linear -> BN -> Activation.
    """

    def __init__(self, output_size: int, bn_config: Mapping[str, Union[float, bool]], name: Optional[str] = None):
        """Constructor.
        """
        super().__init__(name=name)

        bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault("decay_rate", 0.999)
        self._bn_config = bn_config
        self._output_size = output_size

    def __call__(self, inputs: jnp.ndarray, is_training: bool, test_local_stats: bool = False) -> jnp.ndarray:
        x = inputs

        x = hk.Linear(output_size=self._output_size, with_bias=False, name="linear_0")(x)
        x = hk.BatchNorm(name="batchnorm_0", **self._bn_config)(x, is_training, test_local_stats=test_local_stats)
        x = jax.nn.relu(x)

        return x


class TNet(hk.Module):
    """TNet (Transform Net) from the PointNet paper.

    TNet learns transformation that transform given 
    cloud points to normalize.

    Cloud points may describe transformed (e.g rotated)
    object, TNet aim is learn how compansate the rotatation
    and always preset all objects in uniform way.
    
    We are going to use TNet for input (Nx3x1)
    and for features (Nx1x64) and for input we would like
    to have transform matrix as 3x3 and for features 64x64.
    From input estimate the output matrix dimensions is possible
    but not convinent, so the caller needs to provide the dimension
    as an input parameter (k).
    """

    def __init__(self, k: int, bn_config: Mapping[str, Union[float, bool]], name: Optional[str] = None):
        """Constructor."""
        super().__init__(name=name)

        bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault("decay_rate", 0.999)

        self._bn_config = bn_config
        self._k = k

    def __call__(self, inputs: jnp.ndarray, is_training: bool, test_local_stats: bool = False) -> jnp.ndarray:
        _, num_points, points_dim, _ = inputs.shape

        x = inputs
        x = MLP(output_channels=64,
                kernel_shape=(1, points_dim),
                bn_config=self._bn_config,
                padding='VALID',
                name="tmlp_64_1x3")(x, is_training, test_local_stats=test_local_stats)
        x = MLP(output_channels=128, kernel_shape=1, bn_config=self._bn_config, padding='VALID',
                name="tmlp_128_1x1")(x, is_training, test_local_stats=test_local_stats)
        x = MLP(output_channels=1024, kernel_shape=1, bn_config=self._bn_config, padding='VALID',
                name="tmlp_1024_1x1")(x, is_training, test_local_stats=test_local_stats)
        x = hk.MaxPool(window_shape=(num_points, 1, 1), strides=(2, 2, 1), padding='VALID', name="tmaxpool2d")(x)
        x = hk.Flatten(name="treshape")(x)
        x = FC(512, bn_config=self._bn_config, name="tfc_512")(x, is_training, test_local_stats=test_local_stats)
        x = FC(256, bn_config=self._bn_config, name="tfc_256")(x, is_training, test_local_stats=test_local_stats)
        x = hk.Linear(self._k * self._k,
                      with_bias=True,
                      w_init=hk.initializers.Identity(gain=0),
                      b_init=hk.initializers.Constant(jnp.eye(self._k, self._k).flatten()),
                      name="tlinear")(x)
        x = hk.Reshape(output_shape=(self._k, self._k), name="treshape_matrix")(x)

        return x


class PointNet(hk.Module):
    """PointNet implementation."""

    def __init__(self, class_num: int, bn_config: Mapping[str, Union[float, bool]], name: Optional[str] = None):
        """Constructor."""
        super().__init__(name=name)

        bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault("decay_rate", 0.999)

        self._bn_config = bn_config
        self._class_num = class_num

    def __call__(self,
                 inputs: jnp.ndarray,
                 is_training: bool,
                 test_local_stats: bool = False) -> Mapping[str, jnp.ndarray]:

        _, num_points, points_dim = inputs.shape

        x = inputs
        as_grey_image = jnp.expand_dims(x, -1)
        input_transformer = TNet(3, self._bn_config, name="InputTNet")(as_grey_image,
                                                                       is_training,
                                                                       test_local_stats=test_local_stats)
        x = jax.vmap(jnp.matmul)(x, input_transformer)
        x = jnp.expand_dims(x, -1)
        x = MLP(64, kernel_shape=(1, points_dim), bn_config=self._bn_config, padding='VALID',
                name="pmlp_input_64")(x, is_training, test_local_stats=test_local_stats)
        x = MLP(64, kernel_shape=(1, 1), bn_config=self._bn_config, padding='VALID',
                name="pmlp_input_64_1")(x, is_training, test_local_stats=test_local_stats)
        feature_transformer = TNet(64, self._bn_config, name="FeatureTNet")(x,
                                                                            is_training,
                                                                            test_local_stats=test_local_stats)
        x = jnp.squeeze(x, 2)
        x = jax.vmap(jnp.matmul)(x, feature_transformer)
        x = jnp.expand_dims(x, 2)

        x = MLP(64, kernel_shape=(1, 1), bn_config=self._bn_config, padding='VALID',
                name="pmlp_localfeature_64")(x, is_training, test_local_stats=test_local_stats)
        x = MLP(128, kernel_shape=(1, 1), bn_config=self._bn_config, padding='VALID',
                name="pmlp_localfeature_128")(x, is_training, test_local_stats=test_local_stats)
        x = MLP(1024, kernel_shape=(1, 1), bn_config=self._bn_config, padding='VALID',
                name="pmlp_localfeature_1024")(x, is_training, test_local_stats=test_local_stats)
        x = hk.MaxPool(window_shape=(num_points, 1, 1), strides=(2, 2, 1), padding='VALID', name="pmaxpool2d")(x)

        # PointNet classification head
        x = hk.Flatten(name="preshape_clshead")(x)
        x = FC(512, bn_config=self._bn_config, name="pfc_512")(x, is_training, test_local_stats=test_local_stats)
        x = hk.dropout(hk.next_rng_key(), 0.3, x)
        x = FC(256, bn_config=self._bn_config, name="pfc_256")(x, is_training, test_local_stats=test_local_stats)
        x = hk.dropout(hk.next_rng_key(), 0.3, x)
        x = hk.Linear(self._class_num, with_bias=True)(x)

        return {"logits": x, "feature_transformer": feature_transformer}
