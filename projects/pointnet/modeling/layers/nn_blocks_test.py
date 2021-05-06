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
"""Tests for mlp."""

import tensorflow as tf
from projects.pointnet.modeling.layers.nn_blocks import MLP, FC


class MLPTest(tf.test.TestCase):

    def test_mlp_layer_creation(self):
        # 10 points in 3d represented as a grey image
        inputs = tf.keras.Input(shape=[10, 3, 1], batch_size=1)
        mlp_layer = MLP(64, (1, 3))
        features = mlp_layer(inputs)

        self.assertEqual([1, 10, 3, 64], features.shape.as_list())


class FCTest(tf.test.TestCase):

    def test_mlp_layer_creation(self):
        # 10 points in 3d represented as a grey image
        inputs = tf.keras.Input(shape=[10], batch_size=1)
        fc_layer = FC(64)
        features = fc_layer(inputs)

        self.assertEqual([1, 64], features.shape.as_list())


if __name__ == "__main__":
    tf.test.main()
