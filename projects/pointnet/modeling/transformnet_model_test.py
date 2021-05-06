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

import tensorflow as tf

from projects.pointnet.modeling.transformnet_model import TransformNetModel


class TransformNetModelTest(tf.test.TestCase):

    def test_input_transformnet_model_constructor(self):
        inputs = tf.keras.Input(shape=(1024, 3, 1), batch_size=1)

        tnet = TransformNetModel(tf.keras.layers.InputSpec(shape=(1, 1024, 3, 1)), 3)
        features = tnet(inputs, training=True)

        self.assertEqual([1, 3, 3], features.shape.as_list())

    def test_feature_transformnet_model_constructor(self):
        inputs = tf.keras.Input(shape=(1024, 1, 64), batch_size=1)

        tnet = TransformNetModel(tf.keras.layers.InputSpec(shape=(1, 1024, 1, 64)), 64)
        features = tnet(inputs, training=True)

        self.assertEqual([1, 64, 64], features.shape.as_list())


if __name__ == "__main__":
    tf.test.main()
