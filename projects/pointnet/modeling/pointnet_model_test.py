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
from projects.pointnet.modeling.pointnet_model import PointNetModel


class PointNetModelTest(tf.test.TestCase):

    def test_pointnet_model_constructor(self):
        inputs = tf.keras.Input(shape=(1024, 3), batch_size=1)

        input_tnet = TransformNetModel(tf.keras.layers.InputSpec(shape=(1, 1024, 3, 1)), 3)
        feature_tnet = TransformNetModel(tf.keras.layers.InputSpec(shape=(1, 1024, 1, 64)), 64)
        point_net = PointNetModel(tf.keras.layers.InputSpec(shape=(1, 1024, 3)), 40, input_tnet, feature_tnet)

        features = point_net(inputs)

        self.assertEqual([1, 40], features['classes_score'].shape.as_list())


if __name__ == "__main__":
    tf.test.main()
