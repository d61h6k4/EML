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

import unittest
import jax

import jax.numpy as jnp
import haiku as hk

from projects.pointnet.modeling import pointnet


class MLPTest(unittest.TestCase):

    @hk.testing.transform_and_run
    def test_sanity_check(self):
        image = jnp.ones([2, 10, 3, 1])
        model = pointnet.MLP(10, 1, {})

        for is_training in (True, False):
            logits = model(image, is_training=is_training)
            self.assertEqual(logits.shape, (2, 10, 3, 10))


class FCTest(unittest.TestCase):

    @hk.testing.transform_and_run
    def test_sanity_check(self):
        image = jnp.ones([2, 10])
        model = pointnet.FC(64, {})

        for is_training in (True, False):
            logits = model(image, is_training=is_training)
            self.assertEqual(logits.shape, (2, 64))


class TNetTest(unittest.TestCase):

    @hk.testing.transform_and_run
    def test_input_transform_shape(self):
        image = jnp.ones([2, 10, 3, 1])
        model = pointnet.TNet(3, {})

        logits = model(image, is_training=True)
        self.assertEqual(logits.shape, (2, 3, 3))

    @hk.testing.transform_and_run
    def test_feature_transform_shape(self):
        image = jnp.ones([2, 10, 1, 64])
        model = pointnet.TNet(64, {})

        logits = model(image, is_training=True)
        self.assertEqual(logits.shape, (2, 64, 64))


class PointNetTest(unittest.TestCase):

    @hk.testing.transform_and_run
    def test_input_transform_shape(self):
        image = jnp.ones([2, 10, 3])
        rng = jax.random.PRNGKey(42)

        model = pointnet.PointNet(2, {})
        preds = model(image, is_training=True, rng=rng)
        self.assertEqual(preds["x"].shape, (2, 2))
        self.assertEqual(preds["feature_transformer"].shape, (2, 64, 64))


if __name__ == "__main__":
    unittest.main()
