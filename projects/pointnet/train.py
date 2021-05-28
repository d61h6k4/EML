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

from typing import Text, Mapping, Tuple, Iterable, NamedTuple, Generator

import argparse
import functools
import jax
import optax
import logging
import rich.logging
import rich.progress
import tree

import numpy as np
import jax.numpy as jnp
import haiku as hk
import tensorflow as tf
import tensorflow_datasets as tfds
import projects.pointnet.modeling.pointnet_model as pointnet

import datasets.modelnet

# Hyper parameters.
TRAIN_BATCH_SIZE = 32
TRAIN_INIT_RANDOM_SEED = 42
TRAIN_LR_WARMUP_EPOCHS = 5
TRAIN_EPOCHS = 50
TRAIN_LR_INIT = 0.32
TRAIN_NUM_EXAMPLES = 3991
TRAIN_SMOOTHING = 0
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_LOG_EVERY = 200

Batch = Mapping[Text, np.ndarray]
Scalars = Mapping[Text, jnp.ndarray]


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_batch_size', type=int, default=TRAIN_BATCH_SIZE, help="Train batch size.")
    parser.add_argument('--train_init_random_seed',
                        type=int,
                        default=TRAIN_INIT_RANDOM_SEED,
                        help="Train init random seed value.")
    parser.add_argument('--train_lr_warmup_epochs',
                        type=int,
                        default=TRAIN_LR_WARMUP_EPOCHS,
                        help="Train learning rate warmup epochs num.")
    parser.add_argument('--train_epochs', type=int, default=TRAIN_EPOCHS, help="Train epochs num.")
    parser.add_argument('--train_lr_init', type=float, default=TRAIN_LR_INIT, help="Train learning rate initial value.")
    parser.add_argument('--train_log_every',
                        type=int,
                        default=TRAIN_LOG_EVERY,
                        help="Controls the frequency of log display updating.")

    args = parser.parse_args()

    global_variables = globals()
    global_variables['TRAIN_BATCH_SIZE'] = args.train_batch_size
    global_variables['TRAIN_INIT_RANDOM_SEED'] = args.train_init_random_seed
    global_variables['TRAIN_LR_WARMUP_EPOCHS'] = args.train_lr_warmup_epochs
    global_variables['TRAIN_EPOCHS'] = args.train_epochs
    global_variables['TRAIN_LR_INIT'] = args.train_lr_init
    global_variables['TRAIN_LOG_EVERY'] = args.train_log_every


def load(split: tfds.Split, *, is_training: bool, batch_size: int) -> Generator[Batch, None, None]:
    """Load the given split of ModelNet10."""
    ds = tfds.load("ModelNet/10", split=split)
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    if is_training:
        options.experimental_deterministic = False
    ds = ds.with_options(options)

    if is_training:
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=10 * batch_size, seed=0)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    yield from tfds.as_numpy(ds)


def _forward(batch: Batch, is_training: bool) -> jnp.ndarray:
    """Forward application of PointNet."""
    point_cloud = batch["point_cloud"]
    net = pointnet.PointNet(10, bn_config={'decay_rate': 0.9}, name="PointNet")
    return net(point_cloud, is_training=is_training)


forward = hk.transform_with_state(_forward)


def lr_schedule(step: jnp.ndarray) -> jnp.ndarray:
    """Cosine learning rate schedule."""
    steps_per_epoch = TRAIN_NUM_EXAMPLES / TRAIN_BATCH_SIZE
    warmup_steps = TRAIN_LR_WARMUP_EPOCHS * steps_per_epoch
    training_steps = TRAIN_EPOCHS * steps_per_epoch

    lr = TRAIN_LR_INIT * TRAIN_BATCH_SIZE / 256
    scaled_step = jnp.maximum(step - warmup_steps, 0) / (training_steps - warmup_steps)
    lr *= 0.5 * (1.0 + jnp.cos(jnp.pi * scaled_step))
    if warmup_steps:
        lr *= jnp.minimum(step / warmup_steps, 1.0)

    return lr


def make_optimizer() -> optax.GradientTransformation:
    """SGD with nesterov momentum and a custom lr schedule."""
    return optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(lr_schedule), optax.scale(-1))


def l2_loss(params: Iterable[jnp.ndarray]) -> jnp.ndarray:
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


def orthogonal_loss(m: jnp.ndarray) -> jnp.ndarray:
    k = m.shape[1]
    return 0.5 * jnp.sum(jnp.square(jnp.eye(k, k) - jnp.matmul(m.transpose(), m)))


def loss_fn(params: hk.Params, state: hk.State, batch: Batch,
            rng: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, hk.State]]:
    """Compute a regularized loss for the given batch."""
    preds, new_state = forward.apply(params, state, rng, batch, is_training=True)
    logits = preds["logits"]
    feature_transformer = preds["feature_transformer"]

    labels = jax.nn.one_hot(batch['label'], 10)
    if TRAIN_SMOOTHING:
        labels = optax.smooth_labels(labels, TRAIN_SMOOTHING)
    class_loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
    feature_transformer_loss = jax.vmap(orthogonal_loss)(feature_transformer).mean()

    l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params) if 'batchnorm' not in mod_name]

    total_loss = class_loss + feature_transformer_loss + TRAIN_WEIGHT_DECAY * l2_loss(l2_params)
    return total_loss, (total_loss, class_loss, feature_transformer_loss, new_state)


def train_step(train_state: TrainState, batch: Batch, rng: jnp.ndarray) -> Tuple[TrainState, Scalars]:
    """Applies an update to parameters and returns new state"""
    params, state, opt_state = train_state
    grads, (loss, class_loss, ftran_loss, new_state) = jax.grad(loss_fn, has_aux=True)(params, state, batch, rng)

    # Compute and apply updates via our optimizer
    updates, new_opt_state = make_optimizer().update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    # Scalars to log
    scalars = {'train_loss': loss, 'class_loss': class_loss, 'feature_transformer_loss': ftran_loss}

    train_state = TrainState(new_params, new_state, new_opt_state)
    return train_state, scalars


def initial_state(rng: jnp.ndarray, batch: Batch) -> TrainState:
    """Compute the initial network state."""
    params, state = forward.init(rng, batch, is_training=True)
    opt_state = make_optimizer().init(params)

    return TrainState(params, state, opt_state)


def main():
    parse_args()

    train_dataset = load(tfds.Split.TRAIN, is_training=True, batch_size=TRAIN_BATCH_SIZE)

    rng = jax.random.PRNGKey(TRAIN_INIT_RANDOM_SEED)
    # Initialization requires an example input.
    batch = next(train_dataset)
    train_state = initial_state(rng, batch)

    num_train_steps = int(TRAIN_EPOCHS * TRAIN_NUM_EXAMPLES / TRAIN_BATCH_SIZE)
    for step_num in rich.progress.track(range(num_train_steps), description='Trainig...'):
        # Take a single training step
        with jax.profiler.StepTraceAnnotation('train', step_num=step_num):
            batch = next(train_dataset)
            current_rng, rng = jax.random.split(rng)
            train_state, train_scalars = train_step(train_state, batch, current_rng)

        # Log progress at fixed intervals.
        if step_num and step_num % TRAIN_LOG_EVERY == 0:
            train_scalars = jax.tree_map(lambda v: np.mean(v).item(), jax.device_get(train_scalars))
            logging.info('[Train %s/%s] %s', step_num, num_train_steps, train_scalars)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        handlers=[rich.logging.RichHandler()])
    main()
