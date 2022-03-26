# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""General utils (used across different modules)."""

import copy
import functools
import json
import logging
import math
import operator
import os
import matplotlib
import matplotlib.cm
import ml_collections
import numpy as np

import vocab
import tensorflow as tf


def json_serializable(val):
  try:
    json.dumps(val)
    return True
  except TypeError:
    return False


def tf_float32(t):
  return tf.cast(t, tf.float32)


def flatten_batch_dims(t, out_rank):
  """Merge first few dims to have out_rank."""
  if t.shape.rank == out_rank:
    return t
  if t.shape.rank < out_rank:
    raise ValueError('Tensor has rank %d. Expected at least %d' %
                     (t.shape.rank, out_rank))
  shape_list = shape_as_list(t)
  in_rank = len(shape_list)
  split = in_rank - out_rank + 1
  inner_dims = shape_list[split:]
  new_bsz = functools.reduce(operator.mul, shape_list[:split])
  out_shape = [new_bsz] + inner_dims
  return tf.reshape(t, out_shape)


def flatten_non_batch_dims(t, out_rank):
  """Merge last few dims to have out_rank."""
  if t.shape.rank == out_rank:
    return t
  if t.shape.rank < out_rank:
    raise ValueError('Tensor has rank %d. Expected at least %d' %
                     (t.shape.rank, out_rank))
  shape_list = shape_as_list(t)
  split = out_rank - 1
  inner_dims = shape_list[:split]
  new_last_dim = functools.reduce(operator.mul, shape_list[split:])
  out_shape = inner_dims + [new_last_dim]
  return tf.reshape(t, out_shape)


def tile_along_batch(t, factor):
  """Tile tensor in the first/batch dimension."""
  if factor == 1:
    return t
  t = tf.expand_dims(t, 1)
  multiples = [1] * t.shape.rank
  multiples[1] = factor
  t = tf.tile(t, multiples)
  shape = shape_as_list(t)
  return tf.reshape(t, [shape[0] * shape[1]] + shape[2:])


def shape_as_list(t):
  # Assumes rank of `t` is statically known.
  shape = t.shape.as_list()
  dynamic_shape = tf.shape(t)
  return [
      shape[i] if shape[i] is not None else dynamic_shape[i]
      for i in range(len(shape))
  ]


def pad_to_max_len(data, max_len, dim):
  """Pad the data tensor to max length on dim."""
  shape = shape_as_list(data)
  padding_shape, new_shape = copy.copy(shape), copy.copy(shape)
  padding_shape[dim] = max_len - padding_shape[dim]
  new_shape[dim] = max_len
  paddings = tf.zeros(padding_shape, dtype=data.dtype)
  return tf.reshape(tf.concat([data, paddings], axis=dim), new_shape)


def quantize(coordinates, bins):
  """Quantization of (normalized) coordinates in [0, 1]."""
  coordinates = tf.cast(tf.round(coordinates * (bins - 1)), tf.int64)
  coordinates = tf.clip_by_value(coordinates, 0, bins - 1)
  return coordinates


def dequantize(boxes, bins):
  """Dequantization of discrete tokens of coordinates in [0, bins-1]."""
  boxes = tf.cast(boxes, tf.float32)
  boxes = boxes / (bins - 1)
  return boxes


def yx2xy(seq):
  x = np.asarray(seq[1::2]).reshape([-1, 1])
  y = np.asarray(seq[::2]).reshape([-1, 1])
  return np.concatenate([x, y], axis=-1).reshape([-1]).tolist()


def scale_points(points, scale):
  """Scales points.

  Args:
    points: Tensor with shape [num_points * 2], [batch, num_points * 2] or
      [batch, instances, num_points * 2] where points are organized in
      (y, x) format.
    scale: Tensor with shape [2] or [batch, 2].

  Returns:
    Tensor with same shape as points.
  """
  points_orig = points
  orig_shape = tf.shape(points)
  coords_len = points.shape[-1]
  if points.shape.rank == 1:
    points = tf.reshape(points, [coords_len // 2, 2])
  elif points.shape.rank == 2:
    points = tf.reshape(points, [-1, coords_len // 2, 2])
  else:
    points = tf.reshape(points, [-1, orig_shape[1], coords_len // 2, 2])
    scale = tf.expand_dims(scale, -2)
  points = points * scale
  points = tf.reshape(points, orig_shape)
  points = preserve_reserved_tokens(points, points_orig)
  return points


def preserve_reserved_tokens(points, points_orig):
  """Preserve reserved tokens in points according to points_orig."""
  return replace_reserved_tokens(points, points_orig, dict(zip(vocab.FLOATS,
                                                               vocab.FLOATS)))


def replace_reserved_tokens(seq, ref_seq, replacements):
  for key, replacement in replacements.items():
    seq = tf.where(
        tf.equal(ref_seq, key), tf.constant(replacement, seq.dtype), seq)
  return seq


def restore_from_checkpoint(model_dir, except_partial, **kwargs):
  """Restores the latest ckpt.

  Args:
    model_dir: directory of checkpoint to be restored.
    except_partial: whether to allow partially restoring the checkpoint.
    **kwargs: arguments for `tf.train.Checkpoint` so it knows what to restore,
      e.g., `model=model, global_step=global_step, optimizer=optimizer`.

  Returns:
    latest_ckpt: The full path to the latest checkpoint or None if no
      checkpoint was found.
    checkpoint object
    verify_restored: function for verification
  """
  verify_restored = None
  checkpoint = tf.train.Checkpoint(**kwargs)
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
    if except_partial:
      status = checkpoint.restore(latest_ckpt).expect_partial()
    else:
      status = checkpoint.restore(latest_ckpt)
    verify_restored = status.assert_consumed
  return latest_ckpt, checkpoint, verify_restored


def check_checkpoint_restored(strict_verifiers, loose_verifiers=()):
  """Verification after model variables built."""
  strict_verifiers_new = []
  for strict_verifier in strict_verifiers:  # Stop exp from running.
    if strict_verifier:
      strict_verifier()
      strict_verifier = None
    strict_verifiers_new.append(strict_verifier)
  loose_verifiers_new = []
  for loose_verifier in loose_verifiers:  # Give warning in the log.
    if loose_verifier:
      try:
        loose_verifier()
      except AssertionError as e:
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logging.info('+++++++++++Checkpoint verification msg begin+++++++++++')
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logging.info(e)
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logging.info('+++++++++++Checkpoint verification msg ends+++++++++++')
        logging.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      loose_verifier = None
    loose_verifiers_new.append(loose_verifier)
  return strict_verifiers_new, loose_verifiers_new


def build_strategy(use_tpu, master):
  """Returns a tf.distribute.Strategy."""
  if use_tpu:
    cluster = tf.distribute.cluster_resolver.TPUClusterResolver(master)
    tf.config.experimental_connect_to_cluster(cluster)
    topology = tf.tpu.experimental.initialize_tpu_system(cluster)
    logging.info('Topology:')
    logging.info('num_tasks: %d', topology.num_tasks)
    logging.info('num_tpus_per_task: %d', topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(cluster)
  else:  # For (multiple) GPUs.
    cross_device_ops = None  # tf.distribute.NcclAllReduce() by default
    # if the default cross_device_ops fails, try either of the following two
    # by uncommenting it.
    # cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
    # cross_device_ops = tf.distribute.ReductionToOneDevice()
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
    logging.info('Running using MirroredStrategy on %d replicas',
                 strategy.num_replicas_in_sync)
  return strategy


def get_train_steps(dataset, train_steps, train_epochs, train_batch_size):
  """Determine the number of training steps."""
  num_train_examples = dataset.num_train_examples
  return train_steps or (
      num_train_examples * train_epochs // train_batch_size + 1)


def get_eval_steps(dataset, eval_steps, eval_batch_size):
  """Determine the number of eval steps."""
  num_eval_examples = dataset.num_eval_examples
  if num_eval_examples % eval_batch_size != 0:
    raise ValueError('Only divisible eval batch sizes are currently supported.')
  # TODO(b/181662974): Revert this and support non-even batch sizes.
  # return eval_steps or int(
  #     math.ceil(num_eval_examples / eval_batch_size))
  return eval_steps or (int(
      math.floor(num_eval_examples /
                 eval_batch_size)) if num_eval_examples else None)


def get_checkpoint_steps(dataset, checkpoint_steps,
                         checkpoint_epochs, train_batch_size):
  """Determine the number of checkpoint steps."""
  num_train_examples = dataset.num_train_examples
  return checkpoint_steps or checkpoint_epochs * int(
      round(num_train_examples / train_batch_size))


def count_params(model, verbose=True):
  """Count parameters in `tf.keras.models.Model`."""
  if verbose:
    logging.info('Trainable variables:')
  total_params = 0
  for var in model.trainable_variables:
    if verbose:
      logging.info('%s\t%s', var.name, var.shape)
    total_params += np.prod(var.shape)
  total_params = total_params / 1e6
  if verbose:
    logging.info('Total number of parameters: {:.2f}M'.format(total_params))
  return total_params


def merge_list_of_dict(list_of_dict):
  """Merge a list of dictionary (with shared keys) into a single dictionary."""
  if len(list_of_dict) == 1:
    return list_of_dict[0]
  dict_new = {}
  for key in list_of_dict[0].keys():
    dict_new[key] = tf.stack([x[key] for x in list_of_dict])
  return dict_new


def get_and_log_config(config, config_override, model_dir, training):
  """Get the config and log it."""
  config = update_config_from_string(config, config_override)
  config.model_dir = model_dir
  logging.info('Config: %s', config)

  # Log config to the model directory for training jobs.
  config_filepath = os.path.join(model_dir, 'config.json')
  if training and not tf.io.gfile.exists(config_filepath):
    tf.io.gfile.makedirs(model_dir)
    with tf.io.gfile.GFile(config_filepath, 'w') as f:
      f.write(config.to_json(indent=2, sort_keys=True))

  return config


def update_config_from_flattened_key(config, key, value):

  def is_float(v) -> bool:
    try:
      float(v)
      return True
    except ValueError:
      return False

  def is_int(v) -> bool:
    try:
      int(v)
      return True
    except ValueError:
      return False

  def to_bool(v) -> bool:
    if v == 'True' or v == 'true':
      return True
    elif v == 'False' or v == 'false':
      return False
    else:
      raise ValueError()

  config = copy.deepcopy(config)
  k_list = key.split('.', 1)
  k = k_list[0]
  if is_int(k):
    k = int(k)

  if len(k_list) > 1:
    config[k] = update_config_from_flattened_key(config[k], k_list[1], value)
  else:
    if k in config and isinstance(config.get(k), bool):
      value = to_bool(value)
    elif config.get(k) is not None:
      value = type(config.get(k))(value)
    elif is_int(value):
      value = int(value)
    elif is_float(value):
      value = float(value)
    config.update_from_flattened_dict({k: value})
  return config


def update_config_from_string(config: ml_collections.ConfigDict,
                              config_override: str):
  """Update a config from a string such as 'a=3,b=foo'.

  Args:
    config: a ml_collections config dict.
    config_override: a string in the format of 'k1=v1,k2=v2'. The keys can be
      flattened keys such as 'a.b.c', or for list items, 'd.0.e' etc.

  Returns:
    The updated config.
  """
  if config_override:
    for o in config_override.split(','):
      k, v = o.split('=')
      config = update_config_from_flattened_key(config, k, v)
  return config


def colorize(images, vmin=None, vmax=None, cmap=None):
  """Convert grayscaled images into into colored images.

  Args:
    images: grayscale image tensor of shape (h, w) or (bsz, h, w).
    vmin: the minimum value of the range used for normalization.
      (Default: minimum of images)
    vmax: the maximum value of the range used for normalization.
      (Default: maximum of images)
    cmap: a valid cmap named for use with matplotlib's `get_cmap`.
      (Default: 'gray')

  Returns:
    a colored image tensor of shape (h, w) or (bsz, h, w).
  """
  vmin = tf.reduce_min(images) if vmin is None else vmin
  vmax = tf.reduce_max(images) if vmax is None else vmax
  images = (images - vmin) / (vmax - vmin)
  images = tf.squeeze(images)  # squeeze last dim if it exists

  indices = tf.cast(tf.round(images * 255), tf.int32)
  cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'viridis')
  colors = tf.constant(cm.colors, dtype=tf.float32)
  return tf.gather(colors, indices)
