import functools
import operator
from typing import Callable

from keras_cv.datasets.pascal_voc.segmentation import load as load_voc
import ml_collections
import numpy as np
import tensorflow as tf

from data import dataset as dataset_lib
import utils


@dataset_lib.DatasetRegistry.register('voc/2012_panoptic_segmentation')
# Define a Dataset class to use for finetuning.
class VocDataset:
  def __init__(self, config: ml_collections.ConfigDict):
    """Constructs the dataset."""
    self.config = config.dataset
    self.task_config = config.task

  def filter_example(self, example, training):
    # Filter out examples with no instances, to avoid error when converting
    # RaggedTensor to tensor: `Invalid first partition input. Tensor requires
    # at least one element.`
    print(example)
    return tf.shape(example['objects/bbox'])[0] > 0

  def _flatten_dims(self, example):
    """Flatten first 2 dims when batch is independently duplicated."""

    def flatten_first_2_dims(t):
      """Merge first 2 dims."""
      shape_list = t.shape.as_list()
      new_bsz = functools.reduce(operator.mul, shape_list[:2])
      out_shape = [new_bsz] + shape_list[2:]
      return tf.reshape(t, out_shape)

    return tf.nest.map_structure(flatten_first_2_dims, example)


  def extract(self, example, training):
    """Extracts needed features & annotations into a flat dictionary.

    Note: be consisous about 0 in label, which should probably reserved for
       special use (such as padding).

    Args:
      example: `dict` of raw features.
      training: `bool` of training vs eval mode.

    Returns:
      example: `dict` of relevant features and labels
    """
    # These features are needed by the object detection task.
    features = {
        'image': tf.image.convert_image_dtype(
            example['image'], tf.float32),
        'label_map': tf.concat(
            # [tf.image.convert_image_dtype(example['class_segmentation'], tf.int32),
            #  tf.image.convert_image_dtype(example['object_segmentation'], tf.int32)],
          [tf.cast(example['class_segmentation'], tf.int32),
           tf.cast(example['object_segmentation'], tf.int32)],
          axis=-1),
        'image/id': 0, # dummy int.
    }
    # for RGB data, otherwise channel dim stays None in non-eager mode which leads to problems
    features['image'].set_shape([None, None, 3])

    # bbox = decode_utils.decode_boxes(example)
    bbox = utils.tf_float32(example['objects/bbox'])
    labels = example['objects/label']

    scale = 1. / utils.tf_float32(tf.shape(features['image'])[:2])

    features.update({
        'label': labels,
        'bbox': utils.scale_points(bbox, scale),
    })

    print("ext", features['image'].shape, features['label_map'].shape, example['class_segmentation'].shape, features['label_map'].dtype)
    return features

  def pipeline(self,
               process_single_example: Callable[[tf.data.Dataset, int, bool],
                                                tf.data.Dataset],
               global_batch_size: int, training: bool):
    """Data pipeline from name to preprocessed examples.

    Args:
      process_single_example: a function that takes single example dataset and
        returns processed example dataset.
      global_batch_size: global batch size.
      training: training vs eval mode.

    Returns:
      An input_fn which generates a tf.data.Dataset instance.
    """
    config = self.config
    def input_fn(input_context):
      dataset = load_voc("train" if training else "eval")
      # dataset = self.load_dataset(input_context, training)
      if config.cache_dataset:
        dataset = dataset.cache()

      if input_context:
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        # Sharding is not neccesary for TFDS given read_config above.
        # dataset = dataset.shard(input_context.num_input_pipelines,
        #                         input_context.input_pipeline_id)
      else:
        batch_size = global_batch_size

      if training:
        options = tf.data.Options()
        options.deterministic = False
        options.experimental_slack = True
        dataset = dataset.with_options(options)
        buffer_size = config.get('buffer_size', 0)
        if buffer_size <= 0:
          buffer_size = 10 * batch_size
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat()

      dataset = dataset.filter(
          lambda x: self.filter_example(x, training)
      ).map(
          lambda x: self.extract(x, training),
          num_parallel_calls=tf.data.experimental.AUTOTUNE
      )
      if process_single_example:
        dataset = process_single_example(
            dataset, config.batch_duplicates, training)

      # TODO(b/181662974): Revert this and support non-even batch sizes.
      # dataset = dataset.batch(batch_size, drop_remainder=training)
      dataset = dataset.padded_batch(batch_size, drop_remainder=True)
      if config.batch_duplicates > 1 and training:
        dataset = dataset.map(self._flatten_dims,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      return dataset

    return input_fn

  @property
  def num_train_examples(self):
    return self.config.train_num_examples

  @property
  def num_eval_examples(self):
    return self.config.eval_num_examples if not self.task_config.get(
        'unbatch', False) else None
