# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utility functions for computing FID/Inception scores."""

import jax
import numpy as np
import six
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(inceptionv3=False):
  try:
    if inceptionv3:
      return tf.keras.models.load_model("./inception/v3")
    else:
      return tf.keras.models.load_model("./inception/v1")
    print("Load Inception from local")
  except:
    if inceptionv3:
      return tfhub.load(
        'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5')
    else:
      return tfhub.load(INCEPTION_TFHUB)
    print("Load Inception from TFHUB")



def get_dataset_stats(config):
  """Load the pre-computed dataset statistics."""
  if config.data.dataset == 'CIFAR10':
    filename = 'cifar10_32'
  elif config.data.dataset == 'CELEBA':
    filename = 'celeba'
  elif config.data.dataset == 'CelebAHQ':
    filename = 'celeba_hq_256'
  elif config.data.dataset == 'AFHQ-CAT':
    filename = 'afhq_cat_256'
  elif config.data.dataset == 'LSUN':
    filename = f'lsun_{config.data.category}_{config.data.image_size}'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')
  return filename


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Args:
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    inception_model: A model loaded from TFHub.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
  """Running the inception network. Assuming input is within [0, 255]."""
  if not inceptionv3:
    inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
  else:
    inputs = tf.cast(inputs, tf.float32) / 255.

  return tfgan.eval.run_classifier_fn(
    inputs,
    num_batches=num_batches,
    classifier_fn=classifier_fn_from_tfhub(None, inception_model),
    dtypes=_DEFAULT_DTYPES)


@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
  """Distribute the inception network computation to all available TPUs.

  Args:
    input_tensor: The input images. Assumed to be within [0, 255].
    inception_model: The inception network model obtained from `tfhub`.
    num_batches: The number of batches used for dividing the input.
    inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

  Returns:
    A dictionary with key `pool_3` and `logits`, representing the pool_3 and
      logits of the inception network respectively.
  """
  num_tpus = jax.local_device_count()
  input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  pool3 = []
  logits = [] if not inceptionv3 else None
  device_format = '/TPU:{}' if 'TPU' in str(jax.devices()[0]) else '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      tensor_on_device = tf.identity(tensor)
      res = run_inception_jit(
        tensor_on_device, inception_model, num_batches=num_batches,
        inceptionv3=inceptionv3)

      if not inceptionv3:
        pool3.append(res['pool_3'])
        logits.append(res['logits'])  # pytype: disable=attribute-error
      else:
        pool3.append(res)

  with tf.device('/CPU'):
    return {
      'pool_3': tf.concat(pool3, axis=0),
      'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
    }

def _calculate_fid(m, m_w, sigma, sigma_w):
  """Returns the Frechet distance given the sample mean and covariance."""
  # Find the Tr(sqrt(sigma sigma_w)) component of FID
  sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

  # Compute the two components of FID.

  # First the covariance component.
  # Here, note that trace(A + B) = trace(A) + trace(B)
  trace = tf.linalg.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

  # Next the distance between means.
  mean = tf.reduce_sum(input_tensor=tf.math.squared_difference(
      m, m_w))  # Equivalent to L2 but more stable.
  fid = trace + mean
  return fid

def trace_sqrt_product(sigma, sigma_v):
  """Find the trace of the positive sqrt of product of covariance matrices.

  '_symmetric_matrix_square_root' only works for symmetric matrices, so we
  cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
  ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

  Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
  We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
  Note the following properties:
  (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
     => eigenvalues(A A B B) = eigenvalues (A B B A)
  (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
     => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
  (iii) forall M: trace(M) = sum(eigenvalues(M))
     => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                   = sum(sqrt(eigenvalues(A B B A)))
                                   = sum(eigenvalues(sqrt(A B B A)))
                                   = trace(sqrt(A B B A))
                                   = trace(sqrt(A sigma_v A))
  A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
  use the _symmetric_matrix_square_root function to find the roots of these
  matrices.

  Args:
    sigma: a square, symmetric, real, positive semi-definite covariance matrix
    sigma_v: same as sigma

  Returns:
    The trace of the positive square root of sigma*sigma_v
  """

  # Note sqrt_sigma is called "A" in the proof above
  sqrt_sigma = _symmetric_matrix_square_root(sigma)

  # This is sqrt(A sigma_v A) above
  sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))

  return tf.linalg.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

def _symmetric_matrix_square_root(mat, eps=1e-10):
  """Compute square root of a symmetric matrix.

  Note that this is different from an elementwise square root. We want to
  compute M' where M' = sqrt(mat) such that M' * M' = mat.

  Also note that this method **only** works for symmetric matrices.

  Args:
    mat: Matrix to take the square root of.
    eps: Small epsilon such that any element less than eps will not be square
      rooted to guard against numerical instability.

  Returns:
    Matrix square root of mat.
  """
  # Unlike numpy, tensorflow's return order is (s, u, v)
  s, u, v = tf.linalg.svd(mat)
  # sqrt is unstable around 0, just use 0 in such case
  si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
  # Note that the v returned by Tensorflow is v = V
  # (when referencing the equation A = U S V^T)
  # This is unlike Numpy which returns v = V^T
  return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)
