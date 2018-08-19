from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg19
from keras import backend as K

def gram_matrix(x):
  """
  the gram matrix of an image tensor (feature-wise outer product)
  """
  assert K.ndim(x) == 3
  if K.image_data_format() == 'channels_first':
    features = K.batch_flatten(x)
  else:
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
  gram = K.dot(features, K.transpose(features))
  return gram


def style_loss(style, combination, height, width):
  """
  The "style loss" is designed to maintain the style of the
  reference image in the generated image. It is based on the
  gram matrices (which capture style) of feature maps from
  the style reference image and from the generated image.
  """
  assert K.ndim(style) == 3
  assert K.ndim(combination) == 3
  S = gram_matrix(style)
  C = gram_matrix(combination)
  channels = 3
  size = height * width
  return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))



def content_loss(base, combination):
  """
  An auxiliary loss function
  designed to maintain the "content" of the
  base image in the generated image
  """
  return K.sum(K.square(combination - base))

def total_variation_loss(x, height, width):
  """
  The 3rd loss function, total variation loss,
  designed to keep the generated image locally coherent
  """
  assert K.ndim(x) == 4
  if K.image_data_format() == 'channels_first':
    a = K.square(x[:, :, :height - 1, :width - 1] - x[:, :, 1:, :width - 1])
    b = K.square(x[:, :, :height - 1, :width - 1] - x[:, :, :height - 1, 1:])
  else:
    a = K.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
    b = K.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
  return K.sum(K.pow(a + b, 1.25))


def eval_loss_and_grads(x, f_outputs, height, width):
  if K.image_data_format() == 'channels_first':
    x = x.reshape((1, 3, height, width))
  else:
    x = x.reshape((1, height, width, 3))
  outs = f_outputs([x])
  loss_value = outs[0]
  if len(outs[1:]) == 1:
    grad_values = outs[1].flatten().astype('float64')
  else:
    grad_values = np.array(outs[1:]).flatten().astype('float64')
  return loss_value, grad_values

class Evaluator(object):
  """
  Makes it possible to compute loss and gradients in one pass
  while retrieving them via two separate functions,
  "loss" and "grads". This is done because scipy.optimize
  requires separate functions for loss and gradients,
  but computing them separately would be inefficient.
  """
  def __init__(self, f_outputs, height, width):
    self.loss_value = None
    self.grads_values = None
    self.f_outputs = f_outputs
    self.height = height
    self.width = width

  def loss(self, x):
    assert self.loss_value is None
    loss_value, grad_values = eval_loss_and_grads(x, self.f_outputs, self.height, self.width)
    self.loss_value = loss_value
    self.grad_values = grad_values
    return self.loss_value

  def grads(self, x):
    assert self.loss_value is not None
    grad_values = np.copy(self.grad_values)
    self.loss_value = None
    self.grad_values = None
    return grad_values

def preprocess_image(image_path, target_height, target_width):
  """
  util function to open, resize and format pictures into appropriate tensors
  """
  img = load_img(image_path, target_size=(target_height, target_width))
  img = img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = vgg19.preprocess_input(img)
  return img

def deprocess_image(x, height, width):
  """
  util function to convert a tensor into a valid image
  """
  if K.image_data_format() == 'channels_first':
    x = x.reshape((3, height, width))
    x = x.transpose((1, 2, 0))
  else:
    x = x.reshape((height, width, 3))
  # Remove zero-center by mean pixel
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  # 'BGR'->'RGB'
  x = x[:, :, ::-1]
  x = np.clip(x, 0, 255).astype('uint8')
  return x
