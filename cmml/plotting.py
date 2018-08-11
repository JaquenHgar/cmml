"""
Utiltily functions for plotting things related to machine learning.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(metrics_dict, figsize=(8, 5)):
  """
  Plots multiple metrics in a single plot. Can for example be used to plot
  the training and test accuracy progress during training.

  Parameters
  ----------
  metrics_dict : dict
    Dict of metrics name (str) and metrics values (list of float). The names
    will be used in the legend. The values will be used as y values in the
    plot, the x values will be in range(0, len(y)).
  figsize : tuple
    Size of figure 
  """
  plt.figure(figsize=figsize)
  legend = []
  for metric_name in metrics_dict:
    ys = metrics_dict[metric_name]
    xs = np.arange(0, len(ys))
    plt.plot(xs, ys)
    legend.append(metric_name)
  plt.legend(legend)

def show_greyscale_images(images, labels, indices=np.arange(0, 6), cols=6):
  """
  Plots several images in a grid using matplotlib. Useful for having a quick
  look at images in image classification tasks (e.g. MNIST).

  Parameters
  ----------
  images : array_like
    Array of images with shape: (N, width, height)
  indices : range, list, array-like
    Indices of images that should be shown.
  cols : int
    Number of columns in the output grid (default: 6).
  """
  plot_num = 1
  rows = len(indices) / cols + 1
  plt.figure(figsize=(cols, rows+1))
  for i in indices:
    plt.subplot(rows, cols, plot_num)
    plt.grid(b=False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(images[i], cmap="gray")
    plt.title(f"Label {labels[i]}") 
    
    plot_num = plot_num + 1
