import numpy as np
import math

def ih33_downsample(im, factor):
  size_y = len(im)
  size_x = len(im[0])

  max_y = math.ceil(size_y / factor)
  max_x = math.ceil(size_x / factor)

  new_im = np.pad(im, ((0, max_y*factor-size_y), (0, max_x*factor-size_x)), 'edge')

  return new_im.reshape([max_y, factor, max_x, factor]).mean(3).mean(1)
