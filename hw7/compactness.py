import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def read_img(idx):
  p = 'imgs/img{}.png'.format(idx)
  return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(float)

def preprocess(img):
  [w, h] = img.shape
  aw, ah = 1, 1
  while aw < w: aw = aw * 2
  while ah < h: ah = ah * 2
  s = max(aw, ah)
  return np.pad(img, ((0, aw-w), (0, ah-h)))

def compactness(img):
  v = img.reshape(img.size, 1)
  abs_v = [abs(x) for x in v]
  mn, mx = min(abs_v), max(abs_v)
  """ normalize to [0, 1] and compute pmf """
  normalized_v = [(x-mn)/(mx-mn) for x in abs_v]
  hist, _ = np.histogram(normalized_v, img.size)
  pmf = [x/img.size for x in hist]
  #plt.plot(pmf)
  #plt.show()
  return 1/stats.entropy(pmf)
  
def fft(img):
  ft = np.fft.fft2(img)
  d = dropLowest(ft, 20)
  #plt.imshow(np.vectorize(lambda x: np.log(abs(x)))(d))
  #plt.imshow(np.vectorize(abs)(np.fft.ifft2(d)), cmap='gray')
  #plt.show()
  return d

# from lecture scripts
def dwt2(im):
  """Apply wavelet decomposition to image."""
  if im.shape[0] != im.shape[1] or np.mod(np.log2(im.shape[0]), 1) != 0:
      raise ValueError('Input image must be square, with side length a power of two.')

  def h0(I):
      """Apply filter [1, 1]' and downsample."""
      return (I[0::2, :] + I[1::2, :]) / np.sqrt(2)

  def h1(I):
      """Apply filter [1, -1]' and downsample."""
      return (I[0::2, :] - I[1::2, :]) / np.sqrt(2)

  g00 = im
  wt = np.zeros(im.shape)
  for i in range(int(np.log2(im.shape[0]))):
      g11 = h1(h1(g00).T).T
      g10 = h1(h0(g00).T).T
      g01 = h0(h1(g00).T).T
      g00 = h0(h0(g00).T).T
      wt[:int(im.shape[0] / (2**i)), :int(im.shape[0] / (2**i))] = np.vstack((
          np.hstack((g00, g01)),
          np.hstack((g10, g11))
      ))

  return dropLowest(wt, 20)

def dropLowest(img, percentile):
  abs_img = [abs(x) for x in img.reshape(img.size, 1)]
  criteria = np.percentile(abs_img, percentile)
  def drop(v): 
    return v if abs(v) > criteria else 0 
  return np.vectorize(drop)(img)
 
if __name__ == '__main__':
  res = []
  for idx in range(10):
    print("{}/{}".format(idx, 10))
    img = read_img(idx)
    pimg = preprocess(img)
    res.append([
        compactness(img),
        compactness(fft(pimg)),
        compactness(dwt2(pimg))
    ])

  def print_header():
    print("std|fft|dwt2")

  def print_row(r):
    fmt = list(map(lambda x: "{0:.6f}".format(x), r))
    print("{}|{}|{}".format(*fmt))

  print("results")
  print_header()
  for r in res:
    print_row(r)

  print()
   
  print("mean")
  print_header()
  print_row(np.mean(res, axis=0))

  print()

  print("standard deviation")
  print_header()
  print_row(np.std(res, axis=0))

"""
prints:

0/10
1/10
2/10
3/10
4/10
5/10
6/10
7/10
8/10
9/10
results
std|fft|dwt2
0.201913|0.230903|0.395313
0.210400|0.215258|0.395493
0.220640|0.229771|0.318996
0.185461|0.217288|0.312632
0.197514|0.203763|0.377225
0.200494|0.205324|0.287495
0.196186|0.226002|0.358529
0.195394|0.209886|0.326390
0.206250|0.216041|0.259168
0.191865|0.214524|0.281236

mean
std|fft|dwt2
0.200612|0.216876|0.331248

standard deviation
std|fft|dwt2
0.009449|0.008998|0.046067
"""


