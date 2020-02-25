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
  for idx in range(10):
    img = read_img(idx)
    pimg = preprocess(img)
    print("img {}".format(idx+1))
    print("standard: {}".format(compactness(img)))
    print("fft: {}".format(compactness(fft(pimg))))
    print("dwt: {}".format(compactness(dwt2(pimg))))

"""
Results:

img 1
standard: 0.2019127616233356
fft: 0.2309030722492008
dwt: 0.39531305096986324
img 2
standard: 0.21040045550961525
fft: 0.21525820534448153
dwt: 0.39549293335078034
img 3
standard: 0.220639825287268
fft: 0.22977126352921437
dwt: 0.31899602234009733
img 4
standard: 0.18546076348928414
fft: 0.21728786053782784
dwt: 0.31263225824068863
img 5
standard: 0.1975143954577983
fft: 0.20376302301387392
dwt: 0.3772254221203971
img 6
standard: 0.20049390732047445
fft: 0.20532424952765443
dwt: 0.2874949415019609
img 7
standard: 0.1961856152071466
fft: 0.2260021570663824
dwt: 0.35852858626437867
img 8
standard: 0.19539386965380873
fft: 0.20988639338881374
dwt: 0.32638964243640933
img 9
standard: 0.20624986138357143
fft: 0.21604106970523126
dwt: 0.2591676495291409
img 10
standard: 0.19186515690996164
fft: 0.21452417606810967
dwt: 0.28123594230635446
"""
