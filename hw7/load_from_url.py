import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np

def scrapeImg(src_url, n):
  response = urllib.request.urlopen(src_url)
  cnt = 0
  for line in response:
    if cnt >= n: break

    url = line.decode("utf-8").strip()
    print(url, cnt)
    try: 
      response = urllib.request.urlopen(url)
      if response.getcode() == 200:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('imgs/img{}.png'.format(cnt), image)
        cnt += 1
    except:
      continue 

if __name__ == '__main__':
  src_url = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02116738'
  scrapeImg(src_url, 10)
