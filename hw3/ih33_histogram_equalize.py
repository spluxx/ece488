import numpy as np
import math

def ih33_histogram_equalize(img):
	L = 256
	numPixels = img.size

	hist, _ = np.histogram(img, bins=L, range=(0,L))
	pn = hist/numPixels
	cum_pn = np.cumsum(pn)

	def transform(v):
		return math.floor((L-1)*cum_pn[math.floor(v)])

	v_transform = np.vectorize(transform)

	return v_transform(img), v_transform

	

