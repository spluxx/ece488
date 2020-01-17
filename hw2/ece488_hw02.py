import numpy as np
from math import pi, sqrt, floor

def capture(fcn, T, D, g, t_0=0):
	sample_rate = 1e9
	lumens_per_watt = 683		
	capacitance = 1

	area = pi*(D/2)**2
	t = np.arange(t_0, t_0+T, 1/sample_rate)
	lux = np.array(list(map(fcn, t)))
	energy = np.trapz(lux, t)*area/lumens_per_watt
	voltage = sqrt(2*energy/capacitance)
	return floor(g * voltage)
