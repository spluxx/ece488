from ece488_hw02 import capture
import math

def f(t): return 0.5*math.sin(t)+0.5

print(capture(f, 0.01, 0.1, 1e5, 0))
