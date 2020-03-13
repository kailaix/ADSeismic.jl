import scipy.signal
import numpy as np

def calc_dx_dt(f0, v):
  dx = v * 1/f0 / 7
  dy = dx
  dt = dx / v / 3
  return dx, dt


def gaussian(f0, dt):
  nt = np.around(10/f0 / dt)
  src = scipy.signal.windows.gaussian(nt, std=1/f0/dt)
  return src

def ricker(f0, dt):
  nt = np.around(10/f0/dt)
  src = scipy.signal.ricker(nt, a=1/f0/dt)
  return src


