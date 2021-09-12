# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:31:34 2018

@author: nsde
"""

#%%
import tensorflow as tf

#%%
def mymin(a, b):
    return tf.where(a < b, a, tf.round(b))

#%%
def fmod(numer, denom): # tf.math.mod not registered for tf.float32 on gpu
    tquou = tf.math.floor(numer / denom)
    return numer - tquou * denom

#%%
def findcellidx(ndim, p, nc):
    if ndim==1:   return findcellidx1D(p, *nc)
    elif ndim==2: return findcellidx2D(p, *nc)
    elif ndim==3: return findcellidx3D(p, *nc)
    
#%%
def findcellidx1D(p, nx):
    idx = tf.floor(p * nx)
    idx = tf.minimum(0.0, tf.maximum(idx, nx-1))
    idx = tf.cast(tf.reshape(idx, (-1,)), tf.int32)
    return idx

#%%
def findcellidx2D(p, nx, ny):
    with tf.device(p.device):
        inc_x = tf.cast(1.0 / nx, tf.float32)
        inc_y = tf.cast(1.0 / ny, tf.float32)
    
    p0 = tf.minimum(nx * inc_x - 1e-8, tf.maximum(0.0, p[0]))
    p1 = tf.minimum(ny * inc_y - 1e-8, tf.maximum(0.0, p[1]))
    
    xmod = fmod(p0, inc_x)
    ymod = fmod(p1, inc_y)
    
    x = xmod / inc_x
    y = ymod / inc_y
    
    idx = mymin((nx-1) * tf.ones_like(p0), (p0-xmod) / inc_x) + \
          mymin((ny-1) * tf.ones_like(p1), (p1-ymod) / inc_y) * nx
    idx *= 4
    
    # Out of bound left
    cond1 = (p[0]<=0) & ((p[1]<=0) & (p[1]/inc_y<p[0]/inc_x))
    cond2 = (~ cond1) & (p[0]<=0) & ((p[1] >= ny * inc_y) & (p[1]/inc_y - ny > -p[0]/inc_x))
    cond3 = (~ cond1) & (~ cond2) & (p[0]<=0)
    idx = tf.where(cond2, idx+2, idx) #idx[cond2] += 2
    idx = tf.where(cond3, idx+3, idx) #idx[cond3] += 3

    # Out of bound right
    out = cond1 | cond2 | cond3
    cond4 = (~ out) & (p[0] >= nx*inc_x) & ((p[1]<=0) & (-p[1]/inc_y > p[0]/inc_x - nx))
    cond5 = (~ out) & (~ cond4) & (p[0] >= nx*inc_x) & ((p[1] >= ny*inc_y) & (p[1]/inc_y - ny > p[0]/inc_x-nx))
    cond6 = (~ out) & (~ cond4) & (~ cond5) & (p[0] >= nx*inc_x)
    idx = tf.where(cond5, idx+2, idx) #idx[cond5] += 2
    idx = tf.where(cond6, idx+1, idx) #idx[cond6] += 1
    
    # Out of bound up, nothing to do
    
    # Out of bound down
    out = out | cond4 | cond5 | cond6
    cond7 = (~ out) & (p[1] >= ny*inc_y)
    idx = tf.where(cond7, idx+2, idx) #idx[cond7] += 2

    # Ok, we are inbound
    out = out | cond7
    cond8 = (~ out) & (x<y) & (1-x<y)
    cond9 = (~ out) & (~ cond8) & (x<y)
    cond10 = (~ out) & (~ cond8) & (~ cond9) & (x>=y) & (1-x<y)
    idx = tf.where(cond8, idx+2, idx) #idx[cond8] += 2
    idx = tf.where(cond9, idx+3, idx) #idx[cond9] += 3
    idx = tf.where(cond10, idx+1, idx) #idx[cond10] += 1
    idx = tf.cast(tf.reshape(idx, (-1,)), tf.int32)
    return idx
    
#%%
def findcellidx3D(p, nx, ny, nz):
    # Conditions for points outside
    cond =  tf.logical_or(tf.logical_or(
            tf.logical_or(p[0,:] < 0.0, p[0,:] > 1.0),
            tf.logical_or(p[1,:] < 0.0, p[1,:] > 1.0)),
            tf.logical_or(p[2,:] < 0.0, p[2,:] > 1.0))
    
    # Push the points inside boundary
    inc_x, inc_y, inc_z = 1.0 / nx, 1.0 / ny, 1.0 / nz
    half = 0.5
    points_outside = p[:, cond]
    points_outside -= half
    abs_x = tf.abs(points_outside[0])
    abs_y = tf.abs(points_outside[1])
    abs_z = tf.abs(points_outside[2])
    push_x = (half * inc_x)*(tf.logical_and(abs_x < abs_y, abs_x < abs_z))
    push_y = (half * inc_y)*(tf.logical_and(abs_y < abs_x, abs_x < abs_z))
    push_z = (half * inc_z)*(tf.logical_and(abs_z < abs_x, abs_x < abs_y))
    cond_x = abs_x > half
    cond_y = abs_y > half
    cond_z = abs_z > half
    points_outside[0, cond_x] = (half - push_x[cond_x]) * tf.sign(points_outside[0, cond_x])
    points_outside[1, cond_y] = (half - push_y[cond_y]) * tf.sign(points_outside[1, cond_y])
    points_outside[2, cond_z] = (half - push_z[cond_z]) * tf.sign(points_outside[2, cond_z])
    points_outside += half
    p[:, cond] = points_outside

    # Find row, col, depth placement and cell placement
    inc_x, inc_y, inc_z = 1.0/nx, 1.0/ny, 1.0/nz
    p0 = tf.minimum(nx * inc_x - 1e-8, tf.maximum(0.0, p[0]))
    p1 = tf.minimum(ny * inc_y - 1e-8, tf.maximum(0.0, p[1]))
    p2 = tf.minimum(nz * inc_z - 1e-8, tf.maximum(0.0, p[2]))

    xmod = fmod(p0, inc_x)
    ymod = fmod(p1, inc_y)
    zmod = fmod(p2, inc_z)
    
    i = mymin((nx - 1) * tf.ones_like(p0), ((p0 - xmod) / inc_x))
    j = mymin((ny - 1) * tf.ones_like(p1), ((p1 - ymod) / inc_y))
    k = mymin((nz - 1) * tf.ones_like(p2) , ((p2 - zmod) / inc_z))
    idx = 5 * (i + j * nx + k * nx * ny)

    x = xmod / inc_x
    y = ymod / inc_y
    z = zmod / inc_z
    
    # Find subcell location
    cond = tf.logical_or(tf.logical_or(tf.logical_or(
            ((k%2==0) & (i%2==0) & (j%2==1)),
            ((k%2==0) & (i%2==1) & (j%2==0))),
            ((k%2==1) & (i%2==0) & (j%2==0))),
            ((k%2==1) & (i%2==1) & (j%2==1)))

    tmp = x.copy()
    x = tf.where(cond, y, x) #x[cond] = y[cond]
    y = tf.where(cond, 1-tmp, y) #y[cond] = 1-tmp[cond]
    
    cond1 = -x-y+z >= 0
    cond2 = x+y+z-2 >= 0
    cond3 = -x+y-z >= 0
    cond4 = x-y-z >= 0
    idx = tf.where(cond1, idx+1, idx) #idx[cond1] += 1
    idx = tf.where(cond2 & ~cond1, idx+2, idx) #idx[cond2 & ~cond1] += 2
    idx = tf.where(cond3 & ~cond1 & ~cond2, idx+3, idx) #idx[cond3 & ~cond1 & ~cond2] += 3
    idx = tf.where(cond4 & ~cond1 & ~cond2 & cond3, idx+4, idx) #idx[cond4 & ~cond1 & ~cond2 & ~cond3] += 4
    idx = tf.cast(tf.reshape(idx, (-1,)), tf.int32)
    return idx