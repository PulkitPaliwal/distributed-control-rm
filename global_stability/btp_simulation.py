 import numpy as np
from math import sin, cos, atan2
import torch
import sympy as sp
from scipy import integrate
import tqdm
import multiprocessing
from tqdm.contrib.concurrent import process_map


sample_count = 100000

def foot(x, t, l, k1, k2):
    a = -(x[4] - k1)
    b = -(x[5] - k2)
    t1 = x[6]
    t2 = x[7]
    t3 = x[8]
    p1 = sin(t1)
    p2 = cos(t1)
    p3 = sin(t2)
    p4 = cos(t2)
    p5 = sin(t3)
    p6 = cos(t3)
    p7 = sin(t1 + t2)
    p8 = cos(t1 + t2)
    p9 = sin(t1 + t2 + t3)
    p10 = cos(t1 + t2 + t3)
    p11 = sin(t2 + t3)
    p12 = cos(t2 + t3)
    x_prime = b * p9 + a * p10
    y_prime = b * p10 - a * p9
    a = x_prime
    b = y_prime
    x1dot = -l * p1 * (atan2((l * p3 + l * p11 + a * p11 + b * p12), (l + l * p4 + l * p12 + a * p12 - b * p11)) -
                      atan2((l * p3 + l * p11), (l + l * p4 + l * p12)))
    x2dot = l * p2 * (atan2((l * p3 + l * p11 + a * p11 + b * p12), (l + l * p4 + l * p12 + a * p12 - b * p11)) -
                     atan2((l * p3 + l * p11), (l + l * p4 + l * p12)))
    x3dot = (-l * p1 - l * p7) * (atan2((l * p3 + l * p11 + a * p11 + b * p12), (l + l * p4 + l * p12 + a * p12 - b * p11)) -
                                 atan2((l * p3 + l * p11), (l + l * p4 + l * p12))) - l * p7 * (atan2((l * p5 + a * p5 + b * p6), (l + l * p6 + a * p6 - b * p5)) -
                                                                                                atan2((l * p5), (l + l * p6)))
    x4dot = (l * p2 + l * p8) * (atan2((l * p3 + l * p11 + a * p11 + b * p12), (l + l * p4 + l * p12 + a * p12 - b * p11)) -
                                atan2((l * p3 + l * p11), (l + l * p4 + l * p12))) + l * p8 * (atan2((l * p5 + a * p5 + b * p6), (l + l * p6 + a * p6 - b * p5)) -
                                                                                               atan2((l * p5), (l + l * p6)))
    x5dot = (-l * p1 - l * p7 - l * p9) * (atan2((l * p3 + l * p11 + a * p11 + b * p12), (l + l * p4 + l * p12 + a * p12 - b * p11)) -
                                          atan2((l * p3 + l * p11), (l + l * p4 + l * p12))) + (-l * p7 - l * p9) * (atan2((l * p5 + a * p5 + b * p6), (l + l * p6 + a * p6 - b * p5)) -
                                                                                                                 atan2((l * p5), (l + l * p6))) - l * p9 * atan2(b, (l + a))
    x6dot = (l * p2 + l * p8 + l * p10) * (atan2((l * p3 + l * p11 + a * p11 + b * p12), (l + l * p4 + l * p12 + a * p12 - b * p11)) -
                                          atan2((l * p3 + l * p11), (l + l * p4 + l * p12))) + (l * p8 + l * p10) * (atan2((l * p5 + a * p5 + b * p6), (l + l * p6 + a * p6 - b * p5)) -
                                                                                                                 atan2((l * p5), (l + l * p6))) + l * p10 * atan2(b, (l + a))
    t1dot = (atan2((l * p3 + l * p11 + a * p11 + b * p12), (l + l * p4 + l * p12 + a * p12 - b * p11)) -
             atan2((l * p3 + l * p11), (l + l * p4 + l * p12)))
    t2dot = (atan2((l * p5 + a * p5 + b * p6), (l + l * p6 + a * p6 - b * p5)) -
             atan2((l * p5), (l + l * p6)))
    t3dot = atan2(b, (l + a))
    dXdt = np.array([x1dot, x2dot, x3dot, x4dot, x5dot, x6dot, t1dot, t2dot, t3dot])
    return dXdt




# Define all the system variables
delta_1, delta_2, delta_3, phi_1, phi_2, phi_3, alpha, R, l, D_11, D_12, D_13 = sp.symbols('delta_1 delta_2 delta_3 phi_1 phi_2 phi_3 alpha R l D_11 D_12 D_13')

delta_1 = alpha - sp.atan2((sp.sin(phi_1) + sp.sin(phi_1 + phi_2) + sp.sin(phi_1 + phi_2 + phi_3)), (sp.cos(phi_1) + sp.cos(phi_1 + phi_2) + sp.cos(phi_1 + phi_2 + phi_3)))
delta_2 = sp.atan2((R * sp.sin(alpha) - l * sp.sin(phi_1)), (R * sp.cos(alpha) - l * sp.cos(phi_1))) - sp.atan2((sp.sin(phi_1 + phi_2) + sp.sin(phi_1 + phi_2 + phi_3)), (sp.cos(phi_1 + phi_2) + sp.cos(phi_1 + phi_2 + phi_3)))
delta_3 = sp.atan2((R * sp.sin(alpha) - 2 * l * sp.sin(phi_1 + phi_2 / 2) * sp.cos(phi_2 / 2)), (R * sp.cos(alpha) - 2 * l * sp.cos(phi_1 + phi_2 / 2) * sp.cos(phi_2 / 2))) - (phi_1 + phi_2 + phi_3)


c_1 = sp.cos(phi_1)
c_12 = sp.cos(phi_1 + phi_2)
c_123 = sp.cos(phi_1 + phi_2 + phi_3)
c_3 = sp.cos(phi_3)
c_23 = sp.cos(phi_2 + phi_3)
c_2 = sp.cos(phi_2)

s_1 = sp.sin(phi_1)
s_12 = sp.sin(phi_1 + phi_2)
s_123 = sp.sin(phi_1 + phi_2 + phi_3)
s_3 = sp.sin(phi_3)
s_23 = sp.sin(phi_2 + phi_3)
s_2 = sp.sin(phi_2)

y_1 = l * s_1
y_2 = l * s_12
y_3 = l * s_123
y_E = l * (s_1 + s_12 + s_123)


v_dot = 2 * R * ((y_1 + y_2 + y_3) * delta_1 + (y_2 + y_3) * delta_2 + (y_3) * delta_3) - 2 * l ** 2 * ((s_2 + s_23) * delta_2 + (s_3 + s_23) * delta_3)

qty = -delta_1*delta_1-(delta_1*delta_2*(2+c_23+2*c_3+c_2)+delta_1*delta_3*(1+c_3+c_23))/((s_1+s_12+s_123)**2+(c_1+c_12+c_123)**2)
qty_2 = delta_2*(l*c_1*delta_1*(R-l*c_1) + l**2*s_1**2*delta_1)/(l**2*s_1**2+(R-l*c_1)**2) + delta_2*((c_12*(delta_1+delta_2)+c_123*(delta_1+delta_2+delta_3))*(c_12+c_123)+(s_12*(delta_1+delta_2)+s_123*(delta_1+delta_2+delta_3))*(s_12+s_123))/((s_12+s_123)**2+(c_12+c_123)**2)
qty_2 = -qty_2
qty_3 = delta_3*(-l*(c_1*delta_1+c_12*(delta_1+delta_2))*(R-l*(c_1+c_12))-l**2*(s_1+s_12)*(s_1*delta_1+s_12*(delta_1+delta_2)))/((R-l*c_1-l*c_12)**2+l**2*(s_1+s_12)**2) - delta_3*(delta_1+delta_2+delta_3)


import numpy as np
from scipy.integrate import ode

count = 0
r = 0.5
v_dot_subs_f = -1
d_pr = np.array([0.1, 0.1, 0.1])



samples = []
# while v_dot_subs_f < 1e-4 and count < 100000:
def get_samples(count):
    np.random.seed(count)
    phi_1_subs = -np.pi + 2 * np.pi * np.random.rand()
    phi_2_subs = -r * np.pi + 2 * r * np.pi * np.random.rand()
    phi_3_subs = -r * np.pi + 2 * r * np.pi * np.random.rand()
    R_subs = 0.1 + 2.9 * np.random.rand()
    l_subs = 1
    
    delta_1_subs = float(delta_1.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    delta_2_subs = float(delta_2.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    delta_3_subs = float(delta_3.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    
    l_ = 1
    k1 = R_subs
    k2 = 0
    f = [phi_1_subs, phi_2_subs, phi_3_subs]
    x_0 = [l_ * np.cos(f[0]), l_ * np.sin(f[0]), l_ * np.cos(f[0]) + l_ * np.cos(f[0] + f[1]), l_ * np.sin(f[0]) + l_ * np.sin(f[0] + f[1]), l_ * np.cos(f[0]) + l_ * np.cos(f[0] + f[1]) + l_ * np.cos(f[0] + f[1] + f[2]), l_ * np.sin(f[0]) + l_ * np.sin(f[0] + f[1]) + l_ * np.sin(f[0] + f[1] + f[2]), f[0], f[1], f[2]]
    tspan = np.arange(0, 300.1, 0.1)
  
    sol = integrate.odeint(foot, x_0, tspan, args=(l_, k1, k2))
  
    f1_star = sol[3000, 6]
    f2_star = sol[3000, 7]
    f3_star = sol[3000, 8]

    qty_1_4 = (phi_1_subs - f1_star) * delta_1_subs
    qty_2_4 = (phi_1_subs - f1_star + phi_2_subs - f2_star) * (delta_2_subs + delta_1_subs)
    qty_3_4 = (phi_3_subs - f3_star + phi_1_subs - f1_star + phi_2_subs - f2_star) * (delta_1_subs + delta_2_subs + delta_3_subs)
    v_dot_4 = 2 * (qty_1_4 + qty_2_4 + qty_3_4)

    qty_subs_1 = float(qty.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    qty_subs_2 = float(qty_2.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    qty_subs_3 = float(qty_3.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    v_dot_3 = 2 * (qty_subs_1 + qty_subs_2 + qty_subs_3)

    v_dot_subs = float(v_dot.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))

    v_dot_subs_f = v_dot_3  + v_dot_4  + v_dot_subs

    return ([v_dot_subs, v_dot_3, v_dot_4, v_dot_subs_f-100])


def get_sampled_points(count):
    np.random.seed(count)
    phi_1_subs = -np.pi + 2 * np.pi * np.random.rand()
    phi_2_subs = -r * np.pi + 2 * r * np.pi * np.random.rand()
    phi_3_subs = -r * np.pi + 2 * r * np.pi * np.random.rand()
    R_subs = 0.1 + 2.9 * np.random.rand()
    l_subs = 1
    l_ = 1
    k1 = R_subs
    k2 = 0
    f = [phi_1_subs, phi_2_subs, phi_3_subs]
    x_0 = [l_ * np.cos(f[0]), l_ * np.sin(f[0]), l_ * np.cos(f[0]) + l_ * np.cos(f[0] + f[1]), l_ * np.sin(f[0]) + l_ * np.sin(f[0] + f[1]), l_ * np.cos(f[0]) + l_ * np.cos(f[0] + f[1]) + l_ * np.cos(f[0] + f[1] + f[2]), l_ * np.sin(f[0]) + l_ * np.sin(f[0] + f[1]) + l_ * np.sin(f[0] + f[1] + f[2]), f[0], f[1], f[2]]
    tspan = np.arange(0, 300.1, 0.1)
  
    sol = integrate.odeint(foot, x_0, tspan, args=(l_, k1, k2))
  
    f1_star = sol[3000, 6]
    f2_star = sol[3000, 7]
    f3_star = sol[3000, 8]
    return ([phi_1_subs, phi_2_subs, phi_3_subs, R_subs, l_subs, f1_star, f2_star, f3_star])


sampled_points = np.load('sampled_points.npy', allow_pickle=True)

def get_samples_7_pt(count):
    s = sampled_points[count]
    phi_1_subs = s[0]
    phi_2_subs = s[1]
    phi_3_subs = s[2]
    l_subs = s[4]
    R_subs = s[3]
    f1_star = s[5]
    f2_star = s[6]
    f3_star = s[7]
    
    delta_1_subs = float(delta_1.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    delta_2_subs = float(delta_2.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    delta_3_subs = float(delta_3.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))

    qty_1_4 = (phi_1_subs - f1_star) * delta_1_subs
    qty_2_4 = (phi_1_subs - f1_star + phi_2_subs - f2_star) * (delta_2_subs + delta_1_subs)
    qty_3_4 = (phi_3_subs - f3_star + phi_1_subs - f1_star + phi_2_subs - f2_star) * (delta_1_subs + delta_2_subs + delta_3_subs)
    v_dot_4 = 2 * (qty_1_4 + qty_2_4 + qty_3_4)

    qty_subs_1 = float(qty.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    qty_subs_2 = float(qty_2.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    qty_subs_3 = float(qty_3.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))
    v_dot_3 = 2 * (qty_subs_1 + qty_subs_2 + qty_subs_3)

    v_dot_subs = float(v_dot.subs({phi_1: phi_1_subs, phi_2: phi_2_subs, phi_3: phi_3_subs, l: l_subs, R: R_subs, alpha: 0}))

    v_dot_subs_f = v_dot_3  + v_dot_4  + v_dot_subs

    return ([qty_1_4, qty_2_4, qty_3_4, qty_subs_1, qty_subs_2, qty_subs_3,  v_dot_subs, v_dot_subs_f])


# samples = process_map(get_samples, range(sample_count), max_workers=8, chunksize=1)
samples = process_map(get_samples_7_pt, range(sample_count), max_workers=16, chunksize=1)


# np.save('sample.npy', samples)
# np.save('sampled_points.npy', samples)
np.save('sample_7_pt.npy', samples)
# samples = np.load('samples.npy', allow_pickle=True)