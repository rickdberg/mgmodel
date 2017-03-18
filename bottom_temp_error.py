# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:15:29 2017

@author: rickdberg
"""
import numpy as np
import matplotlib.pyplot as plt

TempD = 18
bottom_temp = np.linspace(0, 25, 100)
Ds = 1.875*10**-2  # m^2 per year free diffusion coefficient at 18C (ref?)


def Dstp(Td, T):
    # Viscosity at reference temperature
    muwd = 4.2844324477E-05 + 1/(1.5700386464E-01*(Td+6.4992620050E+01)**2+-9.1296496657E+01)
    A = 1.5409136040E+00 + 1.9981117208E-02 * Td + -9.5203865864E-05 * Td**2
    B = 7.9739318223E+00 + -7.5614568881E-02 * Td + 4.7237011074E-04 * Td**2
    visd = muwd*(1 + A*0.035 + B*0.035**2)

    # Viscosity vector
    muw = 4.2844324477E-05 + 1/(1.5700386464E-01*(T+6.4992620050E+01)**2+-9.1296496657E+01)
    C = 1.5409136040E+00 + 1.9981117208E-02 * T + -9.5203865864E-05 * T**2
    D = 7.9739318223E+00 + -7.5614568881E-02 * T + 4.7237011074E-04 * T**2
    vis = muw*(1 + C*0.035 + D*0.035**2)
    T = T+273.15
    Td = Td+273.15
    return T/vis*visd*Ds/Td  # Stokes-Einstein equation

# Diffusion coefficient
D_in_situ = Dstp(TempD, bottom_temp)

plt.plot(bottom_temp, D_in_situ/Dstp(TempD, 0))
plt.xlim((0, 25))
plt.grid()


