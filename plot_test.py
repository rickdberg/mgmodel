# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:49:28 2017

@author: rickdberg
"""

idx = 10
print(gradient[idx])
def conc_curve(z, a):
    return concunique[0,1] * np.exp(a*z)  # a = (v - sqrt(v**2 * 4Dk))/2D
conc_fit, conc_cov = optimize.curve_fit(conc_curve, concunique[:dp,0], conc_rand[idx,:], p0=0.01)
conc_fit = conc_fit[0]

conc_interp_fit_plot = conc_curve(np.linspace(concunique[0,0], concunique[-1,0], num=50), conc_fit)

# Set up axes and subplot grid
figure_1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 7))
grid = gridspec.GridSpec(3, 8, wspace=0.7)
ax1 = plt.subplot(grid[0:3, :2])
ax1.grid()
ax2 = plt.subplot(grid[0:3, 2:4], sharey=ax1)
ax2.grid()
ax3 = plt.subplot(grid[0:3, 4:6], sharey=ax1)
ax3.grid()
ax4 = plt.subplot(grid[0:3, 6:8], sharey=ax1)
ax4.grid()

# Figure title
figure_1.suptitle(r"$Expedition\ {},\ Site\ {}\ \ \ \ \ \ \ \ \ \ \ \ \ \ {}\ flux={}\ mol/m^2y$".format(Leg, Site, Solute_db, round(flux,4)), fontsize=20)

# Plot input data
ax1.plot(conc_rand[idx,:], concunique[0:6,0], 'go')
ax1.plot(conc_rand[idx,0:dp], concunique[0:6,0], 'bo', label="Used for curve fit")
ax1.plot(conc_interp_fit_plot, np.linspace(concunique[0,0], concunique[-1,0], num=50), 'k-')
ax2.plot(por[:, 1], por[:, 0], 'mo', label='Measured')
ax2.plot(porcurve(pordepth, porfit), pordepth, 'k-', label='Curve fit', linewidth=3)
ax3.plot(sedtemp(np.arange(concunique[-1,0]), bottom_temp), np.arange(concunique[-1,0]), 'k-', linewidth=3)
ax4.plot(picks[:,1]/1000000, picks[:,0], 'ro', label='Picks')
ax4.plot(sedtimes/1000000, seddepths, 'k-', label='Curve fit', linewidth=2)

# Inset in concentration plot
y2 = np.ceil(concunique[dp-1,0])
x2 = max(concunique[:dp,1])+2
x1 = min(concunique[:dp,1])-2
axins1 = inset_axes(ax1, width="50%", height="30%", loc=5)
axins1.plot(conc_rand[idx,:], concunique[0:6,0], 'go')
axins1.plot(conc_rand[idx,0:dp], concunique[0:6,0], 'bo', label="Used for curve fit")
axins1.plot(conc_interp_fit_plot, np.linspace(concunique[0,0], concunique[-1,0], num=50), 'k-')
axins1.set_xlim(x1-1, x2+1)
axins1.set_ylim(0, y2)
mark_inset(ax1, axins1, loc1=1, loc2=2, fc="none", ec="0.5")

# Additional formatting
ax1.legend(loc='best', fontsize='small')
ax2.legend(loc='best', fontsize='small')
ax4.legend(loc='best', fontsize='small')
ax1.set_ylabel('Depth (mbsf)')
ax1.set_xlabel('Concentration (mM)')
ax2.set_xlabel('Porosity')
ax3.set_xlabel('Temperature (\u00b0C)')
ax4.set_xlabel('Age (Ma)')
ax1.locator_params(axis='x', nbins=4)
ax2.locator_params(axis='x', nbins=4)
ax3.locator_params(axis='x', nbins=4)
ax4.locator_params(axis='x', nbins=4)
axins1.locator_params(axis='x', nbins=3)
ax1.invert_yaxis()
axins1.invert_yaxis()
figure_1.show()


