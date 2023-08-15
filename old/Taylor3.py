from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF
import numpy as np
refstd = refstd            # Reference standard deviation

tr = PolarAxes.PolarTransform()

# Correlation labels
rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
srange=(0, 1.5)
    # Diagram limited to positive correlations
tmax = np.pi/2
tlocs = np.arccos(rlocs)        # Conversion to polar angles
gl1 = GF.FixedLocator(tlocs)    # Positions
tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

# Standard deviation axis extent (in units of reference stddev)
smin = srange[0] * refstd
smax = srange[1] * refstd

ghelper = FA.GridHelperCurveLinear(
    tr,
    extremes=(0, tmax, smin, smax),
    grid_locator1=gl1, tick_formatter1=tf1)

if fig is None:
    fig = PLT.figure()

ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
fig.add_subplot(ax)

# Adjust axes
ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
ax.axis["top"].toggle(ticklabels=True, label=True)
ax.axis["top"].major_ticklabels.set_axis_direction("top")
ax.axis["top"].label.set_axis_direction("top")
ax.axis["top"].label.set_text("Correlation")

ax.axis["left"].set_axis_direction("bottom")  # "X axis"
ax.axis["left"].label.set_text("Standard deviation")

ax.axis["right"].set_axis_direction("top")    # "Y-axis"
ax.axis["right"].toggle(ticklabels=True)
ax.axis["right"].major_ticklabels.set_axis_direction(
    "bottom" if extend else "left")

if self.smin:
    ax.axis["bottom"].toggle(ticklabels=False, label=False)
else:
    ax.axis["bottom"].set_visible(False)          # Unused

self._ax = ax                   # Graphical axes
self.ax = ax.get_aux_axes(tr)   # Polar coordinates

# Add reference point and stddev contour
l, = self.ax.plot([0], self.refstd, 'k*',
                  ls='', ms=10, label=label)
t = NP.linspace(0, self.tmax)
r = NP.zeros_like(t) + self.refstd
self.ax.plot(t, r, 'k--', label='_')

# Collect sample points for latter use (e.g. legend)
self.samplePoints = [l]