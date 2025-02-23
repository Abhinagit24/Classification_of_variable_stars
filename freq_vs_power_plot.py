import numpy as np
from bokeh.plotting import figure,show
from astropy.timeseries import LombScargle
import lightkurve as lk

def freq_vs_power_plot():
    light_curve = lk.read('/Users/abhinapremachandran/Desktop/MS_Research/June2_2024/data/data_rrlyrae/train_rrl/hlsp_tess-spoc_tess_phot_0000000022442201-s0037_tess_v1_lc.fits')
    fluxes = light_curve.flux.value
    times = light_curve.time.value
    nan_indices1 = np.isnan(fluxes)
    nan_indices2 = np.isnan(times)
    fluxes[nan_indices1] = np.mean(fluxes)
    times[nan_indices2] = np.mean(times)
    frequency, power = LombScargle(times, fluxes).autopower()
    light_curve_figure = figure(x_axis_label='Period', y_axis_label='Power')
    light_curve_figure.scatter(x=1/frequency, y=power)
    light_curve_figure.line(x=1/frequency, y=power, line_alpha=0.3)
    show(light_curve_figure)

if __name__ == '__main__':
    freq_vs_power_plot()