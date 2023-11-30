import matplotlib.pyplot as plt
from mantid.plots.utility import MantidAxType
from mantid.api import AnalysisDataService as ADS
from mantid.simpleapi import mtd

if 'fit_Workspace_0' not in mtd:
    raise RuntimeError('You must run the fit in the cef_fit.py script first')

fit_Workspace_2 = ADS.retrieve('fit_Workspace_2')
fit_Workspace_1 = ADS.retrieve('fit_Workspace_1')
fit_Workspace_0 = ADS.retrieve('fit_Workspace_0')

fig, axes = plt.subplots(edgecolor='#ffffff', num='fit_Workspace_0-1', subplot_kw={'projection': 'mantid'})
axes.plot(fit_Workspace_0, color='#1f77b4', label='Ei=18 7K calc', markeredgecolor='#ff7f0e', markerfacecolor='#ff7f0e', wkspIndex=1)
axes.plot(fit_Workspace_1, color='#ff7f0e', label='Ei=10 7K calc', markeredgecolor='#d62728', markerfacecolor='#d62728', wkspIndex=1)
axes.plot(fit_Workspace_2, color='#2ca02c', label='Ei=100 7K calc', markeredgecolor='#8c564b', markerfacecolor='#8c564b', wkspIndex=1)
axes.errorbar(fit_Workspace_1, color='#2ca02c', ecolor='#ff7f0e', elinewidth=1.0, label='Ei=10 7K', linestyle='None', marker='.', markeredgecolor='#ff7f0e', markerfacecolor='#ff7f0e', wkspIndex=0)
axes.errorbar(fit_Workspace_2, color='#2ca02c', elinewidth=1.0, label='Ei=100 7K', linestyle='None', marker='.', wkspIndex=0)
axes.errorbar(fit_Workspace_0, color='#1f77b4', elinewidth=1.0, label='Ei=23 7K', linestyle='None', marker='.', wkspIndex=0)
axes.tick_params(axis='x', which='major', **{'gridOn': False, 'tick1On': True, 'tick2On': False, 'label1On': True, 'label2On': False, 'size': 6, 'tickdir': 'out', 'width': 1})
axes.tick_params(axis='y', which='major', **{'gridOn': False, 'tick1On': True, 'tick2On': False, 'label1On': True, 'label2On': False, 'size': 6, 'tickdir': 'out', 'width': 1})
axes.set_title('StTb2O4 - CEF Fit')
axes.set_xlim([0.4228637113068211, 41.71124999970198])
axes.set_ylim([29.271179077666755, 3454.0800416098996])
axes.set_xscale('log')
axes.set_yscale('log')
legend = axes.legend(fontsize=8.0).set_draggable(True).legend

plt.show()
# Scripting Plots in Mantid:
# https://docs.mantidproject.org/tutorials/python_in_mantid/plotting/02_scripting_plots.html
