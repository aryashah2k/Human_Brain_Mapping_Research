from nilearn import datasets, plotting
import inspect
import numpy as np
from pathlib import Path

# Get actual file used by Destrieux
d = datasets.fetch_atlas_surf_destrieux()
print('Keys:', sorted(d.keys()))
print('map_left shape:', np.array(d.map_left).shape)
print('template:', d.template)

# Find the actual annot file path
nilearn_data = Path.home() / 'nilearn_data' / 'destrieux_surface'
print('\nFiles in destrieux_surface:')
for f in nilearn_data.glob('*'):
    print(' ', f.name, f.stat().st_size, 'bytes')

# Correct plot_surf_roi signature
sig = inspect.signature(plotting.plot_surf_roi)
print('\nplot_surf_roi signature:')
for name, param in sig.parameters.items():
    print(f'  {name}: {param.default}')
