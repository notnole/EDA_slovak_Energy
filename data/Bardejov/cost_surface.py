"""3D surface: cost per MWh EE as a function of EE production and heat demand.

Uses the engineering cost model from Priprava prevadzky TEHO_3.xlsx -> vstupy.
Cost at reference H=2 MW taken directly from the sheet (row 46); scaling with H
derived from the sheet's cost allocation:
    d(cost_EE_per_MWh) / d(H_demand) = -23.08 / MW_EE
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Engineering model constants from vstupy
FUEL_TH = 23.0769   # EUR per MWh thermal (wet chips)

# Sheet values: row 6 (MW_EE), row 8 (heat/EE ratio), row 46 (cost per MWh EE at H=2)
mw_steps    = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
heat_ratio  = np.array([5.3, 4.6, 4.0, 3.7, 3.5, 3.3, 3.2, 3.1, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.9])
cost_at_H2  = np.array([109.71, 106.65, 102.58, 100.20, 98.62, 96.13, 95.45,
                         94.39,  93.07,  93.94,  94.66,  95.27, 95.79, 96.24, 94.26])

# Grid
H_grid = np.linspace(0, 25, 120)
mw_grid = np.linspace(mw_steps.min(), mw_steps.max(), 120)

# Interpolate ratio and reference cost at arbitrary MW_EE
ratio_interp = np.interp(mw_grid, mw_steps, heat_ratio)
cost_H2_interp = np.interp(mw_grid, mw_steps, cost_at_H2)

MW, H = np.meshgrid(mw_grid, H_grid, indexing='ij')
turbine_heat = MW * ratio_interp[:, None]       # heat produced by turbine at each MW_EE
cost_H2_grid = cost_H2_interp[:, None]

# Cost per MWh EE = sheet_cost_at_H2 - (H - 2) * 23.08 / MW_EE
cost = cost_H2_grid - (H - 2.0) * FUEL_TH / MW

# Mask infeasible: can't deliver more heat than turbine produces
infeasible = H > turbine_heat
cost_masked = np.where(infeasible, np.nan, cost)

# --- 3D surface plot ---
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(MW, H, cost_masked, cmap=cm.viridis,
                        edgecolor='none', alpha=0.9,
                        vmin=20, vmax=110)

# Overlay zero-profit isolines at DA=60, 80, 94, 120 EUR/MWh
for DA in [60, 80, 94, 120]:
    ax.contour(MW, H, cost_masked, levels=[DA], colors='red',
                linewidths=1.5, linestyles='--', offset=None)

ax.set_xlabel('EE production (MW)', fontsize=11)
ax.set_ylabel('Heat demand (MWh/h)', fontsize=11)
ax.set_zlabel('Cost per MWh EE (EUR/MWh)', fontsize=11)
ax.set_title('TEHO Bardejov -- EE cost surface\n'
             '(engineering model from Priprava prevadzky, H=2 reference)',
             fontsize=12)

cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('Cost EUR/MWh EE')

ax.view_init(elev=25, azim=-135)

# 2D heatmap (second panel) for clarity
fig2, ax2 = plt.subplots(figsize=(11, 7))
im = ax2.pcolormesh(mw_grid, H_grid, cost_masked.T, cmap='viridis',
                     vmin=20, vmax=110, shading='auto')
cs = ax2.contour(mw_grid, H_grid, cost_masked.T,
                  levels=[40, 60, 80, 94, 120, 150],
                  colors='white', linewidths=1.2)
ax2.clabel(cs, fmt='%d EUR/MWh', fontsize=9)

# Mark operating region: H in [8, 22] typical Jan
ax2.axhline(16, color='red', linestyle=':', alpha=0.7, label='Jan 2025 avg heat (16 MWh/h)')
ax2.axhline(2,  color='orange', linestyle=':', alpha=0.7, label='Summer heat (2 MWh/h)')

ax2.set_xlabel('EE production (MW)')
ax2.set_ylabel('Heat demand (MWh/h)')
ax2.set_title('Cost per MWh EE -- heatmap view\n'
              '(white contours = break-even DA prices)')
ax2.legend(loc='upper left')
fig2.colorbar(im, ax=ax2, label='EUR per MWh EE')

fig.savefig('data/bardejov/cost_surface_3d.png', dpi=130, bbox_inches='tight')
fig2.savefig('data/bardejov/cost_surface_heatmap.png', dpi=130, bbox_inches='tight')
print('[+] Saved cost_surface_3d.png and cost_surface_heatmap.png')

# Print the key table
print('\n[*] Cost per MWh EE at selected (MW_EE, H_demand) points:')
print(f'{"H \\ MW":>8s}', *[f'{m:>6.1f}' for m in [1,2,3,4,5,6,7,8]])
for h in [0, 2, 5, 10, 15, 16, 20, 25]:
    row = []
    for m in [1,2,3,4,5,6,7,8]:
        idx = np.argmin(np.abs(mw_steps - m))
        c0 = cost_at_H2[idx]
        r = heat_ratio[idx]
        th = m * r
        if h > th:
            row.append('  n/a ')
        else:
            c = c0 - (h - 2) * FUEL_TH / m
            row.append(f'{c:6.1f}')
    print(f'{h:>6d}  ', *row)
