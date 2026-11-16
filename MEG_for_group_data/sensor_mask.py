import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arc
from group_plots import pdf

TFA_MASK_COLOR = '#fff49d'
TFA_MASK_EDGE = '#B99D00'
TFA_SENSOR_EDGE = '#b8b8b8'


def sensor_mask(frame, path):
    coords = frame[['x_m','y_m']].to_numpy()
    sensor_indices = np.flatnonzero(frame.in_cluster)
    fig, axis = plt.subplots(figsize=(4.3, 4.0), dpi=300)
    radius = float(np.max(np.linalg.norm(coords, axis=1)) * 1.08)
    axis.add_patch(
        Circle(
            (0, 0),
            radius=radius,
            facecolor="white",
            edgecolor="#6f6f6f",
            linewidth=1.1,
            zorder=1,
        )
    )
    nose = np.array(
        [
            [-0.08 * radius, 0.98 * radius],
            [0.0, 1.14 * radius],
            [0.08 * radius, 0.98 * radius],
        ]
    )
    axis.add_patch(
        Polygon(nose, closed=False, fill=False, edgecolor="#6f6f6f", linewidth=1.1, zorder=2)
    )
    axis.add_patch(
        Arc(
            (-1.02 * radius, 0),
            width=0.20 * radius,
            height=0.34 * radius,
            theta1=270,
            theta2=90,
            color="#6f6f6f",
            linewidth=1.0,
            zorder=2,
        )
    )
    axis.add_patch(
        Arc(
            (1.02 * radius, 0),
            width=0.20 * radius,
            height=0.34 * radius,
            theta1=90,
            theta2=270,
            color="#6f6f6f",
            linewidth=1.0,
            zorder=2,
        )
    )
    axis.scatter(
        coords[:, 0],
        coords[:, 1],
        s=20,
        facecolors="white",
        edgecolors=TFA_SENSOR_EDGE,
        linewidths=0.6,
        zorder=3,
    )
    axis.scatter(
        coords[sensor_indices, 0],
        coords[sensor_indices, 1],
        s=20,
        facecolors=TFA_MASK_COLOR,
        edgecolors=TFA_MASK_EDGE,
        linewidths=0.8,
        zorder=4,
    )
    axis.set_aspect("equal")
    lim = radius * 1.22
    axis.set_xlim(-lim, lim)
    axis.set_ylim(-lim * 0.98, lim * 1.08)
    axis.set_axis_off()

    pdf(fig,path)
