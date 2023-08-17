import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111, projection='3d')

bg = 'black'

fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)


for k in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    k.set_facecolor(bg)
    k.set_edgecolor(bg)

for k in ('x','y','z'):
    ax.tick_params(k, colors=bg)

def SpaceCurve(n=50):
    fx = lambda t: (np.sin(t), np.cos(t), np.sin(t)*np.cos(t))
    sx, sy, sz = [], [], []
    T = np.arange(0, 3*np.pi+np.pi/16, np.pi/16)
    for t in T:
        rx, ry, rz = fx(t)
        sx.append(rx)
        sy.append(ry)
        sz.append(rz)
    return sx, sy, sz

def Parabola():
    fx = lambda x, y: x**2 + y**2
    a = np.arange(-3, 3.1, 0.1)
    ux, uy = np.meshgrid(a, a)
    uz = fx(ux, uy)
    return ux, uy, uz


def Rotation(x, y, z, angle=np.pi/16):
    xrot = np.array([[1, 0, 0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
    yrot = np.array([[np.cos(angle), 0, -np.sin(angle)],[0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
    zrot = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])

    for i in range(len(x)):
        for j in range(len(x[0])):
            Y = np.array([x[i][j], y[i][j], z[i][j]])
            Z = zrot @ (yrot @ (xrot @ Y))
            x[i][j], y[i][j], z[i][j] = Z

    return x, y, z
            




x1, y1, z1 = SpaceCurve()
px, py, pz = Parabola()

scale = 0.1

Rotation(px, py, pz)

for qx, qy, qz in zip(x1, y1, z1):
    ax.axis('off')
    ax.cla()
    ax.plot(x1, y1, z1, color=bg, linewidth=0.01)
    px, py, pz = Rotation(px, py, pz)
    ax.plot_surface(qx + scale*px, qy + scale*py, qz + scale*pz, cmap='hsv')
    ax.grid(False)
    plt.pause(0.01)


plt.show()
