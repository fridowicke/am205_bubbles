import numpy as np
import matplotlib.pyplot as plt

import pyvista as pv
import pyacvd

# generate points based on a continuous test function
def f(x,y):
	return 5*x*np.exp(-x**2 - y**2)

x = np.arange(-2,2,0.1); y = np.arange(-2,2,0.1)
xx, yy = np.meshgrid(x,y)
zz = f(xx, yy)

# plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, zz)
plt.show()

# Generate PolyData object of points for triangulation
points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
pt_cloud = pv.PolyData(points)

# Perform Delaunay triangulation for preliminary mesh
surf = pt_cloud.delaunay_2d()
surf.plot(show_edges=True)

# clip the top and bottom to create open boundaries by defining a
# height field using the test function.
surf['Z Height'] = np.abs(f(surf.points[:,0],surf.points[:,1]))
clip = surf.clip_scalar('Z Height', value=1)
clip.plot(show_edges=True)

# Refine the mesh using Voronoi tessellation with pyacvd
# to obtain more uniform cell shapes
clus = pyacvd.Clustering(clip)
clus.subdivide(3)
clus.cluster(2000) # number of mesh points to use
remesh = clus.create_mesh()

# Plot the final result.
remesh.plot(color='w', show_edges=True)
