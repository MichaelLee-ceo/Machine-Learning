import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def lossFunction(w1, w2):
    return w1 ** 2 + w2 ** 2

def derivativeL(w):
    return np.array([2*w[0], 2*w[1]])

def gradientDescent(start, derivativeF, learningRate, epochs):
    w1_gradient = []
    w2_gradient = []
    w1_gradient.append(start[0])
    w2_gradient.append(start[1])
    prevWeight = start

    for i in range(epochs):
        w = prevWeight - learningRate * derivativeF(prevWeight)
        w1_gradient.append(w[0])
        w2_gradient.append(w[1])
        prevWeight = w

    return np.array(w1_gradient), np.array(w2_gradient)


start = np.array([2, 4])
lr = 0.1
epochs = 40

x1 = np.arange(-5, 5, 0.1)
x2 = np.arange(-5, 5, 0.1)
w1, w2 = np.meshgrid(x1, x2)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(w1, w2, lossFunction(w1, w2), rstride=1, cstride=1, cmap='coolwarm')

min_point = np.array([0., 0.])
min_point_ = min_point[:, np.newaxis]
ax.plot(*min_point_, lossFunction(*min_point_), 'r*', markersize=10)
ax.set_xlabel('w1')
ax.set_ylabel('w2')

w1_gd, w2_gd = gradientDescent(start, derivativeL, lr, epochs)
w_gd = np.column_stack([w1_gd, w2_gd])
print(w_gd)

ax.scatter(w1_gd, w2_gd, lossFunction(w1_gd, w2_gd), color='black')

for i in range(1, epochs+1):
    a = Arrow3D([w1_gd[i - 1], w1_gd[i]], [w2_gd[i - 1], w2_gd[i]],
                [lossFunction(w1_gd, w2_gd)[i - 1], lossFunction(w1_gd, w2_gd)[i]], mutation_scale=12,
                lw=1, arrowstyle="-|>", color="r")
    ax.add_artist(a)
#     ax.annotate3D('', xyz=(w1_gd[i], w2_gd[i], lossFunction(w1_gd, w2_gd)[i]),
#                    xycoords='data',
#                    xytext=(w1_gd[i-1], lossFunction(w1_gd, w2_gd)[i-1]),
#                    textcoords='offset points',
#                    arrowprops=dict(arrowstyle='->', color='r', lw=1))
    # ax.annotate3D('point 3', (0, 0, 1),
    #               xytext=(30, -30),
    #               textcoords='offset points',
    #               bbox=dict(boxstyle="round", fc="lightyellow"),
    #               arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=5))

plt.show()
