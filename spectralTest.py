from sklearn import datasets
import matplotlib.pyplot as plt
import spectral
import k_means
import numpy as np
n_samples = 1500
circles, circles_y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons, moons_y = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs, blobs_y = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure, no_y = np.random.rand(n_samples, 2), None


colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

circ_lap = spectral.setup(circles, 0.125)
circ_eig = spectral.computeEigen(circ_lap, 2)
circle_center, circle_indx = spectral.cluster(circ_eig, 2, False)
plt.figure(1)
plt.scatter(circles[:, 0], circles[:, 1], color=colors[circle_indx].tolist())

circle_center_means, circle_indx_means = k_means.cluster(circles, 2, False)
plt.figure(2)
plt.scatter(circles[:, 0], circles[:, 1], color=colors[circle_indx_means].tolist())



circ_lap = spectral.setup(noisy_moons, 0.125)
circ_eig = spectral.computeEigen(circ_lap, 2)
circle_center, circle_indx = spectral.cluster(circ_eig, 2, False)
plt.figure(3)
plt.scatter(noisy_moons[:, 0], noisy_moons[:, 1], color=colors[circle_indx].tolist())

circle_center_means, circle_indx_means = k_means.cluster(noisy_moons, 2, False)
plt.figure(4)
plt.scatter(noisy_moons[:, 0], noisy_moons[:, 1], color=colors[circle_indx_means].tolist())

"""
circ_lap = spectral.setup(no_structure)
circ_eig = spectral.computeEigen(circ_lap, 2)
circle_center, circle_indx = spectral.cluster(circ_eig, 2, False)
plt.figure(5)
plt.scatter(no_structure[:, 0], no_structure[:, 1], color=colors[circle_indx].tolist())

circle_center_means, circle_indx_means = k_means.cluster(no_structure, 2, False)
plt.figure(6)
plt.scatter(no_structure[:, 0], no_structure[:, 1], color=colors[circle_indx_means].tolist())


"""
plt.show()