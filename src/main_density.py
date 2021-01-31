import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from src.models.density import SplitKNeighborDensity

# ----------------------------------------------------------------------
# Plot a 1D density example
N = 100000
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
        label='input distribution')
colors = ['navy', 'cornflowerblue', 'darkorange']
kernels = ['gaussian', 'tophat', 'epanechnikov']
lw = 2

# for color, kernel in zip(colors, kernels):
#     kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
#     log_dens = kde.score_samples(X_plot)
#     ax.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=lw,
#             linestyle='-', label="kernel = '{0}'".format(kernel))

kde = SplitKNeighborDensity(n_neighbors=1000)
n_splits = 100
X_split = kde.get_random_split(X, n_splits)[0]
kde.fit(X_split)
for kk, k in enumerate([1000]):
    log_dens = kde.score_samples(X_plot, k=[k])
    for key in log_dens:
        ax.plot(X_plot[:, 0], np.exp(log_dens[key]), lw=lw,
                linestyle='-', label="neighbor = {}".format(k))

ax.legend(loc='upper left')
ax.text(6, 0.38, "N={0} points".format(N))
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

# ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)

plt.show()