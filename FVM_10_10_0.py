import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from math import gcd
import time

# =============================
# Start timer
# =============================
t0 = time.time()

# =============================
# Parameters
# =============================
n = 999
acc = 1.0

# =============================
# Percolation lattice
# =============================
m = np.zeros((n+1, n+1), dtype=int)

for i in range(n):
    for j in range(n):
        d = np.random.rand()
        if d <= 1.0:     # bond probability = 1
            m[i, j] = 1

# Defect pattern
m[1:n:2, 0:n:3] = 0
m[0:n:2, 2:n:3] = 0

# =============================
# Hexagonal lattice geometry
# =============================
aa = np.sqrt(3) * acc
a  = np.sqrt(3) / 2

x, y = np.meshgrid(np.arange(1, n+2), np.arange(1, n+2))
nn = x.shape[0]

shift = np.tile([[0], [0.5]], (nn // 2, nn))
x = (x + shift) * acc
y = (a * y) * acc

# Remove inactive sites
inactive = (m == 0)
x[inactive] = 0.0
y[inactive] = 0.0

# =============================
# Plot full lattice
# =============================
plt.figure(1)
plt.plot(x, y, 'k.', markersize=5)
plt.axis('equal')
plt.axis([0, 150, 0, 150])
plt.grid(True)

# =============================
# Chiral vector and tube length
# =============================
ch1, ch2 = 10, 10
tc = 250

ch = aa * np.sqrt(ch1**2 + ch1*ch2 + ch2**2)
dr = gcd(2*ch2 + ch1, 2*ch1 + ch2)
t  = tc * (np.sqrt(3) / dr) * ch
r  = ch / (2 * np.pi)

# =============================
# Define rectangular polygon
# =============================
x0, y0 = x[0, 0], y[0, 0]

xv = np.array([x0, x0 + ch - 0.005, x0 + ch - 0.005, x0, x0])
yv = np.array([y0, y0, y0 + t - 0.005, y0 + t - 0.005, y0])

poly = Path(np.column_stack((xv, yv)))
points = np.column_stack((x.ravel(), y.ravel()))
inside = poly.contains_points(points).reshape(x.shape)

# =============================
# Plot selected region
# =============================
plt.figure(2)
plt.plot(xv, yv, 'k-')
plt.plot(x[inside], y[inside], 'k.', markersize=5)
plt.axis('equal')
plt.axis([0, 150, 0, 150])
plt.grid(True)

# =============================
# Count lattice nodes
# =============================
LN = np.sum(inside[:n, :n])

# =============================
# Unit cell length
# =============================
lnn = 2 * (2 * ch * ch) / (aa * aa * dr)
lnn = int(round(lnn))

# =============================
# Truncated lattice
# =============================
rows = (2 * tc) + 8
cols = lnn - 10

xx = x[:rows, :cols]
yy = y[:rows, :cols]
mi = m[:rows, :cols]

# Boundary removal
mi[:4, :] = 0
mi[rows-4:rows, :] = 0

# =============================
# Neighbor-marking rules
# =============================

# Block 1
for i in range(1, rows-1):
    for j in range(1, cols-1):
        if (j % 3 == 1) and (i % 2 == 0):
            if mi[i, j] == 1 and mi[i+1, j] == 0: mi[i+1, j] = 2
            if mi[i, j] == 1 and mi[i-1, j] == 0: mi[i-1, j] = 2
            if mi[i, j] == 1 and mi[i, j-1] == 0: mi[i, j-1] = 2

# Block 2
for i in range(1, rows-1):
    for j in range(1, cols-1):
        if (j % 3 == 1) and (i % 2 == 1):
            if mi[i, j] == 1 and mi[i+1, j] == 0: mi[i+1, j] = 2
            if mi[i, j] == 1 and mi[i-1, j] == 0: mi[i-1, j] = 2
            if mi[i, j] == 1 and mi[i, j+1] == 0: mi[i, j+1] = 2

# Block 3
for i in range(1, rows-1):
    for j in range(1, cols-1):
        if (j % 3 == 0) and (i % 2 == 0):
            if mi[i, j] == 1 and mi[i+1, j+1] == 0: mi[i+1, j+1] = 2
            if mi[i, j] == 1 and mi[i-1, j+1] == 0: mi[i-1, j+1] = 2
            if mi[i, j] == 1 and mi[i, j-1] == 0: mi[i, j-1] = 2

# Block 4
for i in range(1, rows-1):
    for j in range(1, cols-1):
        if (j % 3 == 2) and (i % 2 == 1):
            if mi[i, j] == 1 and mi[i+1, j-1] == 0: mi[i+1, j-1] = 2
            if mi[i, j] == 1 and mi[i-1, j-1] == 0: mi[i-1, j-1] = 2
            if mi[i, j] == 1 and mi[i, j+1] == 0: mi[i, j+1] = 2

# =============================
# Replication and masking
# =============================
M = np.tile(mi, (1, 3))
M[:, :(cols-3)] = 0
M[:, (2*cols)+3 : 3*cols] = 0

# =============================
# Save results
# =============================
np.savez("n1010_per_10000_0_exact.npz", M=M, LN=LN)

print("Elapsed time:", time.time() - t0)
plt.show()
