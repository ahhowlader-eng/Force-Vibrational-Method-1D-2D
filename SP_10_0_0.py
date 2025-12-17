import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

tic = time.time()

# =========================
# Load data
# =========================
data1 = loadmat('n100_per_1000_0_exact.mat')
M  = data1['M']
LN = int(data1['LN'][0, 0])
mi = data1['mi']
xx = data1['xx']
yy = data1['yy']

data2 = loadmat('n100_0_1350.mat')
U   = data2['U']
UX  = data2['UX']
UY  = data2['UY']
UZ  = data2['UZ']
U0  = data2['U0']
U0X = data2['U0X']
U0Y = data2['U0Y']
U0Z = data2['U0Z']

# =========================
# Allocate arrays
# =========================
mii  = np.zeros((75, 20))
xxx  = np.zeros((75, 20))
yyy  = np.zeros((75, 20))
UXX  = np.zeros((75, 20))
UYY  = np.zeros((75, 20))
UXXX = np.zeros((75, 20))
UYYY = np.zeros((75, 20))
UU   = np.zeros((75, 20))

# =========================
# Copy subdomains
# =========================
for X in range(20):
    for Y in range(75):
        mii[Y, X] = mi[Y+5, X]
        xxx[Y, X] = xx[Y+5, X]
        yyy[Y, X] = yy[Y+5, X]

        UXX[Y, X] = UX[Y+5, X+20]
        UYY[Y, X] = UY[Y+5, X+20]
        UU[Y, X]  = U[Y+5,  X+20]

# =========================
# Absolute value filtering
# =========================
UXXX[UXX < 0] = np.abs(UXX[UXX < 0])
UYYY[UYY < 0] = np.abs(UYY[UYY < 0])

# =========================
# Flatten into 1×1500 arrays
# =========================
x = np.zeros(1500)
y = np.zeros(1500)
c = np.zeros(1500)
d = np.zeros(1500)
e = np.zeros(1500)

for i in range(20):
    start = i * 75
    end   = (i + 1) * 75

    x[start:end] = xxx[:, i]
    y[start:end] = yyy[:, i]
    c[start:end] = UXXX[:, i]
    d[start:end] = UYYY[:, i]
    e[start:end] = UU[:, i]

# =========================
# Plot (scatter)
# =========================
sz = 30

plt.figure(figsize=(8, 6))
sc = plt.scatter(x, y, s=sz, c=e, cmap='viridis')

plt.xlim([-3, 200])
plt.ylim([7, 120])

cbar = plt.colorbar(sc)
cbar.ax.set_ylabel('U', fontsize=10, fontweight='bold')

plt.xticks([5, 10, 15, 20])
plt.yticks([20, 40, 60, 80, 100])

plt.gca().tick_params(labelsize=10)
plt.gca().set_facecolor('white')

plt.show()

print(f"Elapsed time: {time.time() - tic:.2f} s")
