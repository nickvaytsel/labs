import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

N = 10
data = np.loadtxt("data_5.txt", dtype='float')


def SME(x1, x2):
    max2 = np.max(x2)
    temp_sum = 0
    for i in range(0, x1.shape[0]):
        temp_sum = temp_sum + abs(x1[i] - x2[i]) ** 2
    return (temp_sum / x1.shape[0]) ** 0.5 / max2


def gauss(C, d):
    s = N + 1
    y = np.ones(s)
    for i in range(0, s):
        temp_sum = 0
        for j in range(0, i):
            temp_sum = temp_sum + C[i, j] * y[j]
        y[i] = d[i] - temp_sum
        y[i] = y[i] / C[i, i]
    return y


def rev_gauss(C, d):
    s = N + 1
    x = np.ones(s)
    for i in range(s - 1, -1, -1):
        temp_sum = 0
        for j in range(i + 1, s):
            temp_sum = temp_sum + C[i, j] * x[j]
        x[i] = d[i] - temp_sum
        x[i] = x[i] / C[i, i]
    return x


M = data.shape[0]
Eps = data[:, 0]
b = np.array(data[:, 1])
A = np.array([np.ones(M)])

for j in range(1, N + 1):
    temp = np.array([np.power(Eps, j)])
    A = np.append(A, temp, axis=0)

AT = np.transpose(A)
ATA = np.dot(A, AT)
ATb = np.dot(b, AT)
C = np.linalg.cholesky(ATA)
CT = np.transpose(C)

y = gauss(C, ATb)
x = rev_gauss(CT, y)

f = np.dot(x, A)

condA = np.linalg.cond(A)
condATA = np.linalg.cond(ATA)
print("mu(A): {:e}".format(condA), " mu(ATA): {:e}".format(condATA))
print("SME(НУ): {:e}".format(SME(f, b)))

R = np.transpose(np.copy(A))
QTb = np.copy(b)
for i in range(0, R.shape[1]):
    if np.array_equal(R[i][(i + 1):], np.zeros(R.shape[0] - i - 1)) != 1:
        e = np.zeros(R.shape[0] - i)
        e[0] = 1
        u = R[i:, i] + np.dot((np.sign(R[i, i]) ** np.sign(R[i, i])) * np.linalg.norm(R[i:, i]), e)
        u = np.dot(np.linalg.norm(u) ** -1, u)
        R[i:, i:] = R[i:, i:] - np.dot(2, np.outer(u, np.dot(np.transpose(u), R[i:, i:])))
        QTb[i:] = QTb[i:] - np.dot(2, np.dot(u, np.dot(np.transpose(u), QTb[i:])))

x1 = rev_gauss(R, QTb)
f1 = np.dot(x1, A)

print("SME(QR): {:e}".format(SME(f1, b)))
print("time elapsed: {:e}".format(time.time() - start_time))

d1 = (f - b)**2
d2 = (f1 - b)**2


fig1, ax0 = plt.subplots(1)
fig2, ax1 = plt.subplots(1)

ax0.set_title('Метод НУ')
ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax0.plot(Eps, b, label="σ(x)")
ax0.plot(Eps, f, label="f(x)")
ax0.legend(loc='lower right')
ax1.set_title('Метод QR-разложения')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.plot(Eps, b, label="σ(x)")
ax1.plot(Eps, f1, 'tab:green', label="f(x)")
ax1.legend(loc='lower right')
fig3, ax2 = plt.subplots(1)
ax2.set_title('Квадраты разностей')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.plot(Eps, d1, label = "НУ")
ax2.plot(Eps, d2, label = "QR-разложение")
ax2.legend(loc = 'lower right')

plt.show()
