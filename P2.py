from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
Script con el cual calcularemos la constante de Hubble con un
intervalo de confianza del 95%
'''

def bootstrap(data):
    '''
    Simulacion de bootstrap para encontrar el
    intervalo de  confianza (95%)
    '''
    N = data.shape[0]
    N_boot = 10000
    H = np.zeros(N_boot)
    for i in range(N_boot):
        s = np.random.randint(low=0, high=N, size=N)
        datos_dummy = data[s][s]
        #print(datos_dummy)
        x = datos_dummy[:, 0]
        v = datos_dummy[:, 1]
        H_1, cov_1 = curve_fit(minimizar_1, x, v, 2)
        H_2, cov_2 = curve_fit(minimizar_2, v, x, 2)
        prom = (H_1 + H_2) / 2
        H[i] = prom
    H = np.sort(H)
    inf = H[int(N_boot * 0.025)]
    sup = H[int(N_boot * 0.975)]
    print "El intervalo de confianza al 95% es: [{}:{}]".format(inf, sup)


def minimizar_1(x, H):
    return x * H


def minimizar_2(v, H):
    return v / H

#Main
data = np.loadtxt("SNIa.dat", usecols=(1, 2))
x = data[:, 0]
v = data[:, 1]

H_1, cov_1 = curve_fit(minimizar_1, x, v, 2)
H_2, cov_2 = curve_fit(minimizar_2, v, x, 2)

prom = (H_1 + H_2) / 2
print "H = {}".format(prom[0])

l = np.linspace(0., 32000, 10**6)
fig = plt.figure()
plt.scatter(x, v, label="Datos Observacionales")
plt.plot(l, minimizar_1(l, H_1), 'r', label='H*D')
plt.plot(l, minimizar_1(l, H_2), 'g', label='H/v')
plt.plot(l, minimizar_1(l, prom), 'b', label='Promedio')
plt.xlabel("Distancia [Mpc]")
plt.ylabel("Velocidad [km / s]")
plt.legend(loc=2)
plt.show()

confianza = bootstrap(data)
