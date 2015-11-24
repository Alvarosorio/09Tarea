from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def lineal(a, b, x):
    return a*x + b

'''
Script con el cual modelamos una relacion lineal entre el flujo
de la banda i y la banda z
'''

def montecarlo(banda_i, error_i, banda_z, error_z):
    '''realiza una simulacion de montecarlo
    para obtener el intervalo de confianza '''
    N_monte = 10000
    l = len(banda_i)
    beta = np.zeros(N_monte)
    alfa = np.zeros(N_monte)

    for i in range(N_monte):
        r = np.random.normal(0, 1, size=l)
        dummy_i = banda_i + error_i * r
        dummy_z = banda_z + error_z * r
        alfa[i], beta[i] = np.polyfit(dummy_i, dummy_z, 1)

    alfa = np.sort(alfa)
    beta = np.sort(beta)
    inf_alfa = alfa[int(N_monte * 0.025)]
    sup_alfa = alfa[int(N_monte * 0.975)]
    inf_beta = beta[int(N_monte * 0.025)]
    sup_beta = beta[int(N_monte * 0.975)]

    print "El intervalo de confianza para " \
          "la pendiente al 95% es: [{}:{}]".format(inf_alfa, sup_alfa)
    print "El intervalo de confianza para " \
          "el coef de posicion al 95% es: [{}:{}]".format(inf_beta, sup_beta)


data = np.loadtxt("data/DR9Q.dat", usecols=(80, 81, 82, 83))
#los datos normalizados
banda_i = data[:, 0] * 3.631
error_i = data[:, 1] * 3.631
banda_z = data[:, 2] * 3.631
error_z = data[:, 3] * 3.631
r = np.polyfit(banda_i, banda_z, 1)

#Main

x = np.linspace(0., 430, 10**6)
fig = plt.figure()
print "La recta optima : x*{} + {}".format(r[0], r[1])
plt.plot(x, lineal(r[0], r[1], x), 'r', label='Ajuste')
plt.errorbar(banda_i, banda_z, xerr=error_i, yerr=error_z, fmt='o',
             label='Datos Observacionales')
plt.xlabel("Banda i [$10^{-6} Jy$]")
plt.ylabel("Banda z [$10^{-6} Jy$]")
plt.legend(loc=2)
plt.show()

confianza = montecarlo(banda_i, error_i, banda_z, error_z)

