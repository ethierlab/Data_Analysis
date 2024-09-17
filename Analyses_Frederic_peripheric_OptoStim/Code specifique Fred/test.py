import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate
import scipy.special as special
import scipy.fft as fft

# borne=[[1,20],[30,40],[50,60]]
# for b in borne:
#     print(b[0])
# for i in range(12):
#     print(i)
# a=range(12)
# print(a)

# chan=[[1,2,3,4,1],[1,7,8,9,10]]

# for channel in chan:
#     plt.plot(channel)
# plt.show()

# m=0.43
# print( m)

# string="hello_21_342_dsfadf"

# print(string.rfind("_"))
# print(string.rindex("21"))
# a=min(chan[0])
# print(a)

# class Test:
#     x=1
#     def __init__(self,path):
#         self.path = path
#     def printtest(self):
#         print(self.path)

# class Point():
#     "Ceci est une classe point"
# p9 = Point()
# print(p9)

# def test(a= 2):
#     print(a)
# test(3)

# -------- Cours Module 4 ------

# A = np.array([[4, 2], [3, -1]])
# b = np.array([-1, 2])
# x = np.dot(np.linalg.inv(A), b)
# print(x)

# # ---- Méthode PLU ----

# lu, piv = lu_factor(A)
# x = lu_solve((lu, piv), b)
# print('lu',lu)
# print(x)

# # Avec NumPy
# x = np.linalg.solve(A, b)
# print(x)



# x = np.linspace(0, 2*np.pi, 500) # échantillonnage de 500 points sur [0, 2pi]
# cos = np.cos(x)                  # calcul du cos sur l'intervalle échantillonné

# plt.plot(x, cos)

# spl = UnivariateSpline(x, cos, k=4) # calcul de la spline

# primitive = spl.antiderivative()    # calcul de la primitive
# dérivée = spl.derivative()          # calcul de la dérivée

# plt.plot(x, primitive(x), label="primitive")
# plt.plot(x, dérivée(x), label="dérivée")
# plt.legend()
# plt.show()

# result = integrate.quad(lambda x: special.jv(2, x), 0, 5)
# print(result)

f0 = 5
fe = 50  # fréquence d'échantillonage
T = 1   # durée du signal
N = T*fe # nombre d'échantillons

t = np.arange(0, N) / fe        # tableau temporel
freq = np.arange(0, N)*fe / N   # tableau fréquentiel

# On créé un signal bruité en ajoutant un bruit gaussien à un sinus
signal = 1/2 * np.sin(2*np.pi*f0*t) + np.random.normal(0, 0.3, N)

# On affiche le signal bruité
plt.plot(t, signal)
plt.xlabel("Temps (s)")
plt.title("Signal temporel bruité")
plt.show()

# On calcule la transformée de Fourier du signal bruité
signal_f = fft.fft(signal)[:N//2+1]

# On affiche le signal bruité en fréquentiel
plt.stem(freq[:N//2+1], np.abs(signal_f))
plt.xlabel("Fréquence (Hz)")
plt.title("Signal en fréquentiel bruité")
plt.show()

centre = 5
sigma = 0.5

# On définit le filtre gaussien
gaussien = np.exp(-(freq[:N//2+1] - centre)**2/sigma**2)

# On affiche le filtre gaussien
plt.plot(freq[:N//2+1], gaussien)
plt.xlabel("Fréquence (Hz)")
plt.title("Filtre Gaussien")
plt.show()

# On applique le filtre au signal fréquentiel
signal_f_deb = gaussien*signal_f

# On affiche le spectre filtré
plt.stem(freq[:N//2+1], np.abs(signal_f_deb))
plt.xlabel("Fréquence (Hz)")
plt.title("Signal fréquentiel filtré")
plt.show()

# Voir la remarque ci-dessous pour cette ligne de code
signal_f_deb_with_neg = np.concatenate([signal_f_deb[:-1], np.conjugate(signal_f_deb[:0:-1])])

# On fait la transformée inverse et on prend seulement les valeurs réelles
signal_t_deb = fft.ifft(signal_f_deb_with_neg).real

# On affiche notre signal bruité et débruité
plt.plot(t, signal, label="Signal bruité")
plt.plot(t, signal_t_deb, label="Signal filtré", linewidth=3, color="r")
plt.title("Signal temporel filtré")
plt.legend()
plt.show()

# Exercice 4.1 :
import numpy as np
from scipy.linalg import lu_factor, lu_solve

A = np.array([[2, 3, 4], [3, 5, -4], [4, 7, -2]])
b = np.array([53, 2, 31])
lu, piv = lu_factor(A)
x = lu_solve((lu, piv), b)

# Exercice 4.2
import numpy as np
import scipy.integrate as integrate
def cercle(x, r=1):
    return np.sqrt(r**2 - x**2)
result = integrate.quad(cercle, 0, 1)
pie = (4*result[0])