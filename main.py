import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal


###############################################################################
#Fuente de Señal
###############################################################################
def fuente(f0, Ts, fase_inicial, N):
    return [np.sin(2*np.pi*t*Ts*f0 + fase_inicial) for t in range(N)]

#Ejercicio 2-------------------------------------------------------------------
s2={
'N' : 1000,
'f0' : 1000,
'fase_inicial' : np.pi / 2,
'Ts' : 1 / (100 * 1000)
}

signal_source = fuente(**s2)

#Ejercicio 3-------------------------------------------------------------------
s1 = fuente(s2['f0'], s2['Ts'], s2['fase_inicial'], s2['N']//2)
#s2 = fuente(f0, Ts, s1[-1] + np.pi, N//2)
#signal_source_3 = s1 + s2



###############################################################################
#Detector de fases
###############################################################################
def detector(xr, xv):
    return [a*b for a,b in zip(xr,xv)]

#Ejercicio 5-------------------------------------------------------------------


xr = fuente(s2['f0'], s2['Ts'], 0, s2['N'] )
xv = fuente(s2['f0'], s2['Ts'], np.pi/4, s2['N'] )
 
x_detect = detector(xr,xv)

#Gráfico temporal
t = [t*s2['Ts'] for t in range(s2['N'])]
plt.plot(t,xr ,t,x_detect)
plt.show()

#Gráfico en frecuencias
w = np.fft.fftfreq( s2['N'])
X_detect = fft(x_detect)
Xr = fft(xr)
plt.plot(w,X_detect, w,Xr)
plt.show()

#Ejercicio 6-------------------------------------------------------------------
#x_6 = fuente((s2['f0'], s2['Ts'], np.pi/2, s2['N'])
#x_6_detect = detector(x_6, xr)

#Gráfico Temporal
#plt.plot(t,xr ,t,x_6_detect)
#plt.show()
#Gráfico en frecuencias
#X_6 = fft(x_6_detect)
#plt.plot(w,X_6)
#plt.show()


#%%
###############################################################################
#Filtro
###############################################################################

type = set(('rc','lead-lag activo'))
#Ejercicio 7
# def filtro(xd, estado_inicial, tipo, C, R1, R2, Ts):
#     if tipo in type:
#         if tipo == 'rc':
#             b = (1)
#             a = (1, R1*C)
#             B, A = signal.bilinear(b, a, 1/Ts)
#             y = signal.lfilter(B, A, xd, zi = estado_inicial)

#             return y

NN = 200
fs = 1000
m1 = fuente(60, 1/fs, 0, NN)
m2 = fuente(5, 1/fs, 0, NN)
ss = np.zeros(NN)
for i in range(NN):
    ss[i] = m1[i] + m2[i]

t = np.linspace(0, NN/fs, NN)
#plt.plot(t, ss, t, m2)
SS = fft(ss)

ww = np.linspace(0,fs,NN)
#plt.plot(ww[0:50], abs(SS)[0:50] )

#Filtro
R1 = 0.03
C = 1
b = (0,1)
a = (R1*C,1)
B, A = signal.bilinear(b, a, fs)
y = signal.lfilter(B, A, ss)
Y = fft(y)
#plt.plot(ww[0:50], abs(Y)[0:50])
#plt.plot(t, y)

delta = np.zeros(NN)
delta[0] = 1
#plt.plot(t, delta)

h = signal.lfilter(B, A, delta)
#plt.plot(t, h)

H = fft(h)
plt.plot(ww[0:50], abs(H)[0:50])


#%%
###############################################################################
#VCO
###############################################################################
#Ejercicio 11
def fase(Ts,f,fase_inicial):
    return (fase_inicial+2*np.pi*f*Ts)%(2*np.pi)

f0= 1000
fm = 40
fs = 10*f0
N = 1000


#Ejercicio 12
xc=fuente(fm, 1/fs, 0, N) #Señal de control
plt.plot(np.linspace(0,N/fs,N),xc)
plt.show()

w = np.fft.fftfreq(N)
Xc= fft(xc)
plt.plot(w,abs(Xc))
plt.show()


#Ejercicio 13

K = 0
Vvco = np.zeros(N)
ph = 0
for i in range(N):
    Vvco[i] = np.sin(ph)
    ph = fase(1/fs, f0 + K * xc[i], ph)

plt.plot(np.linspace(0,N/fs,N), Vvco)
plt.show()
plt.plot( abs(fft(Vvco)) )
plt.show()


#Ejercicio 14
K = 100
Vvco = np.zeros(N)
ph = 0
for i in range(N):
    Vvco[i] = np.sin(ph)
    ph = fase(1/fs, f0 + K * xc[i], ph)

plt.plot(np.linspace(0,N/fs,N), Vvco)
plt.show()
plt.plot( abs(fft(Vvco)) )
plt.show()















#end
