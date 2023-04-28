import numpy as np
from math import factorial as f
from scipy.special import hyp2f1 as F
from scipy.constants import pi
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.integrate import odeint
import matplotlib.ticker as tic
from numpy import log
import time as t

def states (i):
    g = np.sum(2*(2*(np.linspace(0,i-1,i))+1))
    return g

def sigmav_ion(n):

    I = 13.606*1.60218*10**-12/n**2             #ionization energy of level n
    E = np.linspace(I, 100*Te, 9999)            # energy, erg
    f_E = 2*pi**(-0.5)*Te**(-1.5)*E**(0.5)*np.exp(-E/Te)  #distribution function
    sigma_ion=np.zeros(len(E))                  #ionizarion cross-section, cm^2

    for i in range(len(E)):
        sigma_ion[i] = IonCrossSection(n, E[i])

    return trapz((2*E/me)**0.5*sigma_ion*f_E, E)


def IonCrossSection(n, E):

    I = 13.606*1.60218*10**-12/n**2

    if n == 1:
        A0, A1, A2 = 0.18450, -0.032226, -0.034539
        A3, A4, A5 = 1.4003, -2.8115, 2.2986
        mean = 2.567074*10**-37/(I*E)*(A0*log(E/I)+A1*(1-I/E)+A2*(1-I/E)**2+A3*(1-I/E)**3+A4*(1-I/E)**4+A5*(1-I/E)**5)
    elif n == 2:
        A0, A1, A2 = 0.14784, 0.0080871, -0.062270
        A3, A4, A5 = 1.9414, -2.1980, 0.95894
        mean = 2.567074*10**-37/(I*E)*(A0*log(E/I)+ A1*(1-I/E)+A2*(1-I/E)**2+A3*(1-I/E)**3+A4*(1-I/E)**4+A5*(1-I/E)**5)
    elif n == 3:
        A0, A1, A2 =0.058463, -0.051272, 0.85310
        A3, A4, A5 = -0.57014, 0.76684, 0
        mean = 2.567074*10**-37/(I*E)*(A0*log(E/I)+A1*(1-I/E)+A2*(1-I/E)**2+A3*(1-I/E)**3+A4*(1-I/E)**4+A5*(1-I/E)**5)
    elif E-I < 0:
        mean = 0
    else:
        x,r = E/I, 1.94/n**1.57
        g0 = 0.9935 + 0.2328/n - 0.1296/n**2
        g1 = -1/n*(0.6282 - 0.5598/n + 0.5299/n**2)
        g2 = 1/n**2*(0.3887 - 1.181/n + 1.470/n**2)
        A = 32*n/(3**1.5*pi)*(g0/3 + g1/4 + g2/5)
        b = 1/n*(4.0 - 18.63/n + 36.24/n**2 - 28.09/n**3)
        B = 2/3*n**2*(5+b)
        mean = 1.76*n**2/x*(1-np.exp(-r*x))*( A*log(x) + (B-A*log(2*n**2))*(1-1/x)**2 )*10**-16

    return mean


def sigmav_exc (i, k):                  #excitation cross-section, cm^2
    l=1   
    if i != k:

        if i > k:
            n = i
            n1 = k
        else:
            n = k
            n1 = i
        
        if n == 2 and n1 == 1:
            Y_col = 0.3
        elif n1 == 1:
            Y_col = 0.2
        else:
            Y_col = 0.1

        delta_E = Rye*(1/n1**2 - 1/n**2)            #energy between 2 levels, erg
        E = np.linspace(delta_E, 100*Te, 9999)
        sigma = np.empty(len(E))
        f_E = 2*pi**(-0.5)*Te**(-1.5)*E**(0.5)*np.exp(-E/Te)
            
        for j in range(len(E)):
            a1 = -n+l+1
            a2 = -n+l-1
            b = -n1+l
            c_F = 2*l
            z = -4*n*n1/((n-n1)**2)
            Z = 1
            u = (E[j]-delta_E)/delta_E
            nu = Ry*(1/(n1**2)-1/(n**2))
    
            R = (-1)**(n1-1)/(4*f(2*l-1))*(f(n+l)*f(n1+l-1)/(f(n-l-1)*f(n1-l)))**(1/2)*(4*n*n1)**(l+1)*(n-n1)**(n+n1-2*l-2)/((n+n1)**(n+n1))*(F(a1,b,c_F,z)-((n-n1)/(n+n1))**2*F(a2,b,c_F,z))/(Z) #дипольный момент
            os = 1/3*(nu/Ry)*(l/(2*(l-1)+1))*R**2         #oscilator force
                
            sigma[j] = 4*pi*a0**2*os*(Rye/delta_E)**2*(u*log(1.25*(u+1))/(u+1)**2+Y_col/(u+1)) #(6)

        if i > k:
            return states(n1)/states(n)*np.exp(13.606*(1.60218*10**-12)*(n1**-2-n**-2)/Te)*trapz(sigma*f_E*(2*E/me)**0.5, E)
        else:
            return trapz(sigma*f_E*(2*E/me)**0.5, E)       
        
def Aik(i,k):       #calculating of Einstein coefficients
    l = 1
    if i > k:
        n = i
        n1 = k
                
        a1 = -n+l+1
        a2 = -n+l-1
        b = -n1+l
        c_F = 2*l
        z = -4*n*n1/((n-n1)**2)
        Z = 1
        nu = Ry*(1/(n1**2)-1/(n**2))
    
        R = (-1)**(n1-1)/(4*f(2*l-1))*(f(n+l)*f(n1+l-1)/(f(n-l-1)*f(n1-l)))**(1/2)*(4*n*n1)**(l+1)*(n-n1)**(n+n1-2*l-2)/((n+n1)**(n+n1))*(F(a1,b,c_F,z)-((n-n1)/(n+n1))**2*F(a2,b,c_F,z))/(Z)
        A = (64*pi**4*nu**3/(3*h*c**3))*(l/(2*l+1))*e**2*a0**2*(R)**2

        return A

def system (y, t, M, N):
    dydt = np.zeros(N)
   
    for i in range(N):
        for j in range(N):
           dydt[i] = dydt[i]+M[i, j]*y[j]
    
    return dydt

def CRM_sol(ne, T, sigma_M, A_M):                                   #CRM and coronal solutions function
    global M_ion
    M = np.zeros([N,N])
    M_ion = np.zeros([N,N])

    for i in range(N):
        for j in range(N):        
            if i == j:
                M[i,j] = -np.sum(ne*sigma_M[:,i])-np.sum(A_M[:,i])
                M_ion[i,j] = M[i,j]-ne*sigmav_ion(i+1)
            elif i > j:                               #backward process
                M[i,j] = ne*sigma_M[i,j]
                M_ion[i,j] = M[i,j]
            elif i < j:                               #strait process
                M[i,j] = A_M[i,j]+ne*sigma_M[i,j]
                M_ion[i,j] = M[i,j]

    t_step = []
    sol_ion_l = [N0]

    step=8*int((T/10**(-8)))        #количество шагов

    t = np.linspace(0, T, step+1)
    print('step in time: ', t[1]-t[0])

    sol = odeint(system, N0, t, args=(M, N), full_output=0)
    sol_ion = N0

    for p in range(step):
        sol_ion = odeint(system, sol_ion, [t[p],t[p+1]], args=(M_ion, N), full_output=0)
        sol_ion = sol_ion[-1]/np.sum(sol_ion[-1])

        sol_ion_l.append(sol_ion)
                                           
        
    print('ODE solution without ionization\n', sol[-1,:])
    print('ODE solution with ionization\n', sol_ion_l[-1])

    return sol, np.array(sol_ion_l), t

def coronal_sol(M_coronal, N, N0_coronal):

    for i in range(N):
        for j in range(N):        
            if i == j:
                M_coronal[i,j] = -np.sum(ne*sigma_M_coronal[:,i])-np.sum(A_M_coronal[:,i])
            elif i > j:                               #- process
                M_coronal[i,j] = ne*sigma_M_coronal[i,j]
            elif i < j:                               #+ process
                M_coronal[i,j] = A_M_coronal[i,j]+ne*sigma_M_coronal[i,j]

    M_coronal[0] = np.ones(N)
    return np.linalg.solve(M_coronal, N0_coronal)
    
               
a0 = 5.292*10**(-9)                           #cm
Ry = 3.28*10**15                            #Hz
Rye = 2.1799*10**-11                        #erg
c = 3*10**10                                  #cm/s
h = 6.63*10**-27                              #g*cm**2/c
e = 4.8*10**-10
e_si = 1.6*10**-19
me = 9.31*10**-28                             #g
me_si = 9.31**10**-31
T = 10**-4                                   #характерное время
N = 30                                      #число рассматриваемых уровней
N0 = np.zeros(N)                            #начальная населённость
N0[0] = 1.0
N0_coronal = np.zeros(N+1)
N0_coronal[-1] = 1.0

Te0 = 5                                   #eV
Te = Te0*(1.60218*10**-12)                #erg температура электронов

M_coronal=np.zeros([N,N])

try:
    sigma_M = np.loadtxt("sigma_M.txt")
    A_M = np.loadtxt("A_M.txt")
    sigma_M_coronal = np.loadtxt("sigma_M_coronal.txt")
    A_M_coronal = np.loadtxt("A_M_coronal.txt")

except:
    sigma_M = np.zeros([N, N])
    A_M = np.zeros([N, N])

    sigma_M_coronal = np.zeros([N, N])
    A_M_coronal = np.zeros([N, N])


    for i in range(N):
        for j in range(N):
            if i != j:                                 #for back process i!=j, for strait i>j
                sigma_M[i,j]=sigmav_exc(j+1,i+1)
            if i < j:
                A_M[i,j]=Aik(j+1, i+1)
        print('Matrix of A and sigma are filled on {0:.1f}%'.format(i/(N-1)*100))
    print('\n')

    np.savetxt("sigma_M.txt", sigma_M)
    np.savetxt("A_M.txt", A_M)

    for i in range(N):
        for j in range(N):

            if j == 0 and i > j:                                 #for back process i!=j, for strait i>j
                sigma_M_coronal[i,j] = sigmav_exc(j+1,i+1)

            if i < j:
                A_M_coronal[i,j] = Aik(j+1, i+1)
        print('Matrix of A and sigma for coronal are filled on {0:.1f}%'.format(i/(N)*100))

        np.savetxt("sigma_M_coronal.txt", sigma_M)
        np.savetxt("A_M_coronal.txt", A_M)

N_pop = []
N_pop_ion = []
for i in range(5):
    N_pop.append([])
    N_pop_ion.append([])
    
ne_array = np.empty(1)
fig, ax = plt.subplots(1,1)
p=0

ne=10**(8+2*p)                         #cm^-3 не больше 10^20

start_time = t.time()
sol, sol_ion, time = CRM_sol(ne, T, sigma_M, A_M)

print("Time of computing results\n--- %.4f seconds ---" % (t.time() - start_time))

sol_coronal=coronal_sol(M_coronal, N, N0)

print('Quality of norm:\nwithout ionization %f, with ionization %f, coronal %f\n'
      % (np.sum(sol[-1,:]), np.sum(sol_ion[-1,:]), np.sum(sol_coronal)))

print('N_cor/N_sol: ', sol_coronal[:]/sol[-1,:], '\nN_cor/N_sol_ion ', sol_coronal[:]/sol_ion[-1,:])

ax.plot(np.linspace(1, N, N), log(sol[-1,:]),np.linspace(1, N, N), log((sol_ion[-1,:]))
        ,np.linspace(1, N, N), log(sol_coronal[:]), linewidth=5)  
ax.legend(['without ionization', 'with ionization', 'coronal'])
ax.set_title('ODE solution of CRM for $n_{e}$ = '+str('%.2e' % (ne))+' $cm^{-3}$ and $T_{e}=$'+str(Te0)+' eV')
ax.set_xlabel('n', fontsize=20)
ax.set_ylabel('$ln(N)$, %', fontsize=20)
ax.xaxis.set_major_locator(tic.MultipleLocator(1))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(25)
    #ax[p].yaxis.set_major_locator(tic.MultipleLocator(0.1))
ax.grid()

fig2, ax2=plt.subplots()
for i in range(1,6):
    N_pop[i-1]=log(sol[:,i+(i-1)*6])
    N_pop_ion[i-1]=log(sol_ion[:,i+(i-1)*6])
    ax2.plot(time, N_pop[i-1], time, N_pop_ion[i-1])
    
#ax2.plot(time, N_pop[0],time, N_pop[1],time, N_pop[2],time, N_pop[3],time, N_pop[4],
         #time, N_pop_ion[0],time, N_pop_ion[1],time, N_pop_ion[2],time, N_pop_ion[3],time, N_pop_ion[4])

ax2.legend(['2 level', '9 level', '16 level', '23 level', '30 level',
            '2 level_ion', '9 level_ion', '16 level_ion', '23 level_ion', '30 level_ion'], loc='right')
ax2.set_title('Population on levels, $T_{e}=$'+str(Te0)+' eV', fontsize=20)
ax2.set_xlabel('$t, c}$')
ax2.set_ylabel('ln(N), %')
#ax2.xaxis.set_major_locator(tic.MultipleLocator(1))
#ax2.yaxis.set_major_locator(tic.MultipleLocator(0.01))
ax2.grid()

fig,ax3=plt.subplots()
ax3.set_xlabel("n", fontsize=25)
ax3.set_ylabel("$N_{coronal}/N_{CRM}$", fontsize=25)
ax3.set_title('$N_{coronal}/N_{CRM}$ for $n_{e}$ = '+str('%.2e' % (ne))+" $cm^{-3}$", fontsize=20)

for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
	label.set_fontsize(25)

ax3.grid()
ax3.plot(np.linspace(1, N, N),sol_coronal[:]/sol_ion[-1,:], linewidth=5)

plt.show()