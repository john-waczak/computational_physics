import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import bisect

# define everything needed for your
# potential function here
#-----------------------------------------------
# plt.figure(figsize=(20,10))
# xs = np.linspace(-4*L, 4*L, 1000)
# plt.plot(xs, V(xs), 'b')
# plt.xlabel('x')
# plt.ylabel('V(x)')
# plt.show()

#-----------------------------------------------
# start off with a guess for the energy
def eigenPsi(E, Rmax, xmatch, N=5000, outputWavefunction=False):
    # define constants for this particle
    C = 2  # 2 * m / hbar^2
    energy = E

    # define potential function

    L = 10  # width of square well
    V0 = 16  # depth of square well
    def V(x):
        V = np.zeros_like(x)
        V[np.abs(x)<= L] = -V0
        V[np.abs(x)> L] = 0
        return V

    # write Schroding ODE as two coupled first
    # order ODEs
    def SE(x, psi, E):
        phi = np.zeros((2), float)
        phi[0] = psi[1]
        phi[1] = -(C*(E-V(x)))*psi[0]
        return phi

    def meetInMiddle(N, E):
        NL = int((Rmax+xmatch)/(2*Rmax)*N)
        NR = N-NL

        # assuming decaying exponential behavior at extremes
        phiL_0 = [1.E-15]
        phiL_0.append(np.sqrt(-E*C)*phiL_0[0])
        # reverse the slope for odd solutions
        phiR_0 = [1.E-15]
        phiR_0.append(-np.sqrt(-E*C)*phiR_0[0])

        xL = np.linspace(-Rmax, xmatch, NL)
        xR = np.flip(np.linspace(xmatch, Rmax, NR))

        phiL = solve_ivp(SE, [-Rmax, xmatch], phiL_0, t_eval=xL, args=(E,))
        phiR = solve_ivp(SE, [Rmax, xmatch], phiR_0, t_eval=xR, args=(E,))

        left = phiL.y[1, -1]/phiL.y[0, -1]
        right = phiR.y[1, -1]/phiR.y[0, -1]

        if not outputWavefunction:
            return (left-right)
        else:
            return (left-right), phiL, phiR

    return meetInMiddle(N, energy)



# #here's a demo of how to find the groundstate 
E = -16
E0 = bisect(eigenPsi, E*1.1, E/1.1, args=(50, 0))
print(E0)
error, phiL, phiR = eigenPsi(E0, 50, 0, outputWavefunction=True)


plt.figure(figsize=(20, 10))
plt.plot(phiL.t, phiL.y[0, :])
plt.plot(phiR.t, phiR.y[0, :])
plt.title('Ground-state wave function for V(x) with energy Eg = {}'.format(E0))
plt.show()
