import numba
from numba import njit, objmode,jit
import numpy as np
from tqdm import tqdm
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)  # Silence unimportant NUMBA performance warning


au_to_fs = 2.418884326509E-2
fs_to_au = 1/au_to_fs

PhZtoeV = 4.13566553853
eVperHar = 27.2114079527

def Update(i, d, interference, psi, x, G, H, prec, steps, drive, energies, norms, osc_energy, damping_work, orth, sspen, mu_12, kappa, m_0, q_0, eta, gamma, comp_length, h):

    ### Calculate interference related terms at timestep t_i
    d_i = 0
    drive_i = 0

    for particle in (0,1):
        interference[i, particle]   = 2 * np.real(np.conj(psi[i, particle, 0]) * psi[i, particle, 1])
        d_i                         += mu_12 * interference[i, particle] 
        drive_i                     += kappa * mu_12 * q_0 * interference[i, particle] / m_0


    d[i]            = d_i
    drive[i]        = drive_i

    ### Calculate oscillator displacement at timestep t_i+1, x(t_0)=0
    x[i+1]          = Convolution_Trap(drive[:i+2], G[:i+2], h)

    ### Calculate Hamiltonian for timestep t_i
    offdiag         = -kappa * x[i] * q_0 * mu_12
    H[i,0,1]        = offdiag
    H[i,1,0]        = offdiag

    ### Calculate sp-energies and norms for timestep t_i
    for particle in (0,1):
        energies[i, particle]       = np.real(np.dot(np.conj(psi[i, particle, :]), np.dot(H[i, :, :], psi[i, particle, :])))
        norms[i, particle]          = np.real(np.dot(np.conj(psi[i, particle, :]), psi[i, particle, :]))
    
    orth[i] = np.abs(np.vdot(psi[i, 0, :], psi[i, 1, :]))

    ### Calculate oscillator displacement at timestep t_i+1/2
    x_mid           = 0.5 * (x[i] + x[i+1])

    ### Construct Hamiltonian at timestep t_i+1/2
    H_mid           = np.empty((2, 2), dtype=prec)

    offdiag         = -kappa * x_mid * q_0 * mu_12

    H_mid[0, 0]     = sspen[0]
    H_mid[1, 1]     = sspen[1]
    H_mid[0, 1]     = offdiag
    H_mid[1, 0]     = offdiag

    ### Propagate from timestep t_i to timestep t_i+1
    for particle in (0,1):
        psi[i+1, particle, :] = Propagate_eigh(H_mid, psi[i, particle, :], h)
        

    ### Calculate system 2 oscillator energies 
    x_dot_i             = (x[i+1] - x[i-1]) / (2.0 * h)
    damping_work[i]     = damping_work[i-1] + h * m_0 * gamma * x_dot_i**2
    osc_energy[i]       = 0.5 * m_0 * x_dot_i**2 + 0.5 * q_0 * eta * x[i]**2 



    return d, interference, psi, x, G, H, drive, energies, norms, osc_energy, damping_work, orth



def Prop_NUMBA(sspen, mu_12, kappa, m_0, q_0, eta, gamma, comp_length, h, c_list):
    
        prec            = np.complex128
        length          = comp_length * fs_to_au 
        steps           = int(np.floor(length / h)) + 1
        times           = np.arange(steps) * h

        H               = np.zeros((steps,2,2), dtype=prec)
        psi             = np.zeros((steps,2,2), dtype=prec)

        psi[0,0,0]      = c_list[0]
        psi[0,0,1]      = c_list[1]

        psi[0,1,0]      = c_list[2]
        psi[0,1,1]      = c_list[3]

        H[:,0,0]        = sspen[0] 
        H[:,1,1]        = sspen[1]

        d               = np.zeros((steps),   dtype=np.float64)
        interference    = np.zeros((steps,2), dtype=np.float64)
        x               = np.zeros((steps),   dtype=np.float64)
        drive           = np.zeros((steps),   dtype=np.float64)
        energies        = np.zeros((steps,2), dtype=np.float64)
        norms           = np.zeros((steps,2), dtype=np.float64)
        osc_energy      = np.zeros((steps),   dtype=np.float64)
        damping_work    = np.zeros((steps),   dtype=np.float64)
        orth            = np.zeros((steps),   dtype=np.float64)

        omega_0         = np.sqrt(q_0*eta/m_0) 
        G               = Greens_NUMBA(gamma, omega_0, steps, times)
        
        for i in tqdm(range(steps)[:-1]):
            d, interference, psi, x, G, H, drive, energies, norms, osc_energy, damping_work, orth = \
            Update(i, d, interference, psi, x, G, H, prec, steps, drive, energies, norms, osc_energy, damping_work, orth, \
                sspen, mu_12, kappa, m_0, q_0, eta, gamma, comp_length, h)


        return times, psi, d, interference, x, G, H, drive, energies, norms, osc_energy, damping_work, orth



def Greens_NUMBA(gamma, omega_0, steps, times):
    ### Calculate system 2 Green's function
    G = np.zeros(steps)
    sq = np.sqrt(omega_0**2 - (gamma/2)**2)
    i = 0
    for t in times:
        G[i] = np.exp(-0.5 * gamma * t) * (1 / sq) * np.sin(sq * (t))   
        i +=1

    return G




def Convolution_Rect(A, B, h):
    ### Perform convolution of A and B
    m = A.shape[0]
    C = 0
    for i in numba.prange(m):
        C += A[i] * B[-i-1]

    return C * h 

def Convolution_Trap(A, B, h):
    ### Perform convolution of A and B
    m = A.shape[0]
    C = 0
    for i in numba.prange(m-1):
        C += (A[i] * B[-i-1]) + (A[i+1] * B[-i-2])

    return C * h/2




def Propagate_eigh(H, psi, h):
    ### Diagonalize H, propagate exactly
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    amplitudes = np.conj(eigenvectors).T @ psi
    amplitudes *= np.exp(-1j * h * eigenvalues)

    return eigenvectors @ amplitudes


### Just-in-time compilation step of numerical functions, with convolution to be carried out in parallel

jitted = True
if jitted:
    Greens_NUMBA                = numba.njit(Greens_NUMBA )
    Convolution_Rect            = numba.njit(Convolution_Rect, parallel=True)
    Convolution_Trap            = numba.njit(Convolution_Trap, parallel=True)
    Update                      = numba.njit(Update)
    Propagate_eigh              = numba.njit(Propagate_eigh)

### System 1 parameters

sspen           = (-1.283, -0.669)
mu_12           = -0.61/2
kappa           = -0.11

rot             = 1e-4
P_1             = 1.9
P_2             = 2

c_1_1           = (P_1**0.5) * np.cos(rot)
c_1_2           = (P_1**0.5) * np.sin(rot)
c_2_1           = -(P_2**0.5) * np.sin(rot)
c_2_2           = (P_2**0.5) * np.cos(rot)

c_list = [c_1_1, c_1_2, c_2_1, c_2_2]

### System 2 parameters

m_0             = 6
q_0             = -6
eta             = -0.39
gamma           = 0.01

### Numerical Parameters 

comp_length     = 50    #(fs)
h               = 0.01  #(au)


times, psi, d, interference, x, G, H, drive, energies, norms, osc_energy, damping_work, orth = Prop_NUMBA(sspen, mu_12, kappa, m_0, q_0, eta, gamma, comp_length, h, c_list)

print(d+x)
