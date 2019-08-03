

__all__ = ["SpheroidScattering"]

from scipy.integrate import quadrature, quad
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib

from scipy.special import pro_rad1, pro_rad2, pro_ang1
import specfun


class SpheroidScattering:
    
    def __init__(self, tipRadius, length):
        self.tipRadius = tipRadius
        self.length = length
        
        self.a, self.b, self.ksi, self.d = self.Get_ProlateSpheroidParameters(tipRadius, length)
        
        self.N_max = 8

    def SetFrequency(self, freq):
        self.freq = freq

        self.lambda0 = constants.c / freq
        self.k = 2.0*np.pi/self.lambda0
        self.c = self.k*self.d/2.0

    def SetIncidentAngle(self, theta):
        self.theta = theta
        
    def SetFieldAmp(self, E0):
        self.E0 = E0
        
    def SetNumberOfHarmonics(self, N_max):
        self.N_max = N_max

    def Map2DIndexTo1D(self, m_0, m_1, ind_start = 0):
        ## m = m_0 .. m_1-1   n = m ... m_1-1
        map2DTo1D, map1DTo2D = {}, {}
        ind = ind_start
        for m in range(m_0, m_1):
            for n in range(m, m_1):
                map2DTo1D[(m,n)] = ind
                map1DTo2D[ind] = (m, n)
                ind += 1
        return map2DTo1D, map1DTo2D
            
    def GetIncExpansionCoeffs_Amn(self, m, n):
        E0, k, theta_0, c = self.E0, self.k, self.theta, self.c
        eps_m = 2.0
        if m==0:
            eps_m = 1.0
        N_mn = GetInt_Sm_mpn_Sm_mpN(c, m, n-m, n-m)
        A_mn = 2.0* eps_m * pro_ang1(m, n, c, np.cos(theta_0))[0] / N_mn
        
        j_nm1 = 1j**((n-1)%4)
        
        return E0/k * j_nm1 * A_mn

    def ConstructMatrix(self):
        k, theta_0, ksi_0, c_0 = self.k, self.theta, self.ksi, self.c
        E0 = self.E0
        N_t = self.N_max
        
        alphaInd_2DTo1D, alphaInd_1DTo2D = self.Map2DIndexTo1D(0, N_t)
        n_total = len(alphaInd_2DTo1D)
        betaInd_2DTo1D, betaInd_1DTo2D = self.Map2DIndexTo1D(1, N_t+1, n_total)
        n_total += len(betaInd_2DTo1D)
        gammaInd = [n_total + i for i in range(N_t - 1)]
        n_total += len(gammaInd)
        

        ##construct coefficient marix
        A = np.zeros((n_total, n_total), dtype=complex)
        b = np.zeros(n_total, dtype=complex)
        
        ## eta: cos(m*phi) m=1..Nt
        for m in range(0, N_t):
            for N in range(m, N_t):
                ind_row = alphaInd_2DTo1D[(m, N)]
                for n in range(m, N_t):
                    ind_col = alphaInd_2DTo1D[(m, n)]
                    elem = ((ksi_0**2 - 1)*GetDerivativeRadialFunc(4, m, n, c_0, ksi_0) \
                            - ksi_0*m*GetRadialFunc(4, m, n, c_0, ksi_0)) \
                            * GetInt_Sm_mpn_Sm_mpN(c_0, m, n-m, N-m)
                                        
                    A[ind_row, ind_col] += elem

                    ind_col = betaInd_2DTo1D[(m+1, n+1)]
                    elem = -2.0*np.sqrt(ksi_0**2 - 1)*(m+1)*GetRadialFunc(4, m+1, n+1, c_0, ksi_0) \
                            * GetInt_Smp1_mpnp1_Sm_mpN_x_div_sqrt_1mx2(c_0, m, n-m, N-m)
                    
                    A[ind_row, ind_col] += elem
                
                    ##---- rhs
                    A_mn = self.GetIncExpansionCoeffs_Amn(m, n)
                    b[ind_row] -= A_mn * \
                                  ( \
                                      -ksi_0*m*GetRadialFunc(1, m, n, c_0, ksi_0) \
                                      + (ksi_0**2 - 1)*GetDerivativeRadialFunc(1, m, n, c_0, ksi_0) \
                                  ) * GetInt_Sm_mpn_Sm_mpN(c_0, m, n-m, N-m)
                    A_mp2np2 = self.GetIncExpansionCoeffs_Amn(m+2, n+2)
                    b[ind_row] -= A_mp2np2 * \
                                  ( \
                                      ksi_0*(m+2)*GetRadialFunc(1, m+2, n+2, c_0, ksi_0) \
                                      + (ksi_0**2 - 1)*GetDerivativeRadialFunc(1, m+2, n+2, c_0, ksi_0) \
                                  ) * GetInt_Smp2_mpnp2_Sm_mpN(c_0, m, n-m, N-m)
                    if m==0:
                        A_0n = self.GetIncExpansionCoeffs_Amn(0, n)
                        b[ind_row] -= A_0n * (ksi_0**2 - 1)*GetDerivativeRadialFunc(1, m, n, c_0, ksi_0) \
                                           * GetInt_Sm_mpn_Sm_mpN(c_0, m, n-m, N-m)
                    
                
        ## eta: cos(0*phi)
        for N in range(N_t - 1):
            ind_row = gammaInd[N]
            for n in range(N_t - 1):
                ind_col = gammaInd[n]
                
                elem = (-(ksi_0**2 - 1)*GetDerivativeRadialFunc(4, 1, n+1, c_0, ksi_0) \
                        - ksi_0*1*GetRadialFunc(4, 1, n+1, c_0, ksi_0)) \
                        * GetInt_Sm_mpn_Sm_mpN(c_0, 1, n, N)

                A[ind_row, ind_col] += elem
                
                ## rhs
                A_1np1 = self.GetIncExpansionCoeffs_Amn(1, n+1)
                b[ind_row] -= A_1np1 * \
                                  ( \
                                      ksi_0*GetRadialFunc(1, 1, n+1, c_0, ksi_0) \
                                      + (ksi_0**2 - 1)*GetDerivativeRadialFunc(1, 1, n+1, c_0, ksi_0) \
                                  ) * GetInt_Sm_mpn_Sm_mpN(c_0, 1, n, N)
                        
                
        ## phi: sin(m*phi), m=1...Nt-2
        for m in range(0, N_t):
            for N in range(m, N_t):
                ind_row = betaInd_2DTo1D[(m+1, N+1)]
                for n in range(m, N_t):
                    ind_col = alphaInd_2DTo1D[(m, n)]
                    elem = (ksi_0**2 - 1)*GetDerivativeRadialFunc(4, m, n, c_0, ksi_0) \
                            * GetInt_Sm_mpn_Sm_mpN_x(c_0, m, n-m, N-m) \
                        +  \
                            ksi_0 * GetRadialFunc(4, m, n, c_0, ksi_0) \
                            * GetInt_dxSm_mpn_Sm_mpN_1mx2(c_0, m, n-m, N-m) 
                    
                    A[ind_row, ind_col] += elem

                    ind_col = betaInd_2DTo1D[(m+1, n+1)]
                    elem = 2.0*np.sqrt(ksi_0**2 - 1) * \
                            ( \
                                GetRadialFunc(4, m+1, n+1, c_0, ksi_0) \
                                * GetInt_dxSmp1_mpnp1_Sm_mpN_x_sqrt_1mx2(c_0, m, n-m, N-m) \
                            - \
                                ksi_0 * GetDerivativeRadialFunc(4, m+1, n+1, c_0, ksi_0) \
                                * GetInt_Smp1_mpnp1_Sm_mpN_sqrt_1mx2(c_0, m, n-m, N-m) \
                            )

                    A[ind_row, ind_col] += elem

                    ##---- rhs
                    A_mn = self.GetIncExpansionCoeffs_Amn(m, n)
                    b[ind_row] -= A_mn * \
                        ( \
                           ksi_0*GetRadialFunc(1, m, n, c_0, ksi_0)*GetInt_dxSm_mpn_Sm_mpN_1mx2(c_0, m, n-m, N-m) \
                           + (ksi_0**2 - 1)*GetDerivativeRadialFunc(1, m, n, c_0, ksi_0) \
                                           *GetInt_Sm_mpn_Sm_mpN_x(c_0, m, n-m, N-m) \
                        )
                    A_mp2np2 = self.GetIncExpansionCoeffs_Amn(m+2, n+2)
                    b[ind_row] += A_mp2np2 * \
                        ( \
                           ksi_0*GetRadialFunc(1, m+2, n+2, c_0, ksi_0)*GetInt_dxSmp2_mpnp2_Sm_mpN_1mx2(c_0, m, n-m, N-m) \
                           + (ksi_0**2 - 1)*GetDerivativeRadialFunc(1, m+2, n+2, c_0, ksi_0) \
                                           *GetInt_Smp2_mpnp2_Sm_mpN_x(c_0, m, n-m, N-m) \
                        )
                    if m==0:
                        A_0n = self.GetIncExpansionCoeffs_Amn(0, n)
                        b[ind_row] -= A_0n * \
                        ( \
                           ksi_0*GetRadialFunc(1, m, n, c_0, ksi_0)*GetInt_dxSm_mpn_Sm_mpN_1mx2(c_0, m, n-m, N-m) \
                           + (ksi_0**2 - 1)*GetDerivativeRadialFunc(1, m, n, c_0, ksi_0) \
                                           *GetInt_Sm_mpn_Sm_mpN_x(c_0, m, n-m, N-m) \
                        )                    

        
        return A, b

    def GetAlphaBetaGamma_from_X(self, x):
        N_t = self.N_max
        alphaInd_2DTo1D, alphaInd_1DTo2D = self.Map2DIndexTo1D(0, N_t)
        n_total = len(alphaInd_2DTo1D)
        n_end_alpha = n_total
        
        betaInd_2DTo1D, betaInd_1DTo2D = self.Map2DIndexTo1D(1, N_t+1, n_total)
        n_total += len(betaInd_2DTo1D)
        n_end_beta = n_total

        gammaInd = [n_total + i for i in range(N_t - 1)]
        n_total += len(gammaInd)
            
        alpha = np.zeros((N_t, N_t), dtype=complex)
        beta = np.zeros((N_t+1, N_t+1), dtype=complex)
        gamma = np.zeros(N_t, dtype=complex)
        
        for i in range(n_end_alpha):
            alpha[alphaInd_1DTo2D[i]] = x[i]
        print(betaInd_1DTo2D)
        for i in range(n_end_alpha, n_end_beta):
            beta[betaInd_1DTo2D[i]] = x[i]
        for i in range(n_end_beta, len(x)):
            gamma[i - n_end_beta + 1] = x[i]
            
        return alpha, beta, gamma
                
    def Get_ProlateSpheroidParameters(self, tipRadius, length):
        b2_div_a = tipRadius
        a = length/2.0
        b = np.sqrt(b2_div_a * a)
        
        # d*ksi = a     d*sqrt(ksi**2 - 1) = b
        # ksi**2 * (1 - (b/a)**2) = 1
        ksi = 1.0/(1.0 - (b/a)**2)
        d = a / ksi
        
        return a, b, ksi, d
        

    def GetETMonSurface_direct(self, etas, ksi_0, phi_0):
        E_0, k, d = self.E0, self.k, self.d
    
        assert phi_0 == 0
        n = len(etas)
        E_eta = np.zeros(n, dtype=complex)
        E_ksi = np.zeros(n, dtype=complex)
        for i in range(n):
            eta = etas[i]
            z_hat_eta = ksi_0*np.sqrt((1 - eta**2)/(ksi_0**2 - eta**2))
            z_hat_ksi = eta * np.sqrt((ksi_0**2 - 1)/(ksi_0**2 - eta**2))
            x = d/2*np.sqrt(1 - eta**2)*np.sqrt(ksi_0**2 - 1)*np.cos(phi_0)
            E_eta[i] = E_0*np.exp(1j*k*x)*z_hat_eta
            E_ksi[i] = E_0*np.exp(1j*k*x)*z_hat_ksi
        return E_eta, E_ksi

    def GetETMonSurface_expansion(self, etas, ksi_0, phi_0):
        E_0, k, d, c_0 = self.E0, self.k, self.d, self.c
        
        assert phi_0 == 0
        theta_0 = np.pi/2
        n_eta = len(etas)
        E_eta = np.zeros(n_eta, dtype=complex)
        E_ksi = np.zeros(n_eta, dtype=complex)
        N = self.N_max
        for i in range(n_eta):
            eta = etas[i]
            for m in range(N):
                for n in range(m, N):
                    A_mn = self.GetIncExpansionCoeffs_Amn(m, n)
                    E_eta[i] += A_mn * 2*(ksi_0**2 - 1)*GetDerivativeRadialFunc(1, m, n, c_0, ksi_0) \
                                         *pro_ang1(m, n, c_0, eta)[0] \
                                        /(d*np.sqrt(ksi_0**2 - eta**2)*np.sqrt(ksi_0**2 - 1))
                    E_ksi[i] += A_mn * (-2)*(1 - eta**2)*pro_ang1(m, n, c_0, eta)[1] \
                                        *GetRadialFunc(1, m, n, c_0, ksi_0) \
                                        /(d*np.sqrt(ksi_0**2 - eta**2)*np.sqrt(1 - eta**2))
        return E_eta, E_ksi


    def GetFieldOnSurface_(self, alpha, beta, gamma, etas, ksi, phi):
        c, d = self.c, self.d
        k = self.k
        E0 = self.E0
        
        n_pts = len(etas)
        E_eta = np.zeros(n_pts, dtype=complex)
        E_ksi = np.zeros(n_pts, dtype=complex)
        E_phi = np.zeros(n_pts, dtype=complex)
        for i in range(n_pts):
            eta = etas[i]
            M, N = alpha.shape
            for m in range(M):
                for n in range(m, N):
                    E_eta[i] += alpha[m, n]*GetM_mplus1n_o_plus_eta(eta, ksi, phi, m, n, c, d)
                    E_ksi[i] += alpha[m, n]*GetM_mplus1n_o_plus_ksi(eta, ksi, phi, m, n, c, d)
                    E_phi[i] += alpha[m, n]*GetM_mplus1n_o_plus_phi(eta, ksi, phi, m, n, c, d)

            M, N = beta.shape
            for m in range(M):
                for n in range(m, N):
                    E_eta[i] += beta[m, n]*GetM_mn_o_z_eta(eta, ksi, phi, m, n, c, d)
                    E_ksi[i] += beta[m, n]*GetM_mn_o_z_ksi(eta, ksi, phi, m, n, c, d)
                    E_phi[i] += beta[m, n]*GetM_mn_o_z_phi(eta, ksi, phi, m, n, c, d)

            N = len(gamma)
            for n in range(1,N):
                E_eta[i] += gamma[n]*GetM_mminus1n_o_minus_eta(eta, ksi, phi, 1, n, c, d)
                E_ksi[i] += gamma[n]*GetM_mminus1n_o_minus_ksi(eta, ksi, phi, 1, n, c, d)
                E_phi[i] += gamma[n]*GetM_mminus1n_o_minus_phi(eta, ksi, phi, 1, n, c, d)
                  
        return E_eta, E_ksi, E_phi

    def GetFieldOnSurface(self, alpha, beta, gamma, etas, ksi, phi, totalField=True):
        c, d = self.c, self.d
        k = self.k
        E0 = self.E0
        
        n_pts = len(etas)
        E_eta = np.zeros(n_pts, dtype=complex)
        E_ksi = np.zeros(n_pts, dtype=complex)
        E_phi = np.zeros(n_pts, dtype=complex)
        for i in range(n_pts):
            eta = etas[i]
            M, N = alpha.shape
            for m in range(M):
                for n in range(m, N):
                    E_eta[i] += alpha[m, n]*GetM_mplus1n_o_plus_eta(eta, ksi, phi, m, n, c, d)
                    E_ksi[i] += alpha[m, n]*GetM_mplus1n_o_plus_ksi(eta, ksi, phi, m, n, c, d)
                    E_phi[i] += alpha[m, n]*GetM_mplus1n_o_plus_phi(eta, ksi, phi, m, n, c, d)

            M, N = beta.shape
            for m in range(M):
                for n in range(m, N):
                    E_eta[i] += beta[m, n]*GetM_mn_o_z_eta(eta, ksi, phi, m, n, c, d)
                    E_ksi[i] += beta[m, n]*GetM_mn_o_z_ksi(eta, ksi, phi, m, n, c, d)
                    E_phi[i] += beta[m, n]*GetM_mn_o_z_phi(eta, ksi, phi, m, n, c, d)

            N = len(gamma)
            for n in range(1,N):
                E_eta[i] += gamma[n]*GetM_mminus1n_o_minus_eta(eta, ksi, phi, 1, n, c, d)
                E_ksi[i] += gamma[n]*GetM_mminus1n_o_minus_ksi(eta, ksi, phi, 1, n, c, d)
                E_phi[i] += gamma[n]*GetM_mminus1n_o_minus_phi(eta, ksi, phi, 1, n, c, d)

            if totalField:
                z_hat_eta = ksi*np.sqrt((1 - eta**2)/(ksi**2 - eta**2))
                z_hat_ksi = eta * np.sqrt((ksi**2 - 1)/(ksi**2 - eta**2))
                x = d/2*np.sqrt(1 - eta**2)*np.sqrt(ksi**2 - 1)*np.cos(phi)
                E_eta[i] += E0*np.exp(1j*k*x)*z_hat_eta
                E_ksi[i] += E0*np.exp(1j*k*x)*z_hat_ksi
                
        return E_eta, E_ksi, E_phi


    def GetFieldAtCartesianPoints(self, alpha, beta, gamma, r_pts, totalField=True):
        ## r = [x, y, z]
        
        a, c, d = self.a, self.c, self.d
        k = self.k
        E0 = self.E0
        
        n_pts = len(r_pts)
        E_eta = np.zeros(n_pts, dtype=complex)
        E_ksi = np.zeros(n_pts, dtype=complex)
        E_phi = np.zeros(n_pts, dtype=complex)
        for i in range(n_pts):
            x, y, z = r_pts[i]
            eta, ksi, phi = self.CoordinatePointTransformRectToSpheroid(x, y, z)
            M, N = alpha.shape
            for m in range(M):
                for n in range(m, N):
                    E_eta[i] += alpha[m, n]*GetM_mplus1n_o_plus_eta(eta, ksi, phi, m, n, c, d)
                    E_ksi[i] += alpha[m, n]*GetM_mplus1n_o_plus_ksi(eta, ksi, phi, m, n, c, d)
                    E_phi[i] += alpha[m, n]*GetM_mplus1n_o_plus_phi(eta, ksi, phi, m, n, c, d)

            M, N = beta.shape
            for m in range(M):
                for n in range(m, N):
                    E_eta[i] += beta[m, n]*GetM_mn_o_z_eta(eta, ksi, phi, m, n, c, d)
                    E_ksi[i] += beta[m, n]*GetM_mn_o_z_ksi(eta, ksi, phi, m, n, c, d)
                    E_phi[i] += beta[m, n]*GetM_mn_o_z_phi(eta, ksi, phi, m, n, c, d)

            N = len(gamma)
            for n in range(1,N):
                E_eta[i] += gamma[n]*GetM_mminus1n_o_minus_eta(eta, ksi, phi, 1, n, c, d)
                E_ksi[i] += gamma[n]*GetM_mminus1n_o_minus_ksi(eta, ksi, phi, 1, n, c, d)
                E_phi[i] += gamma[n]*GetM_mminus1n_o_minus_phi(eta, ksi, phi, 1, n, c, d)

            if totalField:
                z_hat_eta = ksi*np.sqrt((1 - eta**2)/(ksi**2 - eta**2))
                z_hat_ksi = eta * np.sqrt((ksi**2 - 1)/(ksi**2 - eta**2))
                #x = d/2*np.sqrt(1 - eta**2)*np.sqrt(ksi**2 - 1)*np.cos(phi)
                E_eta[i] += E0*np.exp(1j*k*x)*z_hat_eta
                E_ksi[i] += E0*np.exp(1j*k*x)*z_hat_ksi
                
        return E_eta, E_ksi, E_phi
                    
    def CoordinatePointTransformSpheroidToRect(self, eta, ksi, phi):
        d = self.d
        x = d/2*sqrt((1 - eta**2))*sqrt((ksi**2 - 1))*cos(phi)
        y = d/2*sqrt((1 - eta**2))*sqrt((ksi**2 - 1))*sin(phi)
        z = d/2*eta*ksi
        return x, y, z

    def CoordinatePointTransformRectToSpheroid(self, x, y, z):
        d = self.d
        ksi = (np.sqrt(x**2 + y**2 + (z + d/2)**2) + np.sqrt(x**2 + y**2 + (z - d/2)**2))/d
        eta = (np.sqrt(x**2 + y**2 + (z + d/2)**2) - np.sqrt(x**2 + y**2 + (z - d/2)**2))/d
        phi = np.arctan2(y, x)
        return eta, ksi, phi
        
    def PlotFieldAroundTipAtXZPlane(self, Dx, Dz, nx, nz, alpha, beta, gamma, totalField=True):
        ksi_0 = self.ksi
        a, c, d = self.a, self.c, self.d
        E0, k = self.E0, self.k
    
        x = np.linspace(-Dx/2, Dx/2, nx)
        z = np.linspace(a/2-Dz/2, a/2+Dz/2, nz)
        X, Z = np.meshgrid(x, z, indexing="ij")
        
        r_pts = []
        r_inds = []
        for i in range(nx):
            x_i = x[i]
            for j in range(nz):
                z_j = z[j]
                eta, ksi, phi = self.CoordinatePointTransformRectToSpheroid(x_i, 0.0, z_j)
                if ksi > ksi_0:
                    r_pts.append([x_i, 0.0, z_j])
                    r_inds.append((i, j))
        E_eta, E_ksi, E_phi = self.GetFieldAtCartesianPoints(alpha, beta, gamma, r_pts, totalField)
        
        E_ksi_mesh = np.zeros(X.shape, dtype=complex)
        for i in range(len(r_inds)):
            E_ksi_mesh[r_inds[i]] = E_ksi[i]
            
        E_ksi_mesh *= np.isfinite(E_ksi_mesh)
        
        #plt.imshow(np.log10(np.abs(E_ksi_mesh.T) + 1.0e-100), origin="lower", vmin=-10)
        plt.imshow(np.abs(E_ksi_mesh.T), origin="lower", cmap="rainbow", aspect="auto", 
                   extent=np.array([-Dx/2, Dx/2, -Dz/2, Dz/2])/constants.nano)
        plt.colorbar()
        plt.xlabel(r'x (nm)', fontsize=18)
        plt.ylabel(r'z (nm)', fontsize=18)
        plt.title(r'R={:.2f} nm, f={:.2f} THz, L={:.2f} $\mu$m'.format(self.tipRadius/constants.nano, self.freq/constants.tera, self.length/constants.micro), fontsize=18)
        plt.savefig("../out/tip_field.png", bbox_inches='tight', pad_inches=0.5)
        plt.show()

    def GetAlphaBetaGamma(self, vbose=False):
        A, b = self.ConstructMatrix()
        
        if vbose:
            cond = np.linalg.cond(A)
            print("condition: ", cond)

        x = np.linalg.solve(A, b)
        
        if vbose:
            print("error: ", np.linalg.norm(A.dot(x) - b))

        alpha, beta, gamma = self.GetAlphaBetaGamma_from_X(x)
        
        return alpha, beta, gamma

    def GetTipEnhancement(self, plotMatrices=False, plotFields=False, varifySurfaceField=False):
        print("a = {},  b = {},  ksi = {},  d = {} ".format(self.a, self.b, self.ksi, self.d))
        print("c_0 = {} ".format(self.c))

        A, b = self.ConstructMatrix()

        cond = np.linalg.cond(A)
        print("condition: ", cond)

        x = np.linalg.solve(A, b)
        print("error: ", np.linalg.norm(A.dot(x) - b))

        alpha, beta, gamma = self.GetAlphaBetaGamma_from_X(x)

        if plotMatrices:
            plt.imshow(np.log10(np.abs(alpha) + 1.0e-100))
            plt.colorbar()
            plt.show()
            plt.imshow(np.log10(np.abs(beta) + 1.0e-100))
            plt.colorbar()
            plt.show()

        if varifySurfaceField:
            eps = 1.0e-5
            etas = np.linspace(0, 1.0-eps, 20, endpoint=True)
            phi_0 = 0.0
            ksi_0 = self.ksi
            d = self.d
            E_eta0, E_ksi0 = self.GetETMonSurface_direct(etas, ksi_0, phi_0)
            E_eta1, E_ksi1 = self.GetETMonSurface_expansion(etas, ksi_0, phi_0)
            
            E_eta2, E_ksi2, _ = self.GetFieldOnSurface_(alpha, beta, gamma, etas, ksi_0, phi_0)

            
            #np.set_printoptions(precision=5)
            print('='*50)
            for i in range(len(E_eta0)):
                print("{:.4e} {:.4e} {:.4e} || {:.5f}".format(E_eta0[i], E_eta1[i], E_eta2[i], abs(E_eta2[i]/E_eta0[i])))
            print('-'*50)
            for i in range(len(E_eta0)):
                print("{:.3e} {:.3e} {:.3e} || {:.3f}".format(E_ksi0[i], E_ksi1[i], E_ksi2[i], abs(E_ksi2[i])))
        
        if plotFields:
            self.PlotFieldAroundTipAtXZPlane(Dx=4.0*self.tipRadius, Dz=4.0*self.tipRadius, nx=20, nz=20, 
                                        alpha=alpha, beta=beta, gamma=gamma, 
                                        totalField=True)
                                        
        
        _, E_ksi, _ = self.GetFieldOnSurface(alpha, beta, gamma, etas=[1.0 - 1.0e-5], ksi=self.ksi, phi=0.0)

        return E_ksi[-1], A, b

    def GetTipEnhancementFromMatrix(self, alpha, beta, gamma, varifySurfaceField=False):
        if varifySurfaceField:
            eps = 1.0e-5
            etas = np.linspace(0, 1.0-eps, 20, endpoint=True)
            phi_0 = 0.0
            ksi_0 = self.ksi
            E_eta0, E_ksi0 = self.GetETMonSurface_direct(etas, ksi_0, phi_0)
            E_eta1, E_ksi1 = self.GetETMonSurface_expansion(etas, ksi_0, phi_0)
            
            E_eta2, E_ksi2, _ = self.GetFieldOnSurface(alpha, beta, gamma, etas, ksi_0, phi_0, totalField=False)

            
            #np.set_printoptions(precision=5)
            print('='*50)
            for i in range(len(E_eta0)):
                print("{:.4e} {:.4e} {:.4e} || {:.5f}".format(E_eta0[i], E_eta1[i], E_eta2[i], abs(E_eta2[i]/E_eta0[i])))
            print('-'*50)
            for i in range(len(E_eta0)):
                print("{:.3e} {:.3e} {:.3e} || {:.3f}".format(E_ksi0[i], E_ksi1[i], E_ksi2[i], abs(E_ksi2[i])))

        _, E_ksi, _ = self.GetFieldOnSurface(alpha, beta, gamma, etas=[1.0 - 1.0e-5], ksi=self.ksi, phi=0.0)
        return E_ksi[-1]

    def GetTipEnhancementOverFrequencyBand(self, freq, Nt):
        assert len(freq)==len(Nt)
        enhancement = np.zeros(len(freq), dtype=complex)
        for i in range(len(freq)):
            self.SetFrequency(freq[i])
            self.SetNumberOfHarmonics(Nt[i])
            enhancement[i] = self.GetTipEnhancement(plotMatrices=True, plotFields=True, varifySurfaceField=True)[0]
            print("i:{}   Nt:{}   f:{},    e:{}".format(i, Nt[i], freq[i]/constants.tera, np.abs(enhancement[i])))
            print('='*50)
            print('='*50)
        return enhancement

    def GetAlphaBetaGammaOverFrequencyBand(self, freq, Nt):
        assert len(freq) == len(Nt)
        matrices = []
        for i in range(len(freq)):
            self.SetFrequency(freq[i])
            self.SetNumberOfHarmonics(Nt[i])
            matrices.append(self.GetAlphaBetaGamma())        
        return matrices


def GetRadialFunc(i, m, n, c, x):
    if i == 1:
        return pro_rad1(m, n, c, x)[0]
    elif i == 2:
        return pro_rad2(m, n, c, x)[0]
    elif i == 3:
        return pro_rad1(m, n, c, x)[0] + 1j*pro_rad2(m, n, c, x)[0]
    elif i == 4:
        return pro_rad1(m, n, c, x)[0] - 1j*pro_rad2(m, n, c, x)[0]
    else:
        assert False

def GetDerivativeRadialFunc(i, m, n, c, x):
    if i == 1:
        return pro_rad1(m, n, c, x)[1]
    elif i == 2:
        return pro_rad2(m, n, c, x)[1]
    elif i == 3:
        return pro_rad1(m, n, c, x)[1] + 1j*pro_rad2(m, n, c, x)[1]
    elif i == 4:
        return pro_rad1(m, n, c, x)[1] - 1j*pro_rad2(m, n, c, x)[1]
    else:
        assert False

def GetM_mplus1n_o_plus_eta(eta, ksi, phi, m, n, c, d):
    return  ((ksi**2 - 1.0)*GetDerivativeRadialFunc(4, m, n, c, ksi) - ksi*m*GetRadialFunc(4, m, n, c, ksi))*pro_ang1(m, n, c, eta)[0]*np.cos(phi*(m + 1))/(d*np.sqrt(-eta**2 + ksi**2)*np.sqrt(ksi**2 - 1))
def GetM_mplus1n_o_plus_ksi(eta, ksi, phi, m, n, c, d):
    return ((eta**2 - 1)*pro_ang1(m, n, c, eta)[1] - eta*m*pro_ang1(m, n, c, eta)[0])*GetRadialFunc(4, m, n, c, ksi)*np.cos(phi*(m + 1))/(d*np.sqrt(-eta**2 + 1)*np.sqrt(-eta**2 + ksi**2))
def GetM_mplus1n_o_plus_phi(eta, ksi, phi, m, n, c, d):
    return -(eta*(ksi**2 - 1)*pro_ang1(m, n, c, eta)[0]*GetDerivativeRadialFunc(4, m, n, c, ksi) - ksi*(eta**2 - 1)*GetRadialFunc(4, m, n, c, ksi)*pro_ang1(m, n, c, eta)[1])*np.sin(phi*m + phi)/(d*(eta**2 - ksi**2))

def GetM_mminus1n_o_minus_eta(eta, ksi, phi, m, n, c, d):
    return  ((1.0 - ksi**2)*GetDerivativeRadialFunc(4, m, n, c, ksi) - ksi*m*GetRadialFunc(4, m, n, c, ksi))*pro_ang1(m, n, c, eta)[0]*np.cos(phi*(m - 1))/(d*np.sqrt(-eta**2 + ksi**2)*np.sqrt(ksi**2 - 1))
def GetM_mminus1n_o_minus_ksi(eta, ksi, phi, m, n, c, d):
    return  ((1.0-eta**2)*pro_ang1(m, n, c, eta)[1] - eta*m*pro_ang1(m, n, c, eta)[0])*GetRadialFunc(4, m, n, c, ksi)*np.cos(phi*(m - 1))/(d*np.sqrt(-eta**2 + 1)*np.sqrt(-eta**2 + ksi**2))
def GetM_mminus1n_o_minus_phi(eta, ksi, phi, m, n, c, d):
    return  -(eta*(ksi**2 - 1)*pro_ang1(m, n, c, eta)[0]*GetDerivativeRadialFunc(4, m, n, c, ksi) - ksi*(eta**2 - 1)*GetRadialFunc(4, m, n, c, ksi)*pro_ang1(m, n, c, eta)[1])*np.sin(phi*(m - 1))/(d*(eta**2 - ksi**2))

def GetM_mn_o_z_eta(eta, ksi, phi, m, n, c, d):
    return  -2*eta*m*GetRadialFunc(4, m, n, c, ksi)*pro_ang1(m, n, c, eta)[0]*np.cos(phi*m)/(d*np.sqrt(-eta**2 + 1)*np.sqrt(-eta**2 + ksi**2))
def GetM_mn_o_z_ksi(eta, ksi, phi, m, n, c, d):
    return  2*ksi*m*GetRadialFunc(4, m, n, c, ksi)*pro_ang1(m, n, c, eta)[0]*np.cos(phi*m)/(d*np.sqrt(-eta**2 + ksi**2)*np.sqrt(ksi**2 - 1))
def GetM_mn_o_z_phi(eta, ksi, phi, m, n, c, d):
    return  2*np.sqrt(-eta**2 + 1)*np.sqrt(ksi**2 - 1)*(-eta*GetRadialFunc(4, m, n, c, ksi)*pro_ang1(m, n, c, eta)[1] + ksi*pro_ang1(m, n, c, eta)[0]*GetDerivativeRadialFunc(4, m, n, c, ksi))*np.sin(phi*m)/(d*(eta**2 - ksi**2))



def Get_Int_PmnPmn(m, n):
    if m > n:
        return 0
    fact_mpn_div_fact_mmn = 1.0
    for i in range(n + m, max(n - m, 0), -1):
        fact_mpn_div_fact_mmn *= i
    return 2.0/(2*n + 1)*fact_mpn_div_fact_mmn

def Get_Int_Pm2muPmnu(m, mu, nu):
    if m + 2 > mu or m > nu:
        return 0
    if mu < nu:
        return 0.0
    elif mu == nu:
        fact_div = 1.0
        for i in range(nu + m, max(nu - m - 2, 0), -1):
            fact_div *= i        
        return -2.0/(2.0*nu + 1)*fact_div
    elif (mu + nu) % 2 == 0:
        fact_div = 1.0
        for i in range(nu + m, max(nu - m, 0), -1):
            fact_div *= i                
        return 4*(m + 1)*fact_div
    else:
        return 0.0
    
def Get_Int_PmnPmn_div_1mx2(m, n):
    if m > n:
        return 0
    if m!=0:
        fact_div = 1.0
        for i in range(n + m, max(n - m, 0), -1):
            fact_div *= i                
        return fact_div / m
    else:
        return np.inf
        

def Expand_P_minusm_n(m, n):
    coeff = 0.0
    if abs(m) <= n:
        fact_mpn_div_fact_mmn = 1.0
        for i in range(n + m, max(n - m, 0), -1):
            fact_mpn_div_fact_mmn *= i        
        coeff = 1.0 / fact_mpn_div_fact_mmn
        if m % 2 == 1:
            coeff *= -1.0
    return {(m, n):coeff}

def Expand_xPmn(m, n):
    return {(m, n-1):(n + m)/(2*n + 1), (m, n+1):(n - m + 1)/(2*n + 1)}

def Expand_x_div_sqrt_1mx2_Pmn(m, n):
    return {(m-1, n):-(n + m)*(n - m + 1)/(2*m), (m+1, n):-1/(2*m)}
    
def Expand_1_div_sqrt_1mx2_Pmn__nm1(m, n):
    return {(m-1, n-1):-(n + m)*(n + m - 1)/(2*m), (m+1, n-1):-1/(2*m)}

def Expand_1_div_sqrt_1mx2_Pmn__np1(m, n):
    return {(m-1, n+1):-(n - m + 1)*(n - m + 2)/(2*m), (m+1, n+1):-1/(2*m)}

def Expand_1mx2_dxPmn(m, n):
    return {(m, n-1):(n + 1)*(n + m)/(2*n + 1), (m, n+1):-n*(n - m + 1)/(2*n + 1)}

def Expand_sqrt_1mx2_Pmn__mm1(m, n):
    return {(m-1, n-1):-(n + m)*(n + m - 1)/(2*n + 1), (m-1, n+1):(n - m + 1)*(n - m + 2)/(2*n + 1)}

def Expand_sqrt_1mx2_Pmn__mp1(m, n):
    return {(m+1, n-1):1/(2*n + 1), (m+1, n+1):-1/(2*n + 1)}

def Expand_sqrt_1mx2_dxPmn(m, n):
    return {(m-1, n):(n + m)*(n - m + 1)/2, (m+1, n):-1.0/2}

import copy

def Get_Expansion_Coeff_d_mn(m, n, c, thresh_rel=1.0e-14):
    #m0, n0, c0 = copy.deepcopy(m), copy.deepcopy(n), copy.deepcopy(c)
    cv = specfun.segv(m, n, c, 1)[0]
    #m0, n0, c0 = copy.deepcopy(m), copy.deepcopy(n), copy.deepcopy(c)
    d_mn_ = specfun.sdmn(m, n, c, cv, 1)

    d_mn = d_mn_.copy()

    if m % 2 == 1:
        d_mn *= -1.0
        
    d_inds = None
    if (n - m) % 2 == 0:
        d_inds = np.array([2*i for i in range(len(d_mn))])
    else:
        d_inds = np.array([2*i+1 for i in range(len(d_mn))])
        
    if thresh_rel != 0.0:
        d_mn_max = np.max(np.abs(d_mn))
        filt = np.abs(d_mn) > d_mn_max*thresh_rel
        d_mn = d_mn[filt]
        d_inds = d_inds[filt]
    else:
        ind_max = np.argmax(d_mn==0.0)
        if ind_max > 0:
            d_mn = d_mn[:ind_max]
            d_inds = d_inds[:ind_max]
        
    return d_mn, d_inds

## I_1mNn
def GetInt_Sm_mpn_Sm_mpN(c, m, n, N):
    if n == N:
        d_mn, d_inds = Get_Expansion_Coeff_d_mn(m, m+n, c)
        I = 0.0
        for i in range(len(d_mn)):
            I += d_mn[i]**2 * Get_Int_PmnPmn(m, m+d_inds[i])
        return I
    else:
        return 0.0

def GetInt_Sm_mpn_Sm_mpN_quad1(c, m, n, N):
    def f(x):
        return pro_ang1(m, m+n, c, x)[0]*pro_ang1(m, m+N, c, x)[0]
    return quad(f, -1, 1)

def GetInt_Sm_mpn_Sm_mpN_quad2(c, m, n, N):
    def f(x):
        cv_n = specfun.segv(m, m+n, c, 1)[0]
        cv_N = specfun.segv(m, m+N, c, 1)[0]
        return specfun.aswfa(m, m+n, c, x, 1, cv_n)[0]*specfun.aswfa(m, m+N, c, x, 1, cv_N)[0]
    return quad(f, -1, 1)

## I_2mNn
def GetInt_Smp1_mpnp1_Sm_mpN_x_div_sqrt_1mx2(c, m, n, N):
    d_q, d_q_inds = Get_Expansion_Coeff_d_mn(m+1, m+n+1, c)
    m_q = m + 1
    d_r, d_r_inds = Get_Expansion_Coeff_d_mn(m, m+N, c)
    m_r = m
    len_q = len(d_q_inds)
    len_r = len(d_r_inds)
    
    I = 0.0
    for q in range(len(d_q)):
        q_expansion = Expand_x_div_sqrt_1mx2_Pmn(m_q, m_q + d_q_inds[q])
        for mn_pair in q_expansion:
            m_q_expan, n_q_expan = mn_pair
            coeff = q_expansion[mn_pair]
            if m_q_expan == m_r:
                for r in range(len(d_r)):
                    if n_q_expan == m_r + d_r_inds[r]:
                        I += coeff*d_q[q]*d_r[r]*Get_Int_PmnPmn(m_q_expan, n_q_expan)
            elif m_q_expan == m_r + 2:
                for r in range(len(d_r)):
                    I += coeff*d_q[q]*d_r[r]*Get_Int_Pm2muPmnu(m_r, n_q_expan, m_r + d_r_inds[r])
            else:
                assert False
    return I

def GetInt_Smp1_mpnp1_Sm_mpN_x_div_sqrt_1mx2_quad1(c, m, n, N):
    def f(x):
        return pro_ang1(m+1, m+n+1, c, x)[0]*pro_ang1(m, m+N, c, x)[0]*x/np.sqrt(1 - x**2)
    return quad(f, -1, 1)

def GetInt_Smp1_mpnp1_Sm_mpN_x_div_sqrt_1mx2_quad2(c, m, n, N):
    def f(x):
        cv_n = specfun.segv(m+1, m+n+1, c, 1)[0]
        cv_N = specfun.segv(m, m+N, c, 1)[0]
        return specfun.aswfa(m+1, m+n+1, c, x, 1, cv_n)[0]*specfun.aswfa(m, m+N, c, x, 1, cv_N)[0]*x/np.sqrt(1 - x**2)
    return quad(f, -1, 1)

## I_3mNn
def GetInt_Sm_mpn_Sm_mpN_x(c, m, n, N):
    d_q, d_q_inds = Get_Expansion_Coeff_d_mn(m, m+n, c)
    m_q = m
    d_r, d_r_inds = Get_Expansion_Coeff_d_mn(m, m+N, c)
    m_r = m
    len_q = len(d_q_inds)
    len_r = len(d_r_inds)
    
    I = 0.0
    for q in range(len(d_q)):
        q_expansion = Expand_xPmn(m_q, m_q + d_q_inds[q])
        for mn_pair in q_expansion:
            m_q_expan, n_q_expan = mn_pair
            coeff = q_expansion[mn_pair]
            if m_q_expan == m_r:
                for r in range(len(d_r)):
                    if n_q_expan == m_r + d_r_inds[r]:
                        I += coeff*d_q[q]*d_r[r]*Get_Int_PmnPmn(m_q_expan, n_q_expan)
            else:
                assert False
    return I

def GetInt_Sm_mpn_Sm_mpN_x_quad1(c, m, n, N):
    def f(x):
        return pro_ang1(m, m+n, c, x)[0]*pro_ang1(m, m+N, c, x)[0]*x
    return quad(f, -1, 1)

def GetInt_Sm_mpn_Sm_mpN_x_quad2(c, m, n, N):
    def f(x):
        cv_n = specfun.segv(m, m+n, c, 1)[0]
        cv_N = specfun.segv(m, m+N, c, 1)[0]
        return specfun.aswfa(m, m+n, c, x, 1, cv_n)[0]*specfun.aswfa(m, m+N, c, x, 1, cv_N)[0]*x
    return quad(f, -1, 1)

## I_4mNn
def GetInt_dxSm_mpn_Sm_mpN_1mx2(c, m, n, N):
    d_q, d_q_inds = Get_Expansion_Coeff_d_mn(m, m+n, c)
    m_q = m
    d_r, d_r_inds = Get_Expansion_Coeff_d_mn(m, m+N, c)
    m_r = m
    len_q = len(d_q_inds)
    len_r = len(d_r_inds)
    
    I = 0.0
    for q in range(len(d_q)):
        q_expansion = Expand_1mx2_dxPmn(m_q, m_q + d_q_inds[q])
        for mn_pair in q_expansion:
            m_q_expan, n_q_expan = mn_pair
            coeff = q_expansion[mn_pair]
            if m_q_expan == m_r:
                for r in range(len(d_r)):
                    if n_q_expan == m_r + d_r_inds[r]:
                        I += coeff*d_q[q]*d_r[r]*Get_Int_PmnPmn(m_q_expan, n_q_expan)
            else:
                assert False
    return I

def GetInt_dxSm_mpn_Sm_mpN_1mx2_quad1(c, m, n, N):
    def f(x):
        return pro_ang1(m, m+n, c, x)[1]*pro_ang1(m, m+N, c, x)[0]*(1 - x**2)
    return quad(f, -1, 1)

def GetInt_dxSm_mpn_Sm_mpN_1mx2_quad2(c, m, n, N):
    def f(x):
        cv_n = specfun.segv(m, m+n, c, 1)[0]
        cv_N = specfun.segv(m, m+N, c, 1)[0]
        return specfun.aswfa(m, m+n, c, x, 1, cv_n)[1]*specfun.aswfa(m, m+N, c, x, 1, cv_N)[0]*(1 - x**2)
    return quad(f, -1, 1)

## I_5mNn
def GetInt_Smp1_mpnp1_Sm_mpN_sqrt_1mx2(c, m, n, N):
    d_q, d_q_inds = Get_Expansion_Coeff_d_mn(m+1, m+n+1, c)
    m_q = m + 1
    d_r, d_r_inds = Get_Expansion_Coeff_d_mn(m, m+N, c)
    m_r = m
    len_q = len(d_q_inds)
    len_r = len(d_r_inds)
    
    I = 0.0
    for q in range(len(d_q)):
        q_expansion = Expand_sqrt_1mx2_Pmn__mm1(m_q, m_q + d_q_inds[q])
        for mn_pair in q_expansion:
            m_q_expan, n_q_expan = mn_pair
            coeff = q_expansion[mn_pair]
            if m_q_expan == m_r:
                for r in range(len(d_r)):
                    if n_q_expan == m_r + d_r_inds[r]:
                        I += coeff*d_q[q]*d_r[r]*Get_Int_PmnPmn(m_q_expan, n_q_expan)
            else:
                assert False
    return I

def GetInt_Smp1_mpnp1_Sm_mpN_sqrt_1mx2_quad1(c, m, n, N):
    def f(x):
        return pro_ang1(m+1, m+n+1, c, x)[0]*pro_ang1(m, m+N, c, x)[0]*np.sqrt(1 - x**2)
    return quad(f, -1, 1)

def GetInt_Smp1_mpnp1_Sm_mpN_sqrt_1mx2_quad2(c, m, n, N):
    def f(x):
        cv_n = specfun.segv(m+1, m+n+1, c, 1)[0]
        cv_N = specfun.segv(m, m+N, c, 1)[0]
        return specfun.aswfa(m+1, m+n+1, c, x, 1, cv_n)[0]*specfun.aswfa(m, m+N, c, x, 1, cv_N)[0]*np.sqrt(1 - x**2)
    return quad(f, -1, 1)

## I_6mNn
def GetInt_dxSmp1_mpnp1_Sm_mpN_x_sqrt_1mx2(c, m, n, N):
    d_q, d_q_inds = Get_Expansion_Coeff_d_mn(m+1, m+n+1, c)
    m_q = m + 1
    d_r, d_r_inds = Get_Expansion_Coeff_d_mn(m, m+N, c)
    m_r = m
    len_q = len(d_q_inds)
    len_r = len(d_r_inds)
    
    I = 0.0
    for q in range(len(d_q)):
        q_expansion = Expand_sqrt_1mx2_dxPmn(m_q, m_q + d_q_inds[q])
        for mn_q_pair in q_expansion:
            m_q_expan, n_q_expan = mn_q_pair
            coeff_q = q_expansion[mn_q_pair]
            if m_q_expan == m_r:
                for r in range(len(d_r)):
                    r_expansion = Expand_xPmn(m_r, m_r + d_r_inds[r])
                    for mn_r_pair in r_expansion:
                        m_r_expan, n_r_expan = mn_r_pair
                        coeff_r = r_expansion[mn_r_pair]
                        assert m_r_expan == m_q_expan
                        if n_q_expan == n_r_expan:
                            I += coeff_q*coeff_r*d_q[q]*d_r[r]*Get_Int_PmnPmn(m_q_expan, n_q_expan)
            elif m_q_expan == m_r + 2:
                for r in range(len(d_r)):
                    r_expansion = Expand_xPmn(m_r, m_r + d_r_inds[r])
                    for mn_r_pair in r_expansion:
                        m_r_expan, n_r_expan = mn_r_pair
                        coeff_r = r_expansion[mn_r_pair]
                        assert m_r_expan + 2 == m_q_expan
                        I += coeff_q*coeff_r*d_q[q]*d_r[r]*Get_Int_Pm2muPmnu(m_r_expan, n_q_expan, n_r_expan)
            else:
                assert False
    return I

def GetInt_dxSmp1_mpnp1_Sm_mpN_x_sqrt_1mx2_quad1(c, m, n, N):
    def f(x):
        return pro_ang1(m+1, m+n+1, c, x)[1]*pro_ang1(m, m+N, c, x)[0]*x*np.sqrt(1 - x**2)
    return quad(f, -1, 1)

def GetInt_dxSmp1_mpnp1_Sm_mpN_x_sqrt_1mx2_quad2(c, m, n, N):
    def f(x):
        cv_n = specfun.segv(m+1, m+n+1, c, 1)[0]
        cv_N = specfun.segv(m, m+N, c, 1)[0]
        return specfun.aswfa(m+1, m+n+1, c, x, 1, cv_n)[1]*specfun.aswfa(m, m+N, c, x, 1, cv_N)[0]*x*np.sqrt(1 - x**2)
    return quad(f, -1, 1)

## I_7mNn
def GetInt_Smp2_mpnp2_Sm_mpN(c, m, n, N):
    d_q, d_q_inds = Get_Expansion_Coeff_d_mn(m+2, m+n+2, c)
    m_q = m + 2
    d_r, d_r_inds = Get_Expansion_Coeff_d_mn(m, m+N, c)
    m_r = m
    len_q = len(d_q_inds)
    len_r = len(d_r_inds)
    
    I = 0.0
    for q in range(len(d_q)):
        for r in range(len(d_r)):
            I += d_q[q]*d_r[r]*Get_Int_Pm2muPmnu(m_r, m_q + d_q_inds[q], m_r + d_r_inds[r])
    return I

def GetInt_Smp2_mpnp2_Sm_mpN_quad1(c, m, n, N):
    def f(x):
        return pro_ang1(m+2, m+n+2, c, x)[0]*pro_ang1(m, m+N, c, x)[0]
    return quad(f, -1, 1)

def GetInt_Smp2_mpnp2_Sm_mpN_quad2(c, m, n, N):
    def f(x):
        cv_n = specfun.segv(m+2, m+n+2, c, 1)[0]
        cv_N = specfun.segv(m, m+N, c, 1)[0]
        return specfun.aswfa(m+2, m+n+2, c, x, 1, cv_n)[0]*specfun.aswfa(m, m+N, c, x, 1, cv_N)[0]
    return quad(f, -1, 1)

## I_8mNn
def GetInt_Smp2_mpnp2_Sm_mpN_x(c, m, n, N):
    d_q, d_q_inds = Get_Expansion_Coeff_d_mn(m+2, m+n+2, c)
    m_q = m + 2
    d_r, d_r_inds = Get_Expansion_Coeff_d_mn(m, m+N, c)
    m_r = m
    len_q = len(d_q_inds)
    len_r = len(d_r_inds)
    
    I = 0.0
    for q in range(len(d_q)):
        q_expansion = Expand_xPmn(m_q, m_q + d_q_inds[q])
        for mn_pair in q_expansion:
            m_q_expan, n_q_expan = mn_pair
            coeff = q_expansion[mn_pair]
            if m_q_expan == m_r + 2:
                for r in range(len(d_r)):
                    I += coeff*d_q[q]*d_r[r]*Get_Int_Pm2muPmnu(m_r, n_q_expan, m_r + d_r_inds[r])
            else:
                assert False
    return I

def GetInt_Smp2_mpnp2_Sm_mpN_x_quad1(c, m, n, N):
    def f(x):
        return pro_ang1(m+2, m+n+2, c, x)[0]*pro_ang1(m, m+N, c, x)[0]*x
    return quad(f, -1, 1)

def GetInt_Smp2_mpnp2_Sm_mpN_x_quad2(c, m, n, N):
    def f(x):
        cv_n = specfun.segv(m+2, m+n+2, c, 1)[0]
        cv_N = specfun.segv(m, m+N, c, 1)[0]
        return specfun.aswfa(m+2, m+n+2, c, x, 1, cv_n)[0]*specfun.aswfa(m, m+N, c, x, 1, cv_N)[0]*x
    return quad(f, -1, 1)

## I_9mNn
def GetInt_dxSmp2_mpnp2_Sm_mpN_1mx2(c, m, n, N):
    d_q, d_q_inds = Get_Expansion_Coeff_d_mn(m+2, m+n+2, c)
    m_q = m + 2
    d_r, d_r_inds = Get_Expansion_Coeff_d_mn(m, m+N, c)
    m_r = m
    len_q = len(d_q_inds)
    len_r = len(d_r_inds)
    
    I = 0.0
    for q in range(len(d_q)):
        q_expansion = Expand_1mx2_dxPmn(m_q, m_q + d_q_inds[q])
        for mn_pair in q_expansion:
            m_q_expan, n_q_expan = mn_pair
            coeff = q_expansion[mn_pair]
            if m_q_expan == m_r + 2:
                for r in range(len(d_r)):
                    I += coeff*d_q[q]*d_r[r]*Get_Int_Pm2muPmnu(m_r, n_q_expan, m_r + d_r_inds[r])
            else:
                assert False
    return I

def GetInt_dxSmp2_mpnp2_Sm_mpN_1mx2_quad1(c, m, n, N):
    def f(x):
        return pro_ang1(m+2, m+n+2, c, x)[1]*pro_ang1(m, m+N, c, x)[0]*(1 - x**2)
    return quad(f, -1, 1)

def GetInt_dxSmp2_mpnp2_Sm_mpN_1mx2_quad2(c, m, n, N):
    def f(x):
        cv_n = specfun.segv(m+2, m+n+2, c, 1)[0]
        cv_N = specfun.segv(m, m+N, c, 1)[0]
        return specfun.aswfa(m+2, m+n+2, c, x, 1, cv_n)[1]*specfun.aswfa(m, m+N, c, x, 1, cv_N)[0]*(1 - x**2)
    return quad(f, -1, 1)

## I_10Nn
def GetInt_S0_n_S1_1pN_sqrt_1mx2(c, n, N):
    d_q, d_q_inds = Get_Expansion_Coeff_d_mn(0, n, c)
    m_q = 0
    d_r, d_r_inds = Get_Expansion_Coeff_d_mn(1, 1+N, c)
    m_r = 1
    len_q = len(d_q_inds)
    len_r = len(d_r_inds)
    
    I = 0.0
    for q in range(len(d_q)):
        q_expansion = Expand_sqrt_1mx2_Pmn__mp1(m_q, m_q + d_q_inds[q])
        for mn_pair in q_expansion:
            m_q_expan, n_q_expan = mn_pair
            coeff = q_expansion[mn_pair]
            if m_q_expan == m_r:
                for r in range(len(d_r)):
                    if n_q_expan == m_r + d_r_inds[r]:
                        I += coeff*d_q[q]*d_r[r]*Get_Int_PmnPmn(m_q_expan, n_q_expan)
            else:
                assert False
    return I

def GetInt_S0_n_S1_1pN_sqrt_1mx2_quad1(c, n, N):
    def f(x):
        return pro_ang1(0, n, c, x)[0]*pro_ang1(1, 1+N, c, x)[0]*np.sqrt(1 - x**2)
    return quad(f, -1, 1)

def GetInt_S0_n_S1_1pN_sqrt_1mx2_quad2(c, n, N):
    def f(x):
        cv_n = specfun.segv(0, n, c, 1)[0]
        cv_N = specfun.segv(1, 1+N, c, 1)[0]
        return specfun.aswfa(0, n, c, x, 1, cv_n)[0]*specfun.aswfa(1, 1+N, c, x, 1, cv_N)[0]*np.sqrt(1 - x**2)
    return quad(f, -1, 1)

## I_11Nn
def GetInt_dxS0_n_S1_1pN_sqrt_x_sqrt_1mx2(c, n, N):
    d_q, d_q_inds = Get_Expansion_Coeff_d_mn(0, n, c)
    m_q = 0
    d_r, d_r_inds = Get_Expansion_Coeff_d_mn(1, 1+N, c)
    m_r = 1
    len_q = len(d_q_inds)
    len_r = len(d_r_inds)
    
    I = 0.0
    for q in range(len(d_q)):
        q_expansion = Expand_sqrt_1mx2_dxPmn(m_q, m_q + d_q_inds[q])
        for mn_q_pair in q_expansion:
            m_q_expan, n_q_expan = mn_q_pair
            coeff_q = q_expansion[mn_q_pair]
            if m_q_expan < 0:
                minusm_expansion = Expand_P_minusm_n(abs(m_q_expan), n_q_expan)
                assert len(minusm_expansion) == 1
                m_q_expan, n_q_expan = list(minusm_expansion.keys())[0]
                coeff_q *= list(minusm_expansion.values())[0]
            if m_q_expan == m_r:
                for r in range(len(d_r)):
                    r_expansion = Expand_xPmn(m_r, m_r + d_r_inds[r])
                    for mn_r_pair in r_expansion:
                        m_r_expan, n_r_expan = mn_r_pair
                        coeff_r = r_expansion[mn_r_pair]
                        assert m_r_expan == m_q_expan
                        if n_q_expan == n_r_expan:
                            I += coeff_q*coeff_r*d_q[q]*d_r[r]*Get_Int_PmnPmn(m_q_expan, n_q_expan)
            else:
                assert False
    return I

def GetInt_dxS0_n_S1_1pN_sqrt_x_sqrt_1mx2_quad1(c, n, N):
    def f(x):
        return pro_ang1(0, n, c, x)[1]*pro_ang1(1, 1+N, c, x)[0]*x*np.sqrt(1 - x**2)
    return quad(f, -1, 1)

def GetInt_dxS0_n_S1_1pN_sqrt_x_sqrt_1mx2_quad2(c, n, N):
    def f(x):
        cv_n = specfun.segv(0, n, c, 1)[0]
        cv_N = specfun.segv(1, 1+N, c, 1)[0]
        return specfun.aswfa(0, n, c, x, 1, cv_n)[1]*specfun.aswfa(1, 1+N, c, x, 1, cv_N)[0]*x*np.sqrt(1 - x**2)
    return quad(f, -1, 1)



