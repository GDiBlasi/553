import numpy as np
class Time_Step:
    """
    Differnt definitions of time stepping
    """
        
    @staticmethod
    def dt_2D(V, mu, dx, dy, CFL, ngc, R_s, gamma, T):

        u_mag = np.sqrt(V[ngc:-ngc, ngc:-ngc, 1]**2 + V[ngc:-ngc, ngc:-ngc, 2]**2)
        c_0 = np.sqrt(gamma*R_s*T[ngc:-ngc,ngc:-ngc])
        u_mag += c_0
        
        nu = mu/V[ngc:-ngc, ngc:-ngc, 0] 
        
        dt_adv  = min(CFL * dx / (np.max(np.amax(u_mag)) ), CFL * dy / np.max(np.amax(u_mag)))
        dt_diff = min(CFL * (dx*dx) / (np.max(np.amax(nu)) ), CFL * (dy*dy) / np.max(np.amax(nu)))
        
        dt = min(dt_adv, dt_diff)
        return dt

    @staticmethod
    def dt_diffusive(nu, dx, dy, CFL):
        """
        In 2D, but usually dx and dy are the same 
        Diffusive time step
        nu : kinematic viscosity 
        dx : grid spacing
        CFL : Courant-Friedrichs-Lewy number
        """
        
        dt_dx = CFL * dx**2 / nu
        dt_dy = CFL * dy**2 / nu
        dt = min(dt_dx, dt_dy)
        return dt

