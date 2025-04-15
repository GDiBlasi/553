import numpy as np
from GhostCells import Ghost_Cells 

class EquationOfState:
    # Equation of State (useful functions)

    @staticmethod
    def calculate_pressure_from_temperature(V, R, T, order):
        """
        Calculate pressure using the ideal gas law.
        
        rho = density
        R = specific gas constant
        T = temperature
        
        returns P = pressure
        
        """
        ngc  = Ghost_Cells.number_ghost_cells(order)
        
        # Initialize array using np.zeros
        P = np.zeros(np.shape(T))
        
        P[ngc:-ngc, ngc:-ngc] = V[ngc:-ngc, ngc:-ngc, 0] * R * T[ngc:-ngc, ngc:-ngc]
        
        P           = Ghost_Cells.fill_ghost_cells_2D_single(P, order)
        
        return P
    
    @staticmethod
    def calculate_temperature_from_pressure(V, R, P, order):
        """
        Calculate pressure using the ideal gas law.
        
        rho = density
        R = specific gas constant
        P = pressure
        
        returns T = temperature
        
        """
        ngc  = Ghost_Cells.number_ghost_cells(order)
        
        # Initialize array using np.zeros
        T = np.zeros((np.shape(P)))
        
        T[ngc:-ngc, ngc:-ngc] = P[ngc:-ngc, ngc:-ngc] / (R * V[ngc:-ngc, ngc:-ngc, 0])   
     
        T                     = Ghost_Cells.fill_ghost_cells_2D_single(T, order)
        
        return T
    
    @staticmethod
    def Pressure_from_internal_energy(V, e, gamma, order):
        """
        Calculate pressure using the equation of state.
        
        rho = density
        e = internal energy
        gamma = ratio of specific heats
        
        returns P = pressure
        
        """
        ngc = Ghost_Cells.number_ghost_cells(order)
        
        # Initialize array using np.zeros
        P = np.zeros((np.shape(e)))
        
        P[ngc:-ngc, ngc:-ngc] = (gamma - 1) * V[ngc:-ngc,ngc:-ngc,0] * e[ngc:-ngc, ngc:-ngc]
        
        P                     = Ghost_Cells.fill_ghost_cells_2D_single(P, order)
        
        return P
    
    @staticmethod
    def Temperature_from_internal_energy(e, C_v, order):
        """
        Calculate temperature using the specific heat capacity at constant volume.
        
        e = internal energy
        C_v = specific heat capacity at constant volume
        
        returns T = temperature
        
        """
        ngc = Ghost_Cells.number_ghost_cells(order)
        
        # Initialize array using np.zeros
        T = np.zeros(np.shape(e))
        
        T[ngc:-ngc, ngc:-ngc] = e[ngc:-ngc, ngc:-ngc] / C_v
        
        T           = Ghost_Cells.fill_ghost_cells_2D_single(T, order)
        
        return T
    
    @staticmethod
    def internal_energy_from_pressure(P, V, gamma, order):
        """
        Calculate internal energy using the equation of state.
        
        P = pressure
        V = density
        gamma = ratio of specific heats
        
        returns e = internal energy
        
        """
        ngc = Ghost_Cells.number_ghost_cells(order)
        
        # Initialize array using np.zeros
        e = np.zeros(np.shape(P))
        
        # The NONE is for broadcasting the values of the 2D on the RHS to all P in 3D in LHS
        e[ngc:-ngc, ngc:-ngc] = P[ngc:-ngc, ngc:-ngc] / ((gamma - 1) * V[ngc:-ngc, ngc:-ngc,0])
        
        e           = Ghost_Cells.fill_ghost_cells_2D_single(e, order)
        
        return e
    
    @staticmethod
    def internal_energy_from_temperature(T, C_v, order):
        """
        Calculate internal energy using the specific heat capacity at constant volume.
        
        T = temperature
        C_v = specific heat capacity at constant volume
        
        returns e = internal energy
        
        """
        ngc = Ghost_Cells.number_ghost_cells(order)
        
        # Initialize array using np.zeros
        e = np.zeros(np.shape(T))
        
        e[ngc:-ngc, ngc:-ngc] = C_v * T[ngc:-ngc, ngc:-ngc]
        
        e           =   Ghost_Cells.fill_ghost_cells_2D_single(e, order)
        
        return e