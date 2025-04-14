import numpy as np
from GhostCells import Ghost_Cells

class FiniteDifference_2D:
    # Has third dimension
    @staticmethod
    def df_dx_2D(f, dx, order):
        """
        Compute the first derivative of f with respect to x
        
        Includes, 2nd, 4th, 6th order central scheme in this function
        
        Returns:
        df : array w/out ghost cells
            Derivative of f with respect to x, fluid domain plus ghost cells
        """
                       
        # determine number of ghost cells
        ngc = int(Ghost_Cells.number_ghost_cells(order))
        
        # Initialize array using np.zeros
        df_dx = np.zeros(np.shape(f))
        
        # Central difference 2nd order
        if order == 2:
            df_dx[ngc:-ngc, ngc:-ngc, :] = (
                    + f[ngc+1:      , ngc:-ngc, :] 
                    - f[ngc-1:-ngc-1, ngc:-ngc, :]
                    ) / (2*dx)
        elif order == 4:
            df_dx[ngc:-ngc, ngc:-ngc, :] = (
                    (
                    -   f[ngc+2:      , ngc:-ngc, :] 
                    + 8*f[ngc+1:-ngc+1, ngc:-ngc, :] 
                    - 8*f[ngc-1:-ngc-1, ngc:-ngc, :] 
                    +   f[ngc-2:-ngc-2, ngc:-ngc, :] 
                    )
                    / (12*dx)
            )
        elif order == 6:
            df_dx[ngc:-ngc, ngc:-ngc, :] = (
                    (
                    +    f[ngc+3:      , ngc:-ngc, :] 
                    - 9 *f[ngc+2:-ngc+2, ngc:-ngc, :] 
                    + 45*f[ngc+1:-ngc+1, ngc:-ngc, :] 
                    - 45*f[ngc-1:-ngc-1, ngc:-ngc, :] 
                    + 9 *f[ngc-2:-ngc-2, ngc:-ngc, :] 
                    -    f[ngc-3:-ngc-3, ngc:-ngc, :] 
                    )
                    / (60*dx)
            )
            
        # Fill ghost cells here
        df_dx = Ghost_Cells.fill_ghost_cells_2D(df_dx, order)
                
        return df_dx
        
    @staticmethod
    def df_dy_2D(f, dy, order):
        """
        Compute the first derivative of f with respect to y
        
        Inlcudes, 2nd, 4th, 6th order central scheme in this function
        
        Returns:
        df : array w/out ghost cells
            Derivative of f with respect to y, fluid domain plus ghost cells
        """
                       
        # determine number of ghost cells
        ngc = int(Ghost_Cells.number_ghost_cells(order))
        
        # Initalize array
        df_dy = np.zeros(np.shape(f))
        
        # Central difference 2nd order
        if order == 2:
            df_dy[ngc:-ngc, ngc:-ngc, :] = (
                (
                    f[ngc:-ngc, ngc+1:      , :] 
                +   f[ngc:-ngc, ngc-1:-ngc-1, :]
                ) 
                / (2*dy)            
            )
        elif order == 4:
            df_dy[ngc:-ngc, ngc:-ngc, :] = (
                (
                -   f[ngc:-ngc, ngc+2:      , :] 
                + 8*f[ngc:-ngc, ngc+1:-ngc+1, :] 
                - 8*f[ngc:-ngc, ngc-1:-ngc-1, :] 
                +   f[ngc:-ngc, ngc-2:-ngc-2, :] 
                )
                / (12*dy)
            )
        elif order == 6:
            df_dy[ngc:-ngc, ngc:-ngc, :] = (
                (
                +    f[ngc:-ngc, ngc+3:      , :] 
                - 9 *f[ngc:-ngc, ngc+2:-ngc+2, :] 
                + 45*f[ngc:-ngc, ngc+1:-ngc+1, :] 
                - 45*f[ngc:-ngc, ngc-1:-ngc-1, :] 
                + 9 *f[ngc:-ngc, ngc-2:-ngc-2, :] 
                -    f[ngc:-ngc, ngc-3:-ngc-3, :] 
                )
                / (60*dy)
            )
            
        # Fill gc here
        df_dy = Ghost_Cells.fill_ghost_cells_2D(df_dy, order)    
        return df_dy

    @staticmethod
    def d2f_dx2_2D(f, dx, order):
        """
        Compute the second derivative of f with respect to x    
        
        Inlcudes, 2nd, 4th, 6th order central scheme in this function  
        
        Returns:
        d2f_dx2 : array with ghost cells     
        
        """

        #determine number of ghost cells
        ngc = int(Ghost_Cells.number_ghost_cells(order))
        
        # Initalize array
        d2f_dx2   = np.zeros(np.shape(f))
        
        # Central difference 2nd order
        if order == 2:
            d2f_dx2[ngc:-ngc, ngc:-ngc, :] = (
                (
                    +   f[ngc+1:      , ngc:-ngc, :] 
                    - 2*f[ngc  :-ngc  , ngc:-ngc, :] 
                    +   f[ngc-1:-ngc-1, ngc:-ngc, :]
                ) 
                / (dx**2)
            )
        elif order == 4:
            d2f_dx2[ngc:-ngc, ngc:-ngc, :] = (
                (
                    -    f[ngc+2:      , ngc:-ngc, :] 
                    + 16*f[ngc+1:-ngc+1, ngc:-ngc, :] 
                    - 30*f[ngc  :-ngc  , ngc:-ngc, :] 
                    + 16*f[ngc-1:-ngc-1, ngc:-ngc, :] 
                    -    f[ngc-2:-ngc-2, ngc:-ngc, :]
                ) 
                / (12*dx**2)
            )
        elif order == 6:
            d2f_dx2[ngc:-ngc, ngc:-ngc, :] = (
                (
                    + (1/90) * f[ngc+3:      , ngc:-ngc, :] 
                    - (3/20) * f[ngc+2:-ngc+2, ngc:-ngc, :] 
                    + (3/2)  * f[ngc+1:-ngc+1, ngc:-ngc, :] 
                    - (49/18)* f[ngc  :-ngc  , ngc:-ngc, :] 
                    + (3/2)  * f[ngc-1:-ngc-1, ngc:-ngc, :] 
                    - (3/20) * f[ngc-2:-ngc-2, ngc:-ngc, :] 
                    + (1/90) * f[ngc-3:-ngc-3, ngc:-ngc, :]
                ) 
                / (dx**2)
            )
            
        # Fill gc here
        d2f_dx2 = Ghost_Cells.fill_ghost_cells_2D(d2f_dx2, order)
            
        return d2f_dx2
    
    @staticmethod
    def d2f_dy2_2D(f, dy, order):
        """
        Compute the second derivative of f with respect to y
        
        Inlcudes, 2nd, 4th, 6th order central scheme in this function
        
        Returns:
        d2f_dy2 : array with ghost cells
        """
        
        #determine number of ghost cells
        ngc = int(Ghost_Cells.number_ghost_cells(order))

        # Initalize array
        d2f_dy2   = np.zeros(np.shape(f))
        
        # Central difference 2nd order
        if order == 2:
            d2f_dy2[ngc:-ngc, ngc:-ngc, :] = (
                (
                        f[ngc:-ngc, ngc+1:      , :] 
                    - 2*f[ngc:-ngc, ngc  :-ngc  , :] 
                    +   f[ngc:-ngc, ngc-1:-ngc-1, :]
                ) 
                / (dy**2)
            )
        elif order == 4:
            d2f_dy2[ngc:-ngc, ngc:-ngc, :] = (
                (
                    -    f[ngc:-ngc, ngc+2:      , :] 
                    + 16*f[ngc:-ngc, ngc+1:-ngc+1, :] 
                    - 30*f[ngc:-ngc, ngc  :-ngc  , :] 
                    + 16*f[ngc:-ngc, ngc-1:-ngc-1, :] 
                    -    f[ngc:-ngc, ngc-2:-ngc-2, :]
                ) 
                / (12*dy**2)
            )
        elif order == 6:
            d2f_dy2[ngc:-ngc, ngc:-ngc, :] = (
                (
                    + (1/90) * f[ngc:-ngc, ngc+3:      , :] 
                    - (3/20) * f[ngc:-ngc, ngc+2:-ngc+2, :] 
                    + (3/2)  * f[ngc:-ngc, ngc+1:-ngc+1, :] 
                    - (49/18)* f[ngc:-ngc, ngc  :-ngc  , :] 
                    + (3/2)  * f[ngc:-ngc, ngc-1:-ngc-1, :] 
                    - (3/20) * f[ngc:-ngc, ngc-2:-ngc-2, :] 
                    + (1/90) * f[ngc:-ngc, ngc-3:-ngc-3, :]
                ) 
                / (dy**2)
            )
            
        # Fill gc here
        d2f_dy2 = Ghost_Cells.fill_ghost_cells_2D(d2f_dy2, order)
        
        return d2f_dy2

    
class FiniteDifference_2D_single:
    # No third dimension 
    @staticmethod
    def df_dx_2D(f, dx, order):
        """
        Compute the first derivative of f with respect to x
        
        Includes, 2nd, 4th, 6th order central scheme in this function
        
        Returns:
        df : array w/out ghost cells
            Derivative of f with respect to x, fluid domain plus ghost cells
        """
                       
        # determine number of ghost cells
        ngc = int(Ghost_Cells.number_ghost_cells(order))
        
        # Initialize array using np.zeros
        df_dx = np.zeros(np.shape(f))
        
        # Central difference 2nd order
        if order == 2:
            df_dx[ngc:-ngc, ngc:-ngc] = (
                    + f[ngc+1:      , ngc:-ngc] 
                    - f[ngc-1:-ngc-1, ngc:-ngc]
                    ) / (2*dx)
        elif order == 4:
            df_dx[ngc:-ngc, ngc:-ngc] = (
                    (
                    -   f[ngc+2:      , ngc:-ngc] 
                    + 8*f[ngc+1:-ngc+1, ngc:-ngc] 
                    - 8*f[ngc-1:-ngc-1, ngc:-ngc] 
                    +   f[ngc-2:-ngc-2, ngc:-ngc] 
                    )
                    / (12*dx)
            )
        elif order == 6:
            df_dx[ngc:-ngc, ngc:-ngc] = (
                    (
                    +    f[ngc+3:      , ngc:-ngc] 
                    - 9 *f[ngc+2:-ngc+2, ngc:-ngc] 
                    + 45*f[ngc+1:-ngc+1, ngc:-ngc] 
                    - 45*f[ngc-1:-ngc-1, ngc:-ngc] 
                    + 9 *f[ngc-2:-ngc-2, ngc:-ngc] 
                    -    f[ngc-3:-ngc-3, ngc:-ngc] 
                    )
                    / (60*dx)
            )
            
        # Fill ghost cells here
        df_dx = Ghost_Cells.fill_ghost_cells_2D_single(df_dx, order)
                
        return df_dx
        
    @staticmethod
    def df_dy_2D(f, dy, order):
        """
        Compute the first derivative of f with respect to y
        
        Inlcudes, 2nd, 4th, 6th order central scheme in this function
        
        Returns:
        df : array w/out ghost cells
            Derivative of f with respect to y, fluid domain plus ghost cells
        """
                       
        # determine number of ghost cells
        ngc = int(Ghost_Cells.number_ghost_cells(order))
        
        # Initalize array
        df_dy = np.zeros(np.shape(f))
        
        # Central difference 2nd order
        if order == 2:
            df_dy[ngc:-ngc, ngc:-ngc] = (
                (
                    f[ngc:-ngc, ngc+1:      ] 
                +   f[ngc:-ngc, ngc-1:-ngc-1]
                ) 
                / (2*dy)            
            )
        elif order == 4:
            df_dy[ngc:-ngc, ngc:-ngc] = (
                (
                -   f[ngc:-ngc, ngc+2:      ] 
                + 8*f[ngc:-ngc, ngc+1:-ngc+1] 
                - 8*f[ngc:-ngc, ngc-1:-ngc-1] 
                +   f[ngc:-ngc, ngc-2:-ngc-2] 
                )
                / (12*dy)
            )
        elif order == 6:
            df_dy[ngc:-ngc, ngc:-ngc] = (
                (
                +    f[ngc:-ngc, ngc+3:      ] 
                - 9 *f[ngc:-ngc, ngc+2:-ngc+2] 
                + 45*f[ngc:-ngc, ngc+1:-ngc+1] 
                - 45*f[ngc:-ngc, ngc-1:-ngc-1] 
                + 9 *f[ngc:-ngc, ngc-2:-ngc-2] 
                -    f[ngc:-ngc, ngc-3:-ngc-3] 
                )
                / (60*dy)
            )
            
        # Fill gc here
        df_dy = Ghost_Cells.fill_ghost_cells_2D_single(df_dy, order)    
        return df_dy

    @staticmethod
    def d2f_dx2_2D(f, dx, order):
        """
        Compute the second derivative of f with respect to x    
        
        Inlcudes, 2nd, 4th, 6th order central scheme in this function  
        
        Returns:
        d2f_dx2 : array with ghost cells     
        
        """

        #determine number of ghost cells
        ngc = int(Ghost_Cells.number_ghost_cells(order))
        
        # Initalize array
        d2f_dx2   = np.zeros(np.shape(f))
        
        # Central difference 2nd order
        if order == 2:
            d2f_dx2[ngc:-ngc, ngc:-ngc] = (
                (
                    +   f[ngc+1:      , ngc:-ngc] 
                    - 2*f[ngc  :-ngc  , ngc:-ngc] 
                    +   f[ngc-1:-ngc-1, ngc:-ngc]
                ) 
                / (dx**2)
            )
        elif order == 4:
            d2f_dx2[ngc:-ngc, ngc:-ngc] = (
                (
                    -    f[ngc+2:      , ngc:-ngc] 
                    + 16*f[ngc+1:-ngc+1, ngc:-ngc] 
                    - 30*f[ngc  :-ngc  , ngc:-ngc] 
                    + 16*f[ngc-1:-ngc-1, ngc:-ngc] 
                    -    f[ngc-2:-ngc-2, ngc:-ngc]
                ) 
                / (12*dx**2)
            )
        elif order == 6:
            d2f_dx2[ngc:-ngc, ngc:-ngc] = (
                (
                    + (1/90) * f[ngc+3:      , ngc:-ngc] 
                    - (3/20) * f[ngc+2:-ngc+2, ngc:-ngc] 
                    + (3/2)  * f[ngc+1:-ngc+1, ngc:-ngc] 
                    - (49/18)* f[ngc  :-ngc  , ngc:-ngc] 
                    + (3/2)  * f[ngc-1:-ngc-1, ngc:-ngc] 
                    - (3/20) * f[ngc-2:-ngc-2, ngc:-ngc] 
                    + (1/90) * f[ngc-3:-ngc-3, ngc:-ngc]
                ) 
                / (dx**2)
            )
            
        # Fill gc here
        d2f_dx2 = Ghost_Cells.fill_ghost_cells_2D_single(d2f_dx2, order)
            
        return d2f_dx2
    
    @staticmethod
    def d2f_dy2_2D(f, dy, order):
        """
        Compute the second derivative of f with respect to y
        
        Inlcudes, 2nd, 4th, 6th order central scheme in this function
        
        Returns:
        d2f_dy2 : array with ghost cells
        """
        
        #determine number of ghost cells
        ngc = int(Ghost_Cells.number_ghost_cells(order))

        # Initalize array
        d2f_dy2   = np.zeros(np.shape(f))
        
        # Central difference 2nd order
        if order == 2:
            d2f_dy2[ngc:-ngc, ngc:-ngc] = (
                (
                        f[ngc:-ngc, ngc+1:      ] 
                    - 2*f[ngc:-ngc, ngc  :-ngc  ] 
                    +   f[ngc:-ngc, ngc-1:-ngc-1]
                ) 
                / (dy**2)
            )
        elif order == 4:
            d2f_dy2[ngc:-ngc, ngc:-ngc] = (
                (
                    -    f[ngc:-ngc, ngc+2:      ] 
                    + 16*f[ngc:-ngc, ngc+1:-ngc+1] 
                    - 30*f[ngc:-ngc, ngc  :-ngc  ] 
                    + 16*f[ngc:-ngc, ngc-1:-ngc-1] 
                    -    f[ngc:-ngc, ngc-2:-ngc-2]
                ) 
                / (12*dy**2)
            )
        elif order == 6:
            d2f_dy2[ngc:-ngc, ngc:-ngc] = (
                (
                    + (1/90) * f[ngc:-ngc, ngc+3:      ] 
                    - (3/20) * f[ngc:-ngc, ngc+2:-ngc+2] 
                    + (3/2)  * f[ngc:-ngc, ngc+1:-ngc+1] 
                    - (49/18)* f[ngc:-ngc, ngc  :-ngc  ] 
                    + (3/2)  * f[ngc:-ngc, ngc-1:-ngc-1] 
                    - (3/20) * f[ngc:-ngc, ngc-2:-ngc-2] 
                    + (1/90) * f[ngc:-ngc, ngc-3:-ngc-3]
                ) 
                / (dy**2)
            )
            
        # Fill gc here
        d2f_dy2 = Ghost_Cells.fill_ghost_cells_2D_single(d2f_dy2, order)
        
        return d2f_dy2
    
    
    
    

class FiniteDifference:

    @staticmethod
    def first_derivative(f, dx, order):
        """
        Compute the first derivative of f with respect to x
        
        Inlcudes, 2nd, 4th, 6th order central scheme in this function
        
        Returns:
        df : array w/out ghost cells
            Derivative of f with respect to x, fluid domain plus ghost cells
        """
                       
        # determine number of ghost cells
        ngc = int(Ghost_Cells.number_ghost_cells(order))
        
        n = len(f)  # length of (x + 2*ngc), includes ghost cells
        
        # df is output array with ghost cells
        df = np.zeros(n)
        
        if order == 2:              
        # Central difference 2nd order
            df[ngc:-ngc] = (
            (
                + f[ngc * 2:] 
                - f[: - ngc *2]
            ) 
            / (2*dx)
            )
        elif order == 4:
        # Central difference (4th-order)
            df[ngc:-ngc] = (
            (
                - f[ngc * 2 :] 
                + 8*f[ngc + 1 : n - ngc + 1] 
                - 8*f[ngc - 1 : n - ngc - 1]
                + f[: - ngc * 2]
            ) 
            / (12*dx)
            )
        elif order == 6:
        # Central differnce 6th order
            df[ngc:-ngc] = (
            (
                + f[ngc * 2:]
                - 9*f[ngc * 2 - 1:-ngc + 2]
                + 45*f[ngc * 2 - 2: -ngc + 1]
                - 45*f[ngc - 1: -ngc - 1]
                + 9*f[ngc - 2: - ngc - 2]
                - f[: - ngc - 3]
            ) / (60*dx)
            )
            
        df = Ghost_Cells.fill_ghost_cells(df, order)
        return df
       

    "Finite difference method for 2nd derivative"

    @staticmethod
    def second_derivative(f, dx, order):
        """
        Compute the second derivative of f with respect to x 
        Inlcudes, 2nd, 4th, 6th order central scheme in this function
        Non conservative 2nd derivative
        
        Parameters:
        f : array w/ ghost cells
            Evenly spaced outputs.
            The Function you want to derive
            
        dx : 
            Spacing between data inputs (delta x).
        
        Returns:
        d2f : numpy array fluid domain with ghost cells, 
            Second derivative of f with respect to x.
        """
        # determine number of ghost cells
        ngc = int(Ghost_Cells.number_ghost_cells(order))
        
        n = len(f) # length of (x + 2*ngc)
        
        # d2f is output array with ghost cells
        d2f = np.zeros(n)
                        
        if order == 4:
            # Central difference (4th-order)
            d2f[ngc:-ngc] = (
                (
                    -f[ngc + 2 : n - ngc + 2] 
                    + 16*f[ngc + 1 : n - ngc + 1] 
                    - 30*f[ngc : n - ngc] 
                    + 16*f[ngc - 1 : n - ngc - 1] 
                    - f[ngc - 2 : n - ngc - 2]
                ) 
                / (12*dx**2)
            )
        elif order ==2:
            # Central difference 2nd order
            d2f[ngc:-ngc] = (
                (
                    + f[ngc*2:] 
                    - 2*f[ngc:n-ngc] 
                    + f[:-ngc*2]
                ) 
                / (dx**2)
            )    
        elif order == 6:
            # Central difference 6th order
            d2f[ngc:-ngc] = (                
                    (
                    + (1/90) * f[ngc * 2:]
                    - (3/20) * f[ngc * 2 - 1:-ngc + 2]
                    + (3/2)  * f[ngc * 2 - 2: -ngc + 1]
                    - (49/18)* f[ngc:n-ngc]
                    + (3/2)  * f[ngc - 1: -ngc - 1]
                    - (3/20) * f[ngc - 2: - ngc - 2]
                    + (1/90) * f[: - ngc - 3]
                    ) / (dx**2)                          
            )
            
        d2f = Ghost_Cells.fill_ghost_cells(d2f, order)
        return d2f

    
    