import numpy as np
from matplotlib import pyplot 

class Ghost_Cells:
    """
    Returns array with ghost cells
    
    Initialize with function data and step size.
    
    f : array
        Evenly spaced data of outputs
        Function f
        Ghost cells not filled yet
                    
    order : 
        Highest order 
        Determines how many points to add
        2nd order = 1 point (on each end)
        4th order = 2 points (on each end) etc...
    """

    @staticmethod    
    def number_ghost_cells(order):
        """
        Checks the order and returns the ngc needed
        This is per each side hence the division by 2
        """
        if order == 6:
            return 6 // 2
        elif order == 4:
            return 4 // 2
        elif order == 2:
            return 2 // 2
        else:
            return 0
                  
    @staticmethod
    def fill_ghost_cells_2D(f, order):
        """
        fills array with ghost cells 2D
        """    

        ngc = int(Ghost_Cells.number_ghost_cells(order))
        
        
        f[    : ngc, ngc:-ngc, :] = f[-2*ngc: -ngc,   ngc: -ngc, :] # Left side
        f[-ngc:    , ngc:-ngc, :] = f[   ngc:2*ngc,   ngc: -ngc, :] # Right side
        f[ ngc:-ngc,    : ngc, :] = f[   ngc: -ngc,-2*ngc: -ngc, :] # Top side
        f[ ngc:-ngc,-ngc:    , :] = f[   ngc: -ngc,   ngc:2*ngc, :] # Bottom side
        
        #Corners for cross derivatives
        f[-ngc:   ,-ngc:   ,:] = f[   ngc:2*ngc,   ngc:2*ngc,:]  # Top right gc
        f[    :ngc,-ngc:   ,:] = f[-2*ngc:-ngc ,   ngc:2*ngc,:]  # Top left gc
        f[-ngc:   ,    :ngc,:] = f[   ngc:2*ngc,-2*ngc: -ngc,:]  # Bot right gc
        f[    :ngc,    :ngc,:] = f[-2*ngc:-ngc ,-2*ngc: -ngc,:]  # Bot left gc
        
        return f
    
    @staticmethod
    def fill_ghost_cells_2D_single(f, order):
        """
        fills array with ghost cells 2D
        """    

        ngc = int(Ghost_Cells.number_ghost_cells(order))
        
        
        f[    : ngc, ngc:-ngc] = f[-2*ngc: -ngc,   ngc: -ngc] # Left side
        f[-ngc:    , ngc:-ngc] = f[   ngc:2*ngc,   ngc: -ngc] # Right side
        f[ ngc:-ngc,    : ngc] = f[   ngc: -ngc,-2*ngc: -ngc] # Top side
        f[ ngc:-ngc,-ngc:    ] = f[   ngc: -ngc,   ngc:2*ngc] # Bottom side
        
        #Corners for cross derivatives
        f[-ngc:   ,-ngc:   ] = f[   ngc:2*ngc,   ngc:2*ngc]  # Top right gc
        f[    :ngc,-ngc:   ] = f[-2*ngc:-ngc ,   ngc:2*ngc]  # Top left gc
        f[-ngc:   ,    :ngc] = f[   ngc:2*ngc,-2*ngc: -ngc]  # Bot right gc
        f[    :ngc,    :ngc] = f[-2*ngc:-ngc ,-2*ngc: -ngc]  # Bot left gc
        
        return f

    @staticmethod
    def fill_ghost_cells(f, order):
        """
            fills array with ghost cells

        Args:
            f (array): function with unfilled gc
            order (int): order of derivative

        Returns:
            f: function with filled gc
        """
    
        ngc = int(Ghost_Cells.number_ghost_cells(order))
                
        # Since function is periodc, we "know" the values of gc
        f[:ngc] = f[-ngc*2:-ngc]
        f[-ngc:] = f[ngc:2*ngc]
  
        return f
    
    
