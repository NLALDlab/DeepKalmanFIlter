import numpy as np

def ConstructSGMatrices(WinLen):
    """
    Construct matrices used during SG filtering.
    """
    HalfWinLen = (WinLen - 1) // 2
    Int = np.arange(-HalfWinLen, HalfWinLen + 1)

    # Degree = 3; Fixed for now
    StencilA0 = np.flip((3 / (4 * WinLen * (WinLen**2 - 4))) * (3 * WinLen**2 - 7 - 20 * Int**2))
    StencilA1 = np.flip((1 / (WinLen * (WinLen**2 - 1) * (3 * WinLen**4 - 39 * WinLen**2 + 108))) * 
                        (75 * (3 * WinLen**4 - 18 * WinLen**2 + 31) * Int - 
                         420 * (3 * WinLen**2 - 7) * Int**3))
    
    return StencilA0, StencilA1
