import numpy as np
import pylab as plt

class PSF_stack(object):
    def __init__(self):
        self.psfs = np.array((n_psfs, psf_x_size, psf_y_size), dtype=float)
        self.pos = np.array((n_psfs, 2), dtype=float)

        self.pixel_scale = 1.0
        self.wavelength = 1.0
        self.bandpass = 1.0
        self.telescope = 'Keck1'
        
        return

    def save(self):
        return


class MAOS_PSF_stack(PSF_stack):
    def __init__(self):
        super().__init__(self)

        # Ohter MAOS specific stuff.

    
    
class AIROPA_PSF_stack(PSF_stack):
    def __init__(self):
        super().__init__(self)

        # Ohter MAOS specific stuff.


class OOMAO_PSF_stack(PSF_stack):
    def __init__(self):
        super().__init__(self)

        # Ohter MAOS specific stuff.

    
    
