import numpy as np

def calc_permettivity_drude_model(freq_cm, eps_inf, w_p, w_t):
    """Calculate the permettivity from the Drude model. 
        All units in cm or in inverse cm.
        freq_cm - frequency
        w_p plasma frequency
        w_t  frequency"""
    permettivity = eps_inf - w_p*w_p / (freq_cm*freq_cm + 1.0j * w_t*w_t) + \
        1.0j * (w_p*w_p * w_t)/(freq_cm*(freq_cm*freq_cm+w_t*w_t))
    return permettivity

def inverse_cm_from_m(wavelength: float):
    """
    Convert a wavelength in meters to inverse centimeters.
    
    Args:
        wavelength (float): The wavelength in meters.
        
    Returns:
        float: The frequency in inverse centimeters of the given wavelength.
    """
    return 0.01 / wavelength

def m_from_inverse_cm(frequency):
    """
    Convert a frequency in inverse centimeters to meters.
    
    Args:
        frequency (float): The frequency in inverse centimeters.
        
    Returns:
        float: The wavelength in meters of the given frequency.
    """
    return 0.01 / frequency

def eps2nk(eps):
    """
    Converts complex dielectric constant `eps` to complex refractive index `n + 1j*k`.
    
    Parameters:
    -----------
    eps: complex
        Complex dielectric constant.
        
    Returns:
    --------
    n + 1j*k: complex
        Complex refractive index where `n` is the real part and `k` is the imaginary part.
    """
    n = np.sqrt((np.abs(eps)+np.real(eps))/2)
    k = np.sqrt((np.abs(eps)-np.real(eps))/2)
    return n + 1j * k
    # return SM.sqrt(eps)

def nk2eps(nk):
    n = np.real(nk)
    k = np.imag(nk)
    epsr = n*n - k*k
    epsi = 2 * n * k
    return epsr + 1j * epsi

def eps_drude(omega, w_p, gamma_p, eps_inf):
    return eps_inf -  w_p*w_p / (omega *(omega + 1.0j * gamma_p))

def eps_drude_lorentz(w_cm, w_p, gamma_p, eps_inf, A_L, w_L, gamma_L):
    return eps_inf - w_p*w_p / (w_cm * w_cm + 1.0j * gamma_p* w_cm) + \
        A_L * w_L * w_L / (w_L*w_L - w_cm*w_cm - 1.0j * gamma_L * w_cm)
        
def s_to_cm_inv(sec):
    c = 299792458  # speed of light in m/s
    return c * 1e-12 / sec