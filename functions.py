import math
import numpy as np
def wet_bulb_temperature(T, RH):
    """
    Calcola la temperatura di bulbo umido (Tw) dati T (°C) e RH (%)
    usando la formula approssimata mostrata nell'immagine.
    """
    term1 = T * math.atan(0.151977 * math.sqrt(RH + 8.313659))
    term2 = math.atan(T + RH)
    term3 = -math.atan(RH - 1.676331)
    term4 = 0.00391838 * (RH ** 1.5) * math.atan(0.023101 * RH) -4.686035

    Tw = term1 + term2 + term3 + term4
    return Tw

def input_space(T, RH, min_af, max_af, min_wf, max_wf, af_step, wf_step):
    wb_temp=wet_bulb_temperature(T,RH)
    af_val=np.arange(min_af,max_af+af_step,af_step)
    af_n_val=af_val.shape[0]
    wf_val=np.arange(min_wf,max_wf+wf_step,wf_step)
    wf_n_val=wf_val.shape[0]
    if wb_temp<=-2.5: #la neve può essere prodotta
        U=np.empty((af_n_val,wf_n_val,2))
        for i,af in enumerate(af_val):
            for j,wf in enumerate(wf_val):
                U[i,j]=[af,wf]
    else:
        U=np.array([0,0])
    return U