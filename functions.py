import math
import numpy as np
import json
from numba import njit

@njit
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

@njit
def input_space(T, RH, U_default, d, max_d):
    wb_temp=wet_bulb_temperature(T,RH)
    if wb_temp<=-2.5 and d<max_d: #la neve può essere prodotta
        return U_default
    else:
        return np.array([[0,0]],dtype=np.float32)


def load_transition_matrices(filename):
    """
    Carica una lista di matrici dal JSON e la ricostruisce
    come lista di numpy.ndarray.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [np.array(M, dtype=float) for M in data]

def build_joint_transition(P_T, P_RH):
    """
    P_T: array shape (nT, nT) where P_T[i_prev, i_next] = P(T_{t+1}=i_next | T_t=i_prev)
    P_RH: array shape (nRH, nRH) where P_RH[j_prev, j_next] = P(RH_{t+1}=j_next | RH_t=j_prev)
    Returns: P_joint shape (nT*nRH, nT*nRH)
    """
    # controlli
    def check_rows_stochastic(mat, name):
        row_sums = mat.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError(f"Le righe di {name} non sommano a 1 (min,max) = {row_sums.min(), row_sums.max()}")
    check_rows_stochastic(P_T, "P_T")
    check_rows_stochastic(P_RH, "P_RH")
    
    # prodotto di Kronecker (nota: np.kron(A,B) restituisce blocchi A[i,j]*B)
    P_joint = np.kron(P_T, P_RH)
    # opzionale: verifica righe
    if not np.allclose(P_joint.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("Le righe della matrice congiunta non sommano a 1")
    return P_joint

@njit
def snow_produced(T_t,RH_t,af_t,wf_t):
    alpha = 0.4 #[0.4,1]
    T_wb_soglia= -2.5 #°C
    T_wb = wet_bulb_temperature(T_t,RH_t)
    k = alpha * max(0,T_wb_soglia-T_wb)
    snow_prod = k * min(af_t,wf_t)
    return snow_prod

"""
def snow_melted(T_t,RH_t):
    DDF = 10 #degree day factor (mm/°C/giorno), tipicamente 1-10 mm
    T0 = 0 #°C sopra la quale avviene la fusione della neve, [-1,1] in genere
    A = 10 #metri quadri: superficie occupata dalla neve prodotta
    T_avg = T_t #temperatura media durante il time step
    snow_melt = (DDF/(1000*24)) * max(T_avg-T0,0) * A
    return snow_melt
"""
@njit
def snow_melted(T_t, RH_t, A=30):
    """
    Calcola la neve sciolta in un'ora (in m^3) in base a temperatura e umidità relative.
    
    Parametri:
        T_t : temperatura media dell'ora (°C)
        RH_t : umidità relativa (0-1 o 0-100)
        A : area innevata (m^2)
    """
    
    # Converti RH in 0-1 se necessario
    if RH_t > 1:
        RH = RH_t / 100
    else:
        RH = RH_t

    # Parametri fisici/semiempirici
    DHF = 0.7   # mm/°C/ora, tipico range 0.1–0.7 mm/°C/h
    T0 = 0.0    # temperatura soglia
    
    # Fattore di amplificazione dell'umidità
    humidity_factor = 0.5 + 0.5 * RH   # da 0.5 (secco) a 1.0 (umido)
    
    # Degree-hour melt (mm di acqua equivalente)
    melt_mm = DHF * max(T_t - T0, 0) * humidity_factor
    
    # Conversione mm → m³: 1 mm = 1/1000 m
    melt_volume = (melt_mm / 1000) * A
    
    return melt_volume

@njit
def state_to_index(T_val, RH_val, temp_vals, humid_vals):
    # Usa searchsorted per trovare l'indice dove il valore DOVREBBE essere
    i = np.searchsorted(temp_vals, T_val)
    j = np.searchsorted(humid_vals, RH_val)
    
    n_T = len(temp_vals)
    n_RH = len(humid_vals)

    # Controllo bounds: verifica se l'indice è valido e se il valore corrisponde davvero
    # (searchsorted restituisce l'indice di inserimento anche se il valore non c'è)
    if i >= n_T or np.abs(temp_vals[i] - T_val) > 1e-5:
        return -1 # Temperatura non trovata esattamente
    
    if j >= n_RH or np.abs(humid_vals[j] - RH_val) > 1e-5:
        return -1 # Umidità non trovata esattamente

    return i * n_RH + j

    
@njit
def index_to_state(idx, temp_vals, humid_vals):
    """
    Converte un indice della matrice di transizione congiunta nello stato (T,RH).
    
    Parameters:
        idx : int - indice nella matrice P
        temp_vals : array_like - array dei valori di temperatura per la fascia
        humid_vals : array_like - array dei valori di umidità per la fascia
    
    Returns:
        (T_val, RH_val) : tuple dei valori di temperatura e umidità
    """
    n_RH = len(humid_vals)
    i = idx // n_RH
    j = idx % n_RH
    T_val = temp_vals[i]
    RH_val = humid_vals[j]
    return T_val, RH_val


