import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import json

# bool variable to enable the plot of the pmfs di umidità e temperatura per fascia
print("Digita: True per abilitare i plot | False per disabilitarli.")
string=input(">>")
plot_enabled=True if string=="True" else False

# LETTURA FILEs CSV Caples Lake, 9 nodi
# CREAZIONE DI UN DATAFRAME PER OGNI NODO,
# filtrando i dati per data e selezionando solo le colonne temperatura e umidità
# SALVATAGGIO su 9 FILEs csv i dati filtrati
#-------------------------------------------------
# cartella dove si trovano i file CSV originali
path = "dataset/"

# cartella di output per i csv filtrati
output_path = "dataframes/"

os.makedirs(output_path, exist_ok=True)

# intervallo di date richiesto
start_date = "2016-01-01"
end_date   = "2016-02-29 23:59:00"

# lista con i dataframe risultanti
dfs = []

# legge tutti i csv nella cartella
for node_idx, file in enumerate(sorted(glob.glob(path + "*.csv"))):

    # leggi solo le colonne che interessano
    df = pd.read_csv(file, usecols=["P_time", "Temp_C1(DegC)", "RH(%)"])

    # converte P_time in datetime
    df["P_time"] = pd.to_datetime(df["P_time"])

    # filtra l’intervallo temporale
    df = df[(df["P_time"] >= start_date) & (df["P_time"] <= end_date)]

    # aggiungi alla lista dei dataframe
    dfs.append(df)

    # crea il nome del file CSV di output
    base_name = os.path.splitext(os.path.basename(file))[0]
    csv_file = os.path.join(output_path, f"{base_name}_filtered.csv")

    # salva il dataframe in formato CSV
    df.to_csv(csv_file, index=False)

    print(f"Salvato: {csv_file}")

    #selezione di sensori specifici da cui estrarre dati
    #if(node_idx==0): break

#--------------------------------------------------------------
# DIVISIONE DEI DATI di tutti i sensori PER FASCIA ORARIA,
# salvataggio dati temp e RH su liste di numpy arrays, uno per ogni fascia
num_fasce=24
durata_fascia=24//num_fasce
# le liste conterranno un numero di numpy arrays pari al numero delle fasce
# un numpy array per ogni fascia oraria che conterrà i relativi valori di temp e RH
temp_by_fascia = [[] for _ in range(num_fasce)]
rh_by_fascia   = [[] for _ in range(num_fasce)]

for df in dfs:
    # estrai ora dal timestamp
    hours = df["P_time"].dt.hour

    # calcola indice fascia nel range 0–(num_fasce-1)
    fasce = hours // durata_fascia

    for i in range(num_fasce):
        mask = fasce == i

        # append dei valori corrispondenti alla fascia i
        temp_by_fascia[i].extend(df.loc[mask, "Temp_C1(DegC)"].values)
        rh_by_fascia[i].extend(df.loc[mask, "RH(%)"].values)

# conversione in numpy arrays
temp_by_fascia = [np.array(x) for x in temp_by_fascia]
rh_by_fascia   = [np.array(x) for x in rh_by_fascia]

# stampa del numero di dati utilizzati per la stima dei ciascuna pmf (per fascia)
print("Per ogni fascia vengono utilizzati "+str(len(temp_by_fascia[0]))+" valori per stimare le pmf di temperatura e umidità.\n")
#print(temp_by_fascia[8])
#print(temp_by_fascia[9])
#-----------------------------------------------------------
# CALCOLO PMFs PER FASCIA ORARIA e salvataggio su un unico file json
# Supponiamo di avere già:
# temp_by_fascia = [...]  # lista di 6 numpy array
# rh_by_fascia   = [...]  # lista di 6 numpy array

def compute_pmf(arr, bins=50):
    """
    Calcola la PMF di un array utilizzando istogramma normalizzato
    """
    #print("Max:",np.max(arr))
    #print("Min:",np.min(arr))
    counts, bin_edges = np.histogram(arr, bins=bins, density=True)
    # valori al centro dei bin
    values = (bin_edges[:-1] + bin_edges[1:]) / 2
    pmf = counts / counts.sum()  # normalizza a 1
    #print(values)
    #print(pmf)
    return values, pmf

# Dizionari per salvare le PMF
pmf_temp_dict = {}
pmf_rh_dict = {}
temp_vals_by_fascia = []
rh_vals_by_fascia = []

# Ciclo sulle  fasce orarie
for i in range(num_fasce):
    #if(i%6==0):
    fascia_label = f"{(i*durata_fascia):02d}:00-{(i*durata_fascia+durata_fascia-1):02d}:59"
    # Temperatura
    temp_vals, temp_pmf = compute_pmf(temp_by_fascia[i])
    if plot_enabled:
        plt.figure(figsize=(8,4))
        plt.stem(temp_vals, temp_pmf)
        plt.xlabel("Temperatura (°C)")
        plt.ylabel("PMF")
        plt.title(f"PMF Temperatura - Fascia {i} (ore {i*durata_fascia:02d}:00-{i*durata_fascia+durata_fascia-1:02d}:59)")
        plt.grid(True)
        plt.show()

    # Umidità
    rh_vals, rh_pmf = compute_pmf(rh_by_fascia[i])
    if plot_enabled:
        plt.figure(figsize=(8,4))
        plt.stem(rh_vals, rh_pmf)
        plt.xlabel("Umidità (%)")
        plt.ylabel("PMF")
        plt.title(f"PMF Umidità - Fascia {i} (ore {i*durata_fascia:02d}:00-{i*durata_fascia+durata_fascia-1:02d}:59)")
        plt.grid(True)
        plt.show()


    # creazione dizionario delle pmf della temperatura per fasce
    pmf_temp_dict[fascia_label] = {
        "values": temp_vals.tolist(),
        "pmf": temp_pmf.tolist()
    }

    # creazione dizionario delle pmf dell'umidità per fasce
    pmf_rh_dict[fascia_label] = {
        "values": rh_vals.tolist(),
        "pmf": rh_pmf.tolist()
    }
    temp_vals_by_fascia.append(temp_vals)
    rh_vals_by_fascia.append(rh_vals)

# Salvataggio in file JSON
with open("pmf_temperature.json", "w") as f_temp:
    json.dump(pmf_temp_dict, f_temp, indent=4)

with open("pmf_humidity.json", "w") as f_rh:
    json.dump(pmf_rh_dict, f_rh, indent=4)

print("PMF salvate in pmf_temperature.json e pmf_humidity.json")


#------------------------------------------------------------------------------
#CALCOLO MATRICI DELLE PROBABILITA' CONDIZIONATE PER TEMPERATURA E UMIDITA'

def compute_bins(values, num_bins=50):
    """
    Calcola i bin (edges e centrali) per la discretizzazione.
    """
    counts, edges = np.histogram(values, bins=num_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers

def digitize_values(values, edges):
    """
    Assegna ogni valore al bin corrispondente (indice di stato).
    """
    idx = np.digitize(values, edges) - 1
    idx[idx < 0] = 0
    idx[idx >= len(edges)-1] = len(edges)-2
    return idx

def compute_transition_matrix(idx_t, idx_tp1, n_states):
    """
    Stima la matrice delle probabilità condizionate P(i->j)
    da due array di indici corrispondenti.
    """
    counts = np.zeros((n_states, n_states))

    for a, b in zip(idx_t, idx_tp1):
        counts[a, b] += 1

    row_sums = counts.sum(axis=1)
    P = np.zeros_like(counts)

    for i in range(n_states):
        if row_sums[i] > 0:
            P[i, :] = counts[i, :] / row_sums[i]
        else:
            # nessuna osservazione da quello stato → distribuzione uniforme
            P[i, :] = np.ones(n_states) / n_states

    return P

# ---------------------------------------------------------
# COSTRUZIONE DELLE MATRICI DI TRANSIZIONE PER T E RH
# ---------------------------------------------------------
num_bins = 50
P_T_list = []   # conterrà 23 matrici 50x50
P_RH_list = []  # conterrà 23 matrici 50x50

for f in range(num_fasce):

    # --- TEMPERATURA ---

    # unisci i dati di fascia f e f+1 per costruire bin coerenti
    """
    T_concat = np.concatenate([temp_by_fascia[f], temp_by_fascia[f+1]])
    T_edges, T_states = compute_bins(T_concat, num_bins)
    """

    T_edges_f_curr = temp_vals_by_fascia[f] #valori discreti temperatura fascia corrente
    if f<23:
        T_edges_f_next = temp_vals_by_fascia[f+1] #valori discreti temperatura fascia successiva
    else: #siamo nell'ultima fascia oraria, i prossimi valori da considerare sono quelli della fascia 0
        T_edges_f_next = temp_vals_by_fascia[0] #valori discreti temperatura fascia successiva

    idx_T_t   = digitize_values(temp_by_fascia[f],   T_edges_f_curr)
    if f<23:
        idx_T_tp1 = digitize_values(temp_by_fascia[f+1], T_edges_f_next)
    else: #siamo nell'ultima fascia oraria, i prossimi valori da considerare sono quelli della fascia 0
        idx_T_tp1 = digitize_values(temp_by_fascia[0], T_edges_f_next)

    P_T = compute_transition_matrix(idx_T_t, idx_T_tp1, num_bins)
    P_T_list.append(P_T)

    # --- UMIDITÀ ---
    """
    RH_concat = np.concatenate([rh_by_fascia[f], rh_by_fascia[f+1]])
    RH_edges, RH_states = compute_bins(RH_concat, num_bins)
    """

    RH_edges_f_curr = rh_vals_by_fascia[f] #valori discreti temperatura fascia corrente
    if f<23:
        RH_edges_f_next = rh_vals_by_fascia[f+1] #valori discreti temperatura fascia successiva
    else: #siamo nell'ultima fascia oraria, i prossimi valori da considerare sono quelli della fascia 0
        RH_edges_f_next = rh_vals_by_fascia[0]

    idx_RH_t   = digitize_values(rh_by_fascia[f],   RH_edges_f_curr)
    if f<23:
        idx_RH_tp1 = digitize_values(rh_by_fascia[f+1], RH_edges_f_next)
    else: #siamo nell'ultima fascia oraria, i prossimi valori da considerare sono quelli della fascia 0
        idx_RH_tp1 = digitize_values(rh_by_fascia[0], RH_edges_f_next)

    P_RH = compute_transition_matrix(idx_RH_t, idx_RH_tp1, num_bins)
    P_RH_list.append(P_RH)

print("Calcolate", len(P_T_list), "matrici di transizione per la Temperatura (50x50)")
print("Calcolate", len(P_RH_list), "matrici di transizione per l'Umidità (50x50)")

#np.set_printoptions(threshold=np.inf)

print(P_T_list[23])

# -----------------------------------------------------
# SALVATAGGIO DELLE MATRICI DI TRANSIZIONE IN JSON
# -----------------------------------------------------

def save_transition_matrices(filename, matrix_list):
    """
    Salva una lista di matrici numpy in un file JSON.
    Ogni matrice viene convertita in una lista annidata.
    """
    serializable = [M.tolist() for M in matrix_list]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4)

# Salvataggio delle due liste
save_transition_matrices("transition_T.json", P_T_list)
save_transition_matrices("transition_RH.json", P_RH_list)

print("Matrici salvate in transition_T.json e transition_RH.json")
