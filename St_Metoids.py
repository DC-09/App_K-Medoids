import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import seaborn as sns # Non usato
import time

# --- Funzione Generazione Dati (Strutturata Demografica/Comportamentale) ---
# INVARIATA (presa dal tuo prompt)
def generate_structured_marketing_data(n_samples, n_centers_demo=2, n_centers_behav=2, cluster_std=0.8, random_state=42):
    """
    Genera dati numerici con struttura latente: 2 feature demo + 2 feature comportamentali.
    Feature finale rinominata in FrequenzaAq.
    """
    rng = np.random.RandomState(random_state)
    n_features_per_group = 2
    X_demo, y_demo = make_blobs(n_samples=n_samples, n_features=n_features_per_group, centers=np.array([[-1]*n_features_per_group, [1]*n_features_per_group])[:n_centers_demo], cluster_std=cluster_std, random_state=random_state)
    X_behav, y_behav = make_blobs(n_samples=n_samples, n_features=n_features_per_group, centers=np.array([[-1]*n_features_per_group, [1]*n_features_per_group])[:n_centers_behav], cluster_std=cluster_std, random_state=random_state + 1) # Offset seed
    X_combined = np.hstack((X_demo, X_behav))
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    X_shuffled = X_combined[indices]
    scaler_final = MinMaxScaler()
    age_values = (scaler_final.fit_transform(X_shuffled[:, [0]]) * (75 - 18) + 18).round(0).astype(int).flatten()
    income_values = (scaler_final.fit_transform(X_shuffled[:, [1]]) * (100000 - 15000) + 15000).round(2).flatten()
    spending_values = (scaler_final.fit_transform(X_shuffled[:, [2]]) * (500 - 20) + 20).round(2).flatten()
    freq_values = (scaler_final.fit_transform(X_shuffled[:, [3]]) * (50 - 1) + 1).round(0).astype(int).flatten()
    df = pd.DataFrame({
        'Età': age_values,
        'Reddito': income_values,
        'SpesaMedia': spending_values,
        'FrequenzaAq': freq_values
    })
    return df

# --- Funzione per Plot Silhouette ---
# INVARIATA (presa dal tuo prompt)
def plot_silhouette(X_processed, labels, n_clusters, ax):
    """Genera il plot della silhouette SENZA la linea della media."""
    silhouette_avg = silhouette_score(X_processed, labels)
    sample_silhouette_values = silhouette_samples(X_processed, labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = cm.get_cmap("viridis")
        color = cmap(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax.set_title("Silhouette Plot per i Vari Cluster")
    ax.set_xlabel("Coefficiente di Silhouette")
    ax.set_ylabel("Cluster label")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    return silhouette_avg

# --- Funzione K-Medoids (PAM) Manuale (Invariata) ---
# INVARIATA (presa dal tuo prompt)
def kmedoids_manual_pam(X, n_clusters, max_iter, random_state):
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_state)
    if n_clusters > n_samples: raise ValueError(f"k ({n_clusters}) > n_samples ({n_samples})")
    medoid_indices = rng.choice(n_samples, n_clusters, replace=False)
    medoid_indices_history = [medoid_indices.copy()]
    labels_history = []
    cost_history = []
    labels = np.zeros(n_samples, dtype=int)
    cluster_idx_to_medoid_idx = {i: medoid_indices[i] for i in range(n_clusters)}
    for iter_num in range(max_iter):
        current_medoids_coords = X[medoid_indices]
        distances = cdist(X, current_medoids_coords, metric='euclidean')
        new_labels = np.argmin(distances, axis=1)
        labels_history.append(new_labels.copy())
        current_cost = np.sum(np.min(distances, axis=1))
        cost_history.append(current_cost)
        labels = new_labels
        new_medoid_indices = medoid_indices.copy()
        swapped = False
        for k in range(n_clusters):
            current_medoid_for_k_idx = medoid_indices[k]
            cluster_points_indices = np.where(labels == k)[0]
            if len(cluster_points_indices) == 0: continue
            current_cluster_cost = np.sum(np.linalg.norm(X[cluster_points_indices] - X[current_medoid_for_k_idx], axis=1))
            best_swap_cost = current_cluster_cost
            best_swap_candidate_idx = current_medoid_for_k_idx
            for p_idx in cluster_points_indices:
                if p_idx == current_medoid_for_k_idx: continue
                potential_new_cluster_cost = np.sum(np.linalg.norm(X[cluster_points_indices] - X[p_idx], axis=1))
                if potential_new_cluster_cost < best_swap_cost:
                    best_swap_cost = potential_new_cluster_cost
                    best_swap_candidate_idx = p_idx
            if best_swap_candidate_idx != current_medoid_for_k_idx:
                new_medoid_indices[k] = best_swap_candidate_idx
                swapped = True
        if not swapped: break
        else:
            medoid_indices = new_medoid_indices.copy()
            medoid_indices_history.append(medoid_indices.copy())
            cluster_idx_to_medoid_idx = {i: medoid_indices[i] for i in range(n_clusters)}
    n_iter_completed = iter_num + 1
    if len(labels_history) == n_iter_completed: labels_final = labels_history[-1]
    elif len(labels_history) > 0 : labels_final = labels_history[-1]
    else:
         distances = cdist(X, X[medoid_indices_history[0]], metric='euclidean')
         labels_final = np.argmin(distances, axis=1)
         labels_history.append(labels_final.copy())
         cost_history.append(np.sum(np.min(distances, axis=1)))
    return labels_final, medoid_indices_history[-1], medoid_indices_history, labels_history, cost_history, n_iter_completed

# --- Funzione per Visualizzare Iterazione K-Medoids (Invariata) ---

def show_kmedoids_iteration(iteration, X_viz, medoid_indices_history, labels_history, n_clusters, colors):
    fig, ax = plt.subplots(figsize=(7, 4.9)) 
    current_medoid_indices = medoid_indices_history[iteration]
    if np.max(current_medoid_indices) >= X_viz.shape[0]: return fig
    current_medoids_viz = X_viz[current_medoid_indices]
    if iteration < len(labels_history):
        labels = labels_history[iteration]
        for k in range(n_clusters):
            cluster_points = X_viz[labels == k]
            if len(cluster_points) > 0: ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[k], alpha=0.5, s=40)
    else:
         if labels_history:
             labels = labels_history[-1]
             for k in range(n_clusters):
                cluster_points = X_viz[labels == k]
                if len(cluster_points) > 0: ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[k], alpha=0.5, s=40)
         else: ax.scatter(X_viz[:, 0], X_viz[:, 1], alpha=0.5, s=40, color='grey')
    ax.scatter(current_medoids_viz[:, 0], current_medoids_viz[:, 1], color=[colors[k] for k in range(n_clusters)], marker='X', s=180, edgecolor='k', linewidth=1.5, label='Medoidi Attuali')
    if iteration > 0:
        for i in range(iteration):
            if i >= len(medoid_indices_history) or i+1 >= len(medoid_indices_history): continue
            prev_medoid_indices = medoid_indices_history[i]
            curr_medoid_indices_step = medoid_indices_history[i+1]
            if np.max(prev_medoid_indices) >= X_viz.shape[0] or np.max(curr_medoid_indices_step) >= X_viz.shape[0]: continue
            prev_medoids_viz = X_viz[prev_medoid_indices]
            curr_medoids_viz_step = X_viz[curr_medoid_indices_step]
            for k in range(n_clusters):
                 if k < len(prev_medoid_indices) and k < len(curr_medoid_indices_step):
                     if prev_medoid_indices[k] != curr_medoid_indices_step[k]: ax.plot([prev_medoids_viz[k, 0], curr_medoids_viz_step[k, 0]], [prev_medoids_viz[k, 1], curr_medoids_viz_step[k, 1]], color=colors[k], linestyle='--', alpha=0.4, linewidth=1)
                     else: ax.scatter(prev_medoids_viz[k, 0], prev_medoids_viz[k, 1], marker='o', s=15, alpha=0.3, color=colors[k], edgecolor=None)
    if iteration == 0:
        ax.scatter(current_medoids_viz[:, 0], current_medoids_viz[:, 1], marker='P', s=100, c='lime', edgecolor='black', label='Medoidi Iniziali (Iter 0)', alpha=0.8)
        ax.legend()
    ax.set_title(f'K-Medoids - Iterazione {iteration}')
    ax.set_xlabel('Componente Principale 1')
    ax.set_ylabel('Componente Principale 2')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# --- Configurazione Pagina Streamlit ---
# INVARIATA (presa dal tuo prompt)
st.set_page_config(layout="wide", page_title="K-Medoids Didattico")
st.title("App Didattica: K-Medoids")
st.write("""
Questa applicazione dimostra il funzionamento dell'algorimo K-Medoids.
I dati sono generati con 2 feature demografiche e 2 comportamentali
per osservare come il clustering (PAM) gestisce questa struttura.
""")

# --- Sidebar per Input Utente ---
# INVARIATA (presa dal tuo prompt)
st.sidebar.header("Parametri Dati")
n_samples = st.sidebar.slider("Numero Totale Record:", min_value=100, max_value=10000, value=400, step=100)
st.sidebar.header("Parametri K-Medoids")
k_clusters_run = st.sidebar.slider("Numero di Cluster da Trovare (k):", min_value=2, max_value=10, value=4, step=1)
max_iterations_run = st.sidebar.slider("Numero Massimo Iterazioni:", min_value=1, max_value=20, value=10, step=1)
random_state_run = st.sidebar.slider("Random Seed:", min_value=0, max_value=100, value=42)

# --- Parametri Fissi per la Generazione Strutturata ---
# INVARIATA (presa dal tuo prompt)
N_CENTERS_DEMO_FIXED = 2
N_CENTERS_BEHAV_FIXED = 2
CLUSTER_STD_FIXED = 0.8

# --- 1. Generazione Dati Strutturati ---
# INVARIATA (presa dal tuo prompt)
st.header("1. Generazione Dataset")
df_original = generate_structured_marketing_data(n_samples=n_samples, n_centers_demo=N_CENTERS_DEMO_FIXED, n_centers_behav=N_CENTERS_BEHAV_FIXED, cluster_std=CLUSTER_STD_FIXED, random_state=random_state_run)
st.write(f"Generato un dataset con {n_samples} clienti e 4 feature (Età, Reddito, SpesaMedia, FrequenzaAq).")
# st.caption(f"(Struttura interna: ...)") # RIMOSSO nel tuo codice
st.write("**Anteprima Dataset Originale:**")
st.dataframe(df_original.head())
numerical_features = df_original.columns.tolist()
# Download Button df_original (invariato nel tuo codice)
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
csv_original = convert_df_to_csv(df_original)
st.download_button(label="Scarica Dataset Originale (CSV)", data=csv_original, file_name='dati_originali.csv', mime='text/csv', key='download-original')

# --- Visualizzazione Dati Generati (PCA) ---

st.subheader("Visualizzazione Dati Generati (Prima del Clustering)")
st.write("Visualizzazione PCA 2D dei dati generati.")
scaler_viz = StandardScaler()
try:
    df_scaled_viz = scaler_viz.fit_transform(df_original)
    pca_viz = PCA(n_components=2, random_state=random_state_run)
    data_2d_viz = pca_viz.fit_transform(df_scaled_viz)
    fig_viz, ax_viz = plt.subplots(figsize=(5.6, 4.2)) # figsize mantenuto come nel tuo codice
    ax_viz.scatter(data_2d_viz[:, 0], data_2d_viz[:, 1], alpha=0.6, s=20)
    ax_viz.set_title('Dati Generati (Visualizzazione PCA 2D)')
    ax_viz.set_xlabel('Componente Principale 1')
    ax_viz.set_ylabel('Componente Principale 2')
    ax_viz.grid(True, linestyle='--', alpha=0.5)
    explained_variance_ratio = pca_viz.explained_variance_ratio_
    st.caption(f"Varianza Spiegata da PCA: Componente 1 = {explained_variance_ratio[0]:.2%}, Componente 2 = {explained_variance_ratio[1]:.2%}")
    st.pyplot(fig_viz) # use_container_width non presente nel tuo codice

    st.write("**Componenti Principali (Loadings):**")
    st.caption("Mostra quanto ogni feature originale contribuisce alle prime 2 componenti PCA.")
    loadings = pd.DataFrame(pca_viz.components_.T, columns=['PC1', 'PC2'], index=numerical_features)
    st.dataframe(loadings.style.format("{:.3f}"))
except Exception as e:
    st.error(f"Errore durante la visualizzazione PCA iniziale: {e}")

# --- 2. Preprocessing Dati ---

st.header("2. Preprocessing Dati")
st.write("Standardizzazione delle 4 feature numeriche (media 0, dev. std. 1).")
scaler = StandardScaler()
try:
    df_processed_np = scaler.fit_transform(df_original)
    df_processed = pd.DataFrame(df_processed_np, columns=numerical_features, index=df_original.index)
    st.write("**Anteprima Dataset Preprocessato:**")
    st.dataframe(df_processed.head())
    # st.write(f"Dimensioni dopo preprocessing: ...") # RIMOSSO nel tuo codice
    # Download Button df_processed (invariato nel tuo codice)
    csv_processed = convert_df_to_csv(df_processed)
    st.download_button(label="Scarica Dataset Preprocessato (CSV)", data=csv_processed, file_name='dati_processati.csv', mime='text/csv', key='download-processed')
except Exception as e:
    st.error(f"Errore durante il preprocessing: {e}")
    st.stop()

# --- 3. Algoritmo K-Medoids (Manuale PAM) ---
st.header("3. Clustering K-Medoids (Manuale PAM)")
st.write(f"Esecuzione K-Medoids con k = {k_clusters_run}, max_iter = {max_iterations_run}...")

try:
    with st.spinner("Esecuzione K-Medoids ..."):
        labels_final, medoid_indices_final, medoid_indices_history, labels_history, cost_history, n_iter_completed = kmedoids_manual_pam( df_processed_np, n_clusters=k_clusters_run, max_iter=max_iterations_run, random_state=random_state_run)
    st.success(f"Algoritmo K-Medoids completato in **{n_iter_completed}** iterazioni.")
    if cost_history:
         fig_cost, ax_cost = plt.subplots(figsize=(5.6, 4.2)) 
         ax_cost.plot(range(len(cost_history)), cost_history, marker='o', markersize=4)
         ax_cost.set_title("Evoluzione del Costo Totale per Iterazione")
         ax_cost.set_xlabel("Iterazione")
         ax_cost.set_ylabel("Costo Totale (Somma Distanze)")
         ax_cost.grid(True)
         st.pyplot(fig_cost) 
         st.write(f"Costo finale: {cost_history[-1]:.2f}")
    df_analysis = df_original.copy()
    df_analysis['Cluster'] = labels_final
    
    csv_analysis = convert_df_to_csv(df_analysis)
    st.download_button(label="Scarica Risultati Clustering (CSV)", data=csv_analysis, file_name='risultati_clustering.csv', mime='text/csv', key='download-analysis')

    

except Exception as e:
    st.error(f"Errore durante l'esecuzione di K-Medoids: {e}")
    st.exception(e)
    st.stop()


# --- 4. Analisi e Visualizzazione Risultati ---

st.header("4. Analisi e Visualizzazione dei Cluster")
if df_processed_np.shape[1] >= 2:
    try: pca = PCA(n_components=2, random_state=random_state_run); data_2d = pca.fit_transform(df_processed_np)
    except Exception as e: st.error(f"Errore durante PCA per visualizzazione: {e}"); st.stop()
elif df_processed_np.shape[1] == 1: st.warning("PCA non applicabile con una sola feature."); data_2d = np.hstack((df_processed_np, np.zeros_like(df_processed_np)))
else: st.warning("Nessuna feature numerica disponibile per PCA."); st.stop()

# --- Sezione Animazione ---

st.subheader("Visualizzazione Iterazioni K-Medoids")
if not medoid_indices_history or not labels_history:
     st.warning("Dati storici non disponibili per la visualizzazione (possibile k > n_samples?).")
else:
    iteration_text = st.empty()
    plot_placeholder = st.empty()
    cmap_viz = cm.get_cmap('viridis', max(k_clusters_run, 10))
    colors_viz = [cmap_viz(float(i) / k_clusters_run) for i in range(k_clusters_run)]
    num_display_iterations = len(medoid_indices_history)
    max_slider_value = max(0, num_display_iterations - 1)
    selected_iteration = st.slider( "Seleziona iterazione:", min_value=0, max_value=max_slider_value, value=max_slider_value)
    if selected_iteration < num_display_iterations:
        iteration_text.write(f"**Iterazione {selected_iteration}/{num_display_iterations - 1}**")
        plot_placeholder.pyplot(show_kmedoids_iteration(selected_iteration, data_2d, medoid_indices_history, labels_history, k_clusters_run, colors_viz)) # use_container_width non presente
    else: st.warning("Nessuna iterazione da visualizzare.")
    if st.button("Avvia Animazione K-Medoids"):
         for i in range(num_display_iterations):
             iteration_text.write(f"**Iterazione {i}/{num_display_iterations - 1}**")
             plot_placeholder.pyplot(show_kmedoids_iteration(i, data_2d, medoid_indices_history, labels_history, k_clusters_run, colors_viz)) # use_container_width non presente
             time.sleep(0.7)

# --- 5. Valutazione e Descrizione Cluster Finali ---

st.header("5. Valutazione e Descrizione Cluster Finali")
col_eval1, col_eval2 = st.columns(2)
# COLONNA 1: Analisi Descrittiva
with col_eval1:
    st.subheader("Analisi Descrittiva")
    st.write("Basata sui dati *originali* e cluster finali:")
    if 'df_analysis' in locals() and 'Cluster' in df_analysis.columns:
        cluster_sizes = df_analysis['Cluster'].value_counts().sort_index()
        st.write("**Dimensione Cluster:**")
        st.dataframe(cluster_sizes.rename("Numero Campioni"))
        unique_clusters = sorted(df_analysis['Cluster'].unique())
        for i in unique_clusters:
            cluster_data = df_analysis[df_analysis['Cluster'] == i]
            st.markdown(f"**Cluster {i}**")
            if not cluster_data.empty:
                st.write("*Feature (Media):*")
                st.dataframe(cluster_data[numerical_features].mean().to_frame('Media').T)
            st.markdown("---")
    else:
        st.warning("Dati di analisi o etichette cluster non disponibili.")

#  Tabella Medoidi Finali ---
    st.markdown("---") # Separatore
    st.subheader("Clienti Rappresentativi (Medoidi Finali)")
    st.write("Questi sono i dati originali dei clienti scelti come punto più centrale (medoide) per ciascun cluster alla fine dell'algoritmo.")
    # Assicurati che medoid_indices_final sia definito e non vuoto
    if 'medoid_indices_final' in locals() and medoid_indices_final is not None and len(medoid_indices_final) > 0:
         # Crea un DataFrame per una visualizzazione migliore, includendo l'ID del cluster
         medoids_df_list = []
         for cluster_idx, medoid_idx in enumerate(medoid_indices_final):
              medoid_data = df_original.iloc[[medoid_idx]].copy() # Usa doppio [[]] per mantenere DataFrame
              medoid_data.insert(0, 'Cluster Assegnato', cluster_idx)
              medoids_df_list.append(medoid_data)
         if medoids_df_list:
             final_medoids_df = pd.concat(medoids_df_list)
             st.dataframe(final_medoids_df.set_index('Cluster Assegnato')) # Mostra con indice cluster
         else:
              st.warning("Nessun dato medoide finale da mostrare.")
    else:
         st.warning("Indici dei medoidi finali non disponibili.")

# COLONNA 2: Analisi Silhouette
with col_eval2:
    st.subheader("Analisi Silhouette")
    st.write("Basata sull'assegnazione finale dei cluster.")
    if 'labels_final' in locals() and labels_final is not None and len(np.unique(labels_final)) > 1 :
        try:
            fig_sil, ax_sil = plt.subplots(figsize=(7, 4.9)) 
            silhouette_avg_val = plot_silhouette(df_processed_np, labels_final, k_clusters_run, ax_sil)
            st.pyplot(fig_sil) # use_container_width non presente nel tuo codice
            st.caption("Punteggi più alti indicano cluster più densi e meglio separati.")
        except Exception as e:
            st.error(f"Errore durante la creazione del plot Silhouette: {e}")
    elif 'labels_final' in locals() and labels_final is not None and len(np.unique(labels_final)) <= 1:
         st.warning("L'analisi Silhouette richiede almeno 2 cluster per essere calcolata.")
    else:
        st.warning("Etichette finali non disponibili per l'analisi Silhouette.")

# --- Fine App ---
st.sidebar.markdown("---")
st.sidebar.info("App didattica K-Medoids - Streamlit") # Mantenuta info originale