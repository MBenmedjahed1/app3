import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
import streamlit as st
from meteostat import Point, Hourly
from windrose import WindroseAxes
import io

# Configuration des données
DATA_URL = 'aaa.txt'
A = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre",
    "Décembre", "Automne", "Hiver", "Printemps", "Été", "Annuel",
]

@st.cache_data
def load_sites_data(url):
    """Charge les données des sites avec cache."""
    try:
        df = pd.read_csv(url)
        return df['site'].tolist(), df['x'].tolist(), df['y'].tolist()
    except Exception as e:
        st.error(f"Erreur de chargement des données: {str(e)}")
        return [], [], []

def f2(a, b, n):
    """Calcul les sommes de occurrences pour la période donnée."""
    zlist = [np.sum((b >= i) & (b < i + 1) & (a == n)) for i in range(25)]
    ff = np.array(zlist) / sum(zlist) if sum(zlist) > 0 else np.zeros_like(zlist)
    return ff

def para(p, v, t):
    """Calcule les paramètres de la distribution Weibull."""
    if t == "Automne":
        fer = (f2(p, v, 9) * 9 + f2(p, v, 10) * 31 + 
                f2(p, v, 11) * 30 + f2(p, v, 12) * 20) / 90
    elif t == "Hiver":
        fer = (f2(p, v, 12) * 11 + f2(p, v, 1) * 31 + 
                f2(p, v, 2) * 28 + f2(p, v, 3) * 21) / 91
    elif t == "Printemps":
        fer = (f2(p, v, 4) * 10 + f2(p, v, 1) * 30 + 
                f2(p, v, 5) * 31 + f2(p, v, 6) * 21) / 92
    elif t == "Été":
        fer = (f2(p, v, 6) * 9 + f2(p, v, 7) * 31 +
               f2(p, v, 8) * 31 + f2(p, v, 9) * 21) / 92
    elif t == "Annuel":
        hgy = (f2(p, v, 9) * 9 + f2(p, v, 10) * 31 + 
               f2(p, v, 11) * 30 + f2(p, v, 12) * 20) / 90
        hgy1 = (f2(p, v, 12) * 11 + f2(p, v, 1) * 31 + 
                 f2(p, v, 2) * 28 + f2(p, v, 3) * 21) / 91
        hgy2 = (f2(p, v, 4) * 10 + f2(p, v, 1) * 30 + 
                 f2(p, v, 5) * 31 + f2(p, v, 6) * 21) / 92
        hgy3 = (f2(p, v, 6) * 9 + f2(p, v, 7) * 31 + 
                 f2(p, v, 8) * 31 + f2(p, v, 9) * 21) / 92
        fer = (hgy1 * 91 + hgy * 90 + hgy2 * 92 + hgy3 * 92) / 365
    else:
        n = int(A.index(t)) + 1
        fer = f2(p, v, n)

    x1 = np.array([i + 0.5 for i in range(25)])
    xm = np.sum(fer * x1)
    xm2 = np.sum(fer * (x1 ** 2))

    # Calculate Weibull parameters
    xec = (xm2 - xm**2) ** 0.5
    k = (xec / xm) ** -1.090
    C = xm / math.gamma(1 + 1/k)

    return fer, C, k, xm

def prepare_plot(fig):
    """Nettoie la mémoire matplotlib après affichage."""
    st.pyplot(fig)
    plt.close(fig)

def plot_weibull(k, c, v,E,frequency):
    """Crée le graphique Weibull."""
    fig, ax = plt.subplots()
    x = np.arange(0, 25, 0.1)
    y = (k/c) * (x/c)**(k-1) * np.exp(-(x/c)**k)
    xx=np.arange(len(frequency))+0.5

    ax.bar(xx, frequency, width=1, alpha=0.7, color='blue', 
           edgecolor='black', label='Histogramme de Fréquence')
    ax.plot(x, y, color='red', label='Courbe de Weibull')
    ax.set_title(f'{E} : C= {round(c, 1)} m/s, k= {round(k, 2)}, ' +
                 r'$\bar{v}$' + f'= {round(v, 1)} m/s', fontsize=8)
    ax.set_xlabel('Vitesse du vent (m/s)')
    ax.set_ylabel('Fréquence')
    ax.set_xlim(0, 25) 
    ax.legend()
    return fig

def plot_wind_rose(wind_speed, wind_direction, title):
    """Crée la rose des vents."""
    fig = plt.figure()
    ax = WindroseAxes.from_ax(fig=fig, rect=[0.1, 0.1, 0.8, 0.8])
    ax.bar(wind_direction, wind_speed, normed=True, bins=np.arange(0, 20, 4), edgecolor="black")
    ax.set_legend(title="Vitesse (m/s)")
    ax.set_title(title)
    return fig

def main():
    """Fonction principale de l'application."""
    st.set_page_config(page_title="Analyse Vent", layout="wide")
    st.title("Analyse des Données Vent")

    # Chargement des données
    site_names, x_coords, y_coords = load_sites_data(DATA_URL)

    if not site_names:
        st.stop()

    # Interface utilisateur
    col1, col2 = st.columns([1, 3])

    with col1:
        selected_site = st.selectbox("Site:", site_names)
        selected_period = st.selectbox("Période:", A)
        start_date = st.date_input("Date début:", datetime.today() - timedelta(days=365), min_value=datetime(1995, 1, 1))
        end_date = st.date_input("Date fin:", datetime.today())
        
        # Convert date inputs to datetime
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)

        if st.button("Analyser", type="primary"):
            st.session_state.analyze = True

    if 'analyze' in st.session_state:
        try:
            index = site_names.index(selected_site)
            lat = float(y_coords[index])
            lon = float(x_coords[index])

            # Récupération des données
            point = Point(lat, lon)
            data = Hourly(point, start_datetime, end_datetime).fetch()

            # Calculs
            wind_speed = data['wspd'] / 3.6  # Conversion en m/s
            wind_direction = data['wdir']
            time = data.index
            m = np.array(time.month)

            # Calculer les paramètres de Weibull
            fer, c, k, xm = para(m, wind_speed, selected_period)

            # Résultats pour toutes les périodes
            results = []
            for t in A:
                fer, C, k, xm = para(m, wind_speed, t)
                results.append([t, round(C, 1), round(k, 2), round(xm, 1)])

            df = pd.DataFrame(results, columns=["Période", "C (échelle)", "k (forme)", "Vitesse moyenne (m/s)"])

            # Affichage des résultats
            with col2:
                st.subheader(f"Résultats pour {selected_site}")

                # Création des onglets
                tab1, tab2, tab3 = st.tabs(["Distribution Weibull", "Rose des Vents", "Tableau Weibull"])

                with tab1:
                    fig_weibull = plot_weibull(k, c, xm,selected_period,fer)
                    prepare_plot(fig_weibull)

                with tab2:
                    fig_windrose = plot_wind_rose(wind_speed, wind_direction, f"Rose des Vents - {selected_site}")
                    prepare_plot(fig_windrose)

                with tab3:
                    st.subheader("Tableau des Paramètres de Weibull")
                    st.dataframe(df)  # Affichage du DataFrame
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Télécharger les résultats", csv, "resultats_weibull.csv", "text/csv")

        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()
