import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
import streamlit as st
from meteostat import Point, Hourly
from windrose import WindroseAxes

# Configuration des données
DATA_URL = 'https://www.dropbox.com/scl/fi/jbvta5sryx6e7hyoj42ka/aaa.txt?rlkey=vwgbhsh9rb5p4pedc2usi28v2&st=2pnzfoy2&raw=1'
A=["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
   "Juillet", "Août", "Septembre", "Octobre","Novembre",
   "Décembre", "Automne", "Hiver", "Printemps", "Été","Annule",]
@st.cache_data
def load_sites_data(url):
    """Charge les données des sites avec cache"""
    try:
        df = pd.read_csv(url)
        return df['site'].tolist(), df['x'].tolist(), df['y'].tolist()
    except Exception as e:
        st.error(f"Erreur de chargement des données: {str(e)}")
        return [], [], []
def f2(a, b, n):
    zlist = [np.sum((b >= i) & (b < i + 1) & (a == n)) for i in range(25)]
    ff = np.array(zlist) / sum(zlist)
    return ff

def para(p, v, t):
    if t > 12:
        fer = f2(p, v, t+1)
    elif t == 12:
        fer = (f2(p, v, 9)*9 + f2(p, v, 10)*31 + f2(p, v, 11)*30 + f2(p, v, 12)*20) / 90
    elif t == 13:
        fer = (f2(p, v, 12)*11 + f2(p, v, 1)*31 + f2(p, v, 2)*28 + f2(p, v, 3)*21) / 91
    elif t == 14:
        fer = (f2(p, v, 4)*10 + f2(p, v, 1)*30 + f2(p, v, 5)*31 + f2(p, v, 6)*21) / 92
    elif t == 15:
        fer = (f2(p, v, 6)*9 + f2(p, v, 7)*31 + f2(p, v, 8)*31 + f2(p, v, 9)*21) / 92
    else:
        hgy  = (f2(p, v, 9)*9  + f2(p, v, 10)*31 + f2(p, v, 11)*30 + f2(p, v, 12)*20) / 90
        hgy1 = (f2(p, v, 12)*11 + f2(p, v, 1)*31 + f2(p, v, 2)*28 + f2(p, v, 3)*21) / 91
        hgy2 = (f2(p, v, 4)*10 + f2(p, v, 1)*30 + f2(p, v, 5)*31 + f2(p, v, 6)*21) / 92
        hgy3 = (f2(p, v, 6)*9  + f2(p, v, 7)*31 + f2(p, v, 8)*31 + f2(p, v, 9)*21) / 92
        fer  = (hgy*90 + hgy1*91 + hgy2*92 + hgy3*92) / 365

    x1 = np.array([i + 0.5 for i in range(25)])
    x2 = x1**2

    xm = np.sum(fer * x1)
    xm2 = np.sum(fer * x2)
    xec = (xm2 - xm**2)**0.5

    k = (xec / xm)**-1.090
    C = xm / math.gamma(1 + 1/k)

    return fer, C, k, xm

def prepare_plot(fig):
    """Nettoie la mémoire matplotlib après affichage"""
    st.pyplot(fig)
    plt.close(fig)

def plot_weibull(k, c, v, frequency, title):
    """Crée le graphique Weibull"""
    fig, ax = plt.subplots()
    x = np.arange(0, 25, 0.1)
    y = np.exp(-(x / c) ** k)
    
    ax.bar(np.arange(len(frequency)), frequency, width=1, alpha=0.7, color='blue')
    ax.plot(x, y, color='red')
    ax.set_title(title)
    prepare_plot(fig)

def plot_wind_rose(wind_speed, wind_direction, title):
    """Crée la rose des vents"""
    fig = plt.figure()
    ax = WindroseAxes.from_ax(fig=fig, rect=[0.1, 0.1, 0.8, 0.8])
    ax.bar(wind_direction, wind_speed, normed=True, bins=np.arange(0, 20, 4), edgecolor="black")
    ax.set_legend(title="Vitesse (m/s)")
    ax.set_title(title)
    prepare_plot(fig)

def main():
    """Fonction principale de l'application"""
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
        selected_per = st.selectbox("période:", A)
        start_date = st.date_input("Date début:", datetime.today() - timedelta(days=30))
        end_date = st.date_input("Date fin:", datetime.today())
        
        if st.button("Analyser", type="primary"):
            st.session_state.analyze = True

    # Traitement des données
    if 'analyze' in st.session_state:
        try:
            index = site_names.index(selected_site)
            lat = float(y_coords[index])
            lon = float(x_coords[index])
            Pindex=int(A.index(selected_per))
            # Conversion des dates
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.min.time())
            
            # Récupération des données
            point = Point(lat, lon)
            data = Hourly(point, start_datetime, end_datetime).fetch()
            
            # Calculs
            wind_speed = data['wspd'] / 3.6  # Conversion en m/s
            wind_direction = data['wdir']
            time=data.index
            j =np.array(time.day)
            m = np.array(time.month)
            dd=para(m,wind_speed, Pindex)
            frequency, _ = np.histogram(wind_speed, bins=np.arange(0, 25, 1))
            c = np.mean(wind_speed) * (1 + 1 / (np.std(wind_speed) / np.mean(wind_speed)) ** 1.090)
            k = (np.std(wind_speed) / np.mean(wind_speed)) ** -1.090

            # Affichage des résultats
            with col2:
                st.subheader(f"Résultats pour {selected_site}")
                tab1, tab2 = st.tabs(["Distribution Weibull", "Rose des Vents"])
                
                with tab1:
                    plot_weibull(k, c, np.mean(wind_speed), frequency, 
                               f"Distribution Weibull - {selected_site}")
                    
                with tab2:
                    plot_wind_rose(wind_speed, wind_direction, 
                                 f"Rose des Vents - {selected_site}")

        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {str(e)}")

if __name__ == "__main__":
    main()
