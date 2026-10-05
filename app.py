"""Interfaccia: streamlit run app.py"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import orbitali as orb

st.set_page_config(page_title="Orbitali atomici", page_icon="⚛️", layout="wide")


@st.cache_data(show_spinner="Calcolo l'orbitale...")
def calcola(n, l, m, modo, percentuale, N):
    return orb.superfici(n, l, m, modo, percentuale, N)


def costruisci_figura(mesh, L):
    fig = go.Figure()
    for s in mesh:
        v, f = s["verts"], s["faces"]
        comune = dict(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                      i=f[:, 0], j=f[:, 1], k=f[:, 2],
                      opacity=0.7, name=s["nome"], showlegend=True,
                      lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3))
        if s["fase"] is None:
            fig.add_trace(go.Mesh3d(color=s["colore"], **comune))
        else:
            fig.add_trace(go.Mesh3d(intensity=s["fase"], intensitymode="vertex",
                                    colorscale="HSV", cmin=-np.pi, cmax=np.pi,
                                    colorbar=dict(title="fase", tickvals=[-np.pi, 0, np.pi],
                                                  ticktext=["−π", "0", "π"], len=0.6),
                                    **comune))

    # assi che attraversano l'origine, con etichette
    A = L * 1.15
    for nome, (dx, dy, dz) in {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}.items():
        fig.add_trace(go.Scatter3d(
            x=[-A * dx, A * dx], y=[-A * dy, A * dy], z=[-A * dz, A * dz],
            mode="lines+text", text=["", nome], textposition="top center",
            textfont=dict(size=16), line=dict(color="black", width=4),
            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                               marker=dict(size=4, color="black"),
                               name="nucleo", showlegend=False))

    rng = [-A * 1.1, A * 1.1]
    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(aspectmode="cube",
                   xaxis=dict(range=rng, title="x [a₀]"),
                   yaxis=dict(range=rng, title="y [a₀]"),
                   zaxis=dict(range=rng, title="z [a₀]")),
        legend=dict(orientation="h", y=0.02))
    return fig


# ------------------------------- barra laterale -------------------------------
st.sidebar.header("Numeri quantici")
n = st.sidebar.selectbox("n  (principale)", list(range(1, 8)), index=1)
l = st.sidebar.selectbox("l  (azimutale)", list(range(n)),
                         format_func=lambda v: f"{v}  ({orb.LETTERE[v]})", index=min(1, n - 1))
m = st.sidebar.selectbox("m  (magnetico)", list(range(-l, l + 1)), index=l)

st.sidebar.header("Visualizzazione")
modo_lbl = st.sidebar.radio("Tipo di orbitale",
                            ["Reale (chimica)", "Complesso (quantistico)"],
                            help="Reale: combinazioni di +m e −m (px, py, ...). "
                                 "Complesso: autofunzioni di Lz, colorate in base alla fase.")
modo = "reale" if modo_lbl.startswith("Reale") else "complesso"
percentuale = st.sidebar.slider("Probabilità racchiusa (%)", 50, 99, 90)
N = st.sidebar.slider("Qualità griglia (N³ punti)", 40, 120, 80, step=10,
                      help="Più alta = più dettaglio, ma più lenta.")

st.sidebar.header("Elemento")
Z = st.sidebar.selectbox("Atomo", list(range(1, 119)),
                         format_func=lambda z: f"{z} – {orb.SIMBOLI[z - 1]}", index=5)

# --------------------------------- pagina ---------------------------------
st.title("⚛️ Orbitali atomici")
st.caption("Orbitali dell'atomo idrogenoide in unità atomiche (a₀ = raggio di Bohr). "
           "Trascina per ruotare, rotella o pinch per zoomare.")

mesh, L = calcola(n, l, m, modo, percentuale, N)
st.subheader(f"Orbitale {orb.nome_orbitale(n, l, m, modo)}")

tab3d, tabrad, tabspieg, tabconf = st.tabs(
    ["🧊 Orbitale 3D", "📈 Distribuzione radiale", "📖 Spiegazione", "🧪 Configurazione elettronica"])

with tab3d:
    st.plotly_chart(costruisci_figura(mesh, L), use_container_width=True)

with tabrad:
    r, P = orb.distribuzione_radiale(n, l, L)
    figr = go.Figure(go.Scatter(x=r, y=P, mode="lines", fill="tozeroy"))
    figr.update_layout(height=420, xaxis_title="r [a₀]", yaxis_title="r² R(r)²",
                       title="Densità di probabilità radiale (4π r² |ψ|² a meno di Y)")
    st.plotly_chart(figr, use_container_width=True)
    st.write(f"Numero di nodi radiali (zeri della curva): **{n - l - 1}**. "
             "Il picco non sta nel nucleo perché il volume del guscio cresce come r², "
             "mentre |ψ|² cala.")

with tabspieg:
    st.markdown(orb.spiegazione(n, l, m, modo))

with tabconf:
    simbolo = orb.SIMBOLI[Z - 1]
    completa, corta = orb.configurazione_testo(Z)
    st.markdown(f"### {simbolo} (Z = {Z})")
    st.markdown(f"**Abbreviata:** {corta}")
    st.markdown(f"**Completa:** {completa}")
    occ = orb.configurazione(Z)
    sott = occ.get((n, l), 0)
    capacita = 2 * (2 * l + 1)
    sigla = f"{n}{orb.LETTERE[l]}"
    if sott:
        st.success(f"Nello stato fondamentale di {simbolo} il sottoguscio **{sigla}** contiene "
                   f"**{sott}** elettroni su un massimo di {capacita} "
                   f"(2 per ognuno dei {2 * l + 1} orbitali, con spin opposti).")
    else:
        st.info(f"Nello stato fondamentale di {simbolo} il sottoguscio **{sigla}** è vuoto.")
    st.caption("Attenzione: negli atomi con più elettroni la parte angolare resta la stessa, "
               "ma la parte radiale cambia (schermatura): l'orbitale disegnato è quello idrogenoide.")
