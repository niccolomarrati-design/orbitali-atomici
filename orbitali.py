"""Motore per gli orbitali dell'atomo idrogenoide (unità atomiche: a0 = 1).

Contenuto:
  - funzioni d'onda ψ = R(r) · Y(θ, φ), in forma complessa e in forma reale (chimica)
  - estrazione delle isosuperfici con marching cubes
  - configurazione elettronica degli elementi
  - testo di spiegazione generato dai numeri quantici
"""
import numpy as np
from scipy.special import genlaguerre, factorial
from skimage.measure import marching_cubes

LETTERE = "spdfghi"
EPSILON = 1e-12

# ----------------------------------------------------------------------------
# Armoniche sferiche (compatibile con scipy vecchi e nuovi)
# ----------------------------------------------------------------------------
try:
    from scipy.special import sph_harm_y

    def _sph(l, m, theta, phi):          # theta = polare, phi = azimutale
        return sph_harm_y(l, m, theta, phi)
except ImportError:                       # scipy < 1.15
    from scipy.special import sph_harm

    def _sph(l, m, theta, phi):
        return sph_harm(m, l, phi, theta)


def armonica(l, m, theta, phi):
    """Y_l^m(θ, φ) valida anche per m < 0."""
    if m < 0:
        return (-1) ** (-m) * np.conj(_sph(l, -m, theta, phi))
    return _sph(l, m, theta, phi)


# ----------------------------------------------------------------------------
# Parte radiale e funzioni d'onda
# ----------------------------------------------------------------------------
def radiale(n, l, r):
    """R_nl(r) normalizzata (Z = 1, unità atomiche)."""
    rho = 2.0 * r / n
    norma = np.sqrt((2.0 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    return norma * np.exp(-rho / 2) * rho ** l * genlaguerre(n - l - 1, 2 * l + 1)(rho)


def sferiche(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(np.clip(z / (r + EPSILON), -1, 1))
    phi = np.arctan2(y, x)
    return r, theta, phi


def psi_complessa(n, l, m, x, y, z):
    """Orbitale 'quantistico': autofunzione di H, L² e Lz (complessa)."""
    r, theta, phi = sferiche(x, y, z)
    return radiale(n, l, r) * armonica(l, m, theta, phi)


def psi_reale(n, l, m, x, y, z):
    """Orbitale 'da chimica': combinazione reale di +m e -m (stessa energia)."""
    r, theta, phi = sferiche(x, y, z)
    if m == 0:
        ang = armonica(l, 0, theta, phi).real
    elif m > 0:      # tipo cos(mφ): pₓ, dxz, dx²−y², ...
        ang = np.sqrt(2) * (-1) ** m * armonica(l, m, theta, phi).real
    else:            # tipo sin(|m|φ): p_y, dyz, dxy, ...
        ang = np.sqrt(2) * (-1) ** m * armonica(l, -m, theta, phi).imag
    return radiale(n, l, r) * ang


# ----------------------------------------------------------------------------
# Dimensioni della griglia e soglia
# ----------------------------------------------------------------------------
def raggio_contenente(n, l, frazione=0.995):
    """Raggio entro cui sta la frazione data della probabilità radiale."""
    rr = np.linspace(0, 6 * n * n + 40, 40000)
    p = rr ** 2 * radiale(n, l, rr) ** 2
    c = np.cumsum(p)
    c /= c[-1]
    return float(rr[min(np.searchsorted(c, frazione), len(rr) - 1)])


def soglia_densita(dens, percentuale):
    """Valore di |ψ|² tale che la regione |ψ|² > soglia contiene la % di probabilità."""
    v = np.sort(dens.ravel())[::-1]
    c = np.cumsum(v)
    c /= c[-1]
    return float(v[min(np.searchsorted(c, percentuale / 100.0), len(v) - 1)])


def superfici(n, l, m, modo="reale", percentuale=90, N=80):
    """Calcola le mesh da disegnare. Restituisce (lista_mesh, L)."""
    L = raggio_contenente(n, l) * 1.05
    asse = np.linspace(-L, L, N)
    passo = 2 * L / (N - 1)
    x, y, z = np.meshgrid(asse, asse, asse, indexing="ij")
    mesh = []

    if modo == "reale":
        psi = psi_reale(n, l, m, x, y, z)
        s = np.sqrt(soglia_densita(psi ** 2, percentuale))
        for segno, colore, nome in ((1, "#e63946", "ψ > 0"), (-1, "#1d6fd8", "ψ < 0")):
            livello = segno * s
            if psi.min() < livello < psi.max():
                v, f, _, _ = marching_cubes(psi, level=livello, spacing=(passo,) * 3)
                mesh.append(dict(verts=v - L, faces=f, colore=colore, nome=nome, fase=None))
    else:
        psi = psi_complessa(n, l, m, x, y, z)
        dens = np.abs(psi) ** 2
        t = soglia_densita(dens, percentuale)
        v, f, _, _ = marching_cubes(dens, level=t, spacing=(passo,) * 3)
        v = v - L
        fase = np.angle(psi_complessa(n, l, m, v[:, 0], v[:, 1], v[:, 2]))
        mesh.append(dict(verts=v, faces=f, colore=None, nome="|ψ|² (colore = fase)", fase=fase))
    return mesh, L


def distribuzione_radiale(n, l, L, punti=600):
    r = np.linspace(0, L, punti)
    return r, r ** 2 * radiale(n, l, r) ** 2


# ----------------------------------------------------------------------------
# Configurazione elettronica
# ----------------------------------------------------------------------------
SIMBOLI = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
           "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba "
           "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
           "Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds "
           "Rg Cn Nh Fl Mc Lv Ts Og").split()

ORDINE_MADELUNG = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0),
                   (4, 2), (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0), (5, 3), (6, 2), (7, 1)]

# Eccezioni al riempimento (n + l crescente): occupazioni da sovrascrivere.
ECCEZIONI = {
    24: {(4, 0): 1, (3, 2): 5},            29: {(4, 0): 1, (3, 2): 10},
    41: {(5, 0): 1, (4, 2): 4},            42: {(5, 0): 1, (4, 2): 5},
    44: {(5, 0): 1, (4, 2): 7},            45: {(5, 0): 1, (4, 2): 8},
    46: {(5, 0): 0, (4, 2): 10},           47: {(5, 0): 1, (4, 2): 10},
    57: {(4, 3): 0, (5, 2): 1},            58: {(4, 3): 1, (5, 2): 1},
    64: {(4, 3): 7, (5, 2): 1},            78: {(6, 0): 1, (5, 2): 9},
    79: {(6, 0): 1, (5, 2): 10},           89: {(5, 3): 0, (6, 2): 1},
    90: {(5, 3): 0, (6, 2): 2},            91: {(5, 3): 2, (6, 2): 1},
    92: {(5, 3): 3, (6, 2): 1},            93: {(5, 3): 4, (6, 2): 1},
    96: {(5, 3): 7, (6, 2): 1},            103: {(6, 2): 0, (7, 1): 1},
}
GAS_NOBILI = [(2, "He"), (10, "Ne"), (18, "Ar"), (36, "Kr"), (54, "Xe"), (86, "Rn")]
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def configurazione(Z):
    """Dizionario {(n, l): elettroni} dello stato fondamentale."""
    occ, resto = {}, Z
    for n, l in ORDINE_MADELUNG:
        e = min(resto, 2 * (2 * l + 1))
        if e:
            occ[(n, l)] = e
        resto -= e
        if resto == 0:
            break
    occ.update(ECCEZIONI.get(Z, {}))
    return {k: v for k, v in occ.items() if v > 0}


def _testo(occ):
    return " ".join(f"{n}{LETTERE[l]}{str(e).translate(_SUP)}"
                    for (n, l), e in sorted(occ.items()))


def configurazione_testo(Z):
    """Restituisce (completa, abbreviata)."""
    occ = configurazione(Z)
    completa = _testo(occ)
    nobili = [(zn, s) for zn, s in GAS_NOBILI if zn < Z]
    if not nobili:
        return completa, completa
    zn, simbolo = nobili[-1]
    nucleo = configurazione(zn)
    resto = {k: v for k, v in occ.items() if nucleo.get(k, 0) != v}
    return completa, f"[{simbolo}] {_testo(resto)}"


# ----------------------------------------------------------------------------
# Nomi e spiegazione
# ----------------------------------------------------------------------------
NOMI_P = {0: "pz", 1: "px", -1: "py"}
NOMI_D = {0: "dz²", 1: "dxz", -1: "dyz", 2: "dx²−y²", -2: "dxy"}


def nome_orbitale(n, l, m, modo):
    if modo == "reale":
        if l == 1:
            return f"{n}{NOMI_P[m]}"
        if l == 2:
            return f"{n}{NOMI_D[m]}"
    return f"{n}{LETTERE[l]} (m = {m})"


def spiegazione(n, l, m, modo):
    lettera = LETTERE[l]
    nodi_rad = n - l - 1
    energia = -13.6057 / n ** 2
    r_medio = (3 * n ** 2 - l * (l + 1)) / 2
    forme = {
        0: "una **sfera**: nessuna direzione è privilegiata, perché Y₀₀ è costante.",
        1: "due **lobi opposti** (a manubrio) separati da un piano nodale che passa dal nucleo.",
        2: "**quattro lobi** a trifoglio; per d(z²) invece due lobi lungo z più un anello sul piano xy.",
        3: "**otto lobi** (o combinazioni di lobi e anelli): forme sempre più articolate.",
    }
    forma = forme.get(l, "forme con molti lobi, separati da l superfici nodali angolari.")

    t = [f"### Orbitale {nome_orbitale(n, l, m, modo)}", ""]
    t.append(f"**Numeri quantici:** n = {n}, l = {l} (sottoguscio *{lettera}*), m = {m}.")
    t.append("")
    t.append(f"**Forma:** {forma}")
    t.append("")
    t.append(
        f"**Nodi.** In totale n − 1 = {n - 1}. Di questi, {nodi_rad} sono **radiali** "
        f"(n − l − 1: sfere in cui ψ si annulla) e {l} sono **angolari** (l: piani o coni "
        f"che passano per il nucleo)." + (
            f" Dei nodi angolari, {abs(m)} sono piani contenenti l'asse z e {l - abs(m)} sono coni."
            if l > 0 else ""))
    t.append("")
    if modo == "reale":
        t.append("**Rappresentazione reale (chimica).** Per m ≠ 0 l'orbitale è una combinazione "
                 "di +m e −m (come pₓ e p_y a partire da m = ±1). Hanno la stessa energia, "
                 "quindi la combinazione è ancora un'autofunzione di H. "
                 "I due colori indicano il **segno di ψ** (rosso positivo, blu negativo): "
                 "il segno non cambia |ψ|², ma conta nella formazione dei legami.")
    else:
        t.append("**Rappresentazione complessa (quantistica).** Qui ψ ∝ e^{imφ}: |ψ|² non dipende "
                 "da φ, quindi per m ≠ 0 la nuvola ha simmetria cilindrica attorno a z "
                 "(una ciambella per p, m = ±1). L'informazione su m sta nella **fase**, "
                 "mostrata come colore: compie |m| giri completi attorno all'asse z, "
                 "con verso opposto per +m e −m.")
    t.append("")
    t.append(f"**Grandezze.** Energia (idrogeno) E = −13.6 eV / n² ≈ **{energia:.2f} eV**; "
             f"degenerazione del livello n: n² = {n ** 2} orbitali; "
             f"raggio medio ⟨r⟩ = (3n² − l(l+1))/2 · a₀ ≈ **{r_medio:.1f} a₀** "
             f"({r_medio * 0.529:.2f} Å).")
    t.append("")
    t.append("**Disegno.** La superficie racchiude la regione dove |ψ|² supera una soglia, "
             "scelta in modo che dentro ci sia la percentuale di probabilità impostata "
             "(90% è la convenzione dei libri). È una convenzione: l'orbitale vero non ha confine.")
    return "\n".join(t)
