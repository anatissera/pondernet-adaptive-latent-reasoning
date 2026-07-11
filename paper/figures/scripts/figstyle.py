"""Estilo compartido de las figuras del paper / poster.

Dos perfiles tipograficos, cada uno casando con el documento que lo va a mostrar:
  - "poster": Palatino (via P052 / URW Palatino), igual que `\\usepackage{palatino}`
    en el poster.
  - "paper":  Times (via Nimbus Roman / Times New Roman), igual que
    `\\usepackage{times}` en main.tex (ACL).

Los tamaños de fuente (sizes()) son los mismos para los dos perfiles -- un poco
mas grandes que la version original, para que las etiquetas se lean al tamaño
del cuerpo de texto tanto en el poster como en el informe.

Paleta anclada en el **indigo tal como se ve en el poster**: el color institucional
UdeSA (RGB 0,18,187) se renderiza sombreado en las cabeceras de baposter como
~#444d8a, y ese es el azul con el que las figuras tienen que casar en ambos perfiles.

Estetica: fondo blanco liso, monocromo indigo con tintes claros (nada de bloques
planos saturados) y un unico acento calido (frambuesa apagada). Sin naranja.

Uso:
    import figstyle
    figstyle.set_style("paper")  # o "poster"
    FS = figstyle.sizes("paper")
    ax.set_xlabel("...", fontsize=FS["label"])
    ... figstyle.BLUE, figstyle.TINT, figstyle.ACCENT, figstyle.BLUES ...
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# --- paleta de la casa -------------------------------------------------------
BLUE = "#444d8a"     # indigo institucional tal como se ve en el poster (protagonista)
BLUE_DK = "#2e3566"  # indigo profundo: extremo del heatmap / enfasis
BLUE_MID = "#6b73a8"  # indigo intermedio: tercera serie en figuras multi-curva
TINT = "#c6c9e2"     # indigo muy claro: baseline como serie, rellenos suaves
BAND = "#e7e8f2"     # indigo casi blanco: bandas / sombreados
ACCENT = "#b0485f"   # frambuesa apagada: baseline / referencia / segunda serie
GRAY = "#b3b2aa"     # neutro calido, para cuando hace falta un gris real
GRID = "#dedcd4"     # grilla apenas visible
TEXT = "#39394a"     # tinta indigo-gris: casa con el indigo, no negro puro

# tri-tono indigo (poco contraste entre si): sirve para variaciones sutiles,
# pero NO para series que hay que distinguir de un vistazo (ver SERIES abajo).
BLUE_SHADES = [BLUE_MID, BLUE, BLUE_DK]

GOLD = "#e0a526"   # dorado miel vivo: calido y lindo, distinto sin ser naranja
TEAL = "#1f9e89"   # verde azulado vivo: contrasta fuerte con indigo y con dorado

# paleta de series con alto contraste entre si (vivas pero armonicas con el
# indigo/frambuesa de la casa) para figuras multi-curva donde las tonalidades de
# un solo color no se distinguen (p.ej. M0/M1/M2 en la frontera del paper).
SERIES = [GOLD, TEAL, BLUE]

# rampa blanco -> indigo institucional para heatmaps (reemplaza el 'Blues' de mpl)
BLUES = LinearSegmentedColormap.from_list(
    "udesa_indigo",
    ["#ffffff", "#e2e4f0", "#b9bdd9", "#8189b6", "#565f97", BLUE_DK],
)

_FONTS = {
    "poster": {
        "font.serif": ["P052", "TeX Gyre Pagella", "URW Palladio L",
                       "Palatino", "DejaVu Serif"],
        "mathtext.rm": "P052", "mathtext.it": "P052:italic", "mathtext.bf": "P052:bold",
    },
    "paper": {
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                       "Times", "DejaVu Serif"],
        "mathtext.rm": "Nimbus Roman", "mathtext.it": "Nimbus Roman:italic",
        "mathtext.bf": "Nimbus Roman:bold",
    },
}

# tamaños compartidos por los dos perfiles: un empujon moderado respecto de las
# figuras originales (labels/indices un poco mas grandes), igual en poster y paper.
_SIZES = {"label": 13.5, "tick": 12, "legend": 12.5, "annotate": 16}


def sizes(profile="paper"):
    return dict(_SIZES)


def set_style(profile="paper"):
    fonts = _FONTS[profile]
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": fonts["font.serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": fonts["mathtext.rm"],
        "mathtext.it": fonts["mathtext.it"],
        "mathtext.bf": fonts["mathtext.bf"],
        "mathtext.fallback": "stix",
        # --- ejes y texto ---
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.linewidth": 0.8,
        "xtick.labelsize": _SIZES["tick"],
        "ytick.labelsize": _SIZES["tick"],
        # --- grilla tenue ---
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        # --- fondo blanco liso ---
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        # --- salida ---
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })
