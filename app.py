import streamlit as st
import random, math, os, io, colorsys, traceback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import colorgram

# -------------------------------
# FILE CONSTANTS
# -------------------------------
PALETTE_FILE = "palette.csv"
REFERENCE_PALETTE_FILE = "reference.csv"

DEFAULT_CSV_DATA = """name,r,g,b
sky,0.4,0.7,1.0
sun,1.0,0.8,0.2
forest,0.2,0.6,0.3
"""

# -------------------------------
# CSV Helpers
# -------------------------------
def initialize_csv(filepath):
    """Ensure palette or reference CSV file exists."""
    if not os.path.exists(filepath):
        if filepath == PALETTE_FILE:
            df = pd.read_csv(io.StringIO(DEFAULT_CSV_DATA))
        else:
            df = pd.DataFrame(columns=["name", "r", "g", "b"])
        df.to_csv(filepath, index=False)

def read_csv_palette(filepath):
    """Read a CSV palette safely."""
    initialize_csv(filepath)
    try:
        df = pd.read_csv(filepath)
        if not {"r", "g", "b"}.issubset(df.columns):
            raise ValueError("Missing RGB columns")
        return df
    except Exception:
        return pd.DataFrame(columns=["name", "r", "g", "b"])

def add_color(name, r, g, b, filepath=PALETTE_FILE):
    """Add one color to a CSV palette."""
    df = read_csv_palette(filepath)
    new_color = pd.DataFrame(
        [{"name": name, "r": float(r), "g": float(g), "b": float(b)}]
    )
    df = pd.concat([df, new_color], ignore_index=True)
    df.to_csv(filepath, index=False)
    return df

def load_palette_from_csv(filepath=PALETTE_FILE):
    """Load RGB tuples from CSV."""
    df = read_csv_palette(filepath)
    if df.empty:
        return [(0.5, 0.5, 0.5)]
    return [(float(r), float(g), float(b)) for r, g, b in zip(df.r, df.g, df.b)]

def extract_palette_from_image(img_bytes, num_colors=6):
    """Extract dominant colors from an uploaded image using colorgram."""
    image_path = "uploaded_image.png"
    with open(image_path, "wb") as f:
        f.write(img_bytes)
    colors = colorgram.extract(image_path, num_colors)
    extracted = [
        {
            "name": f"ref_{i+1}",
            "r": c.rgb.r / 255.0,
            "g": c.rgb.g / 255.0,
            "b": c.rgb.b / 255.0,
        }
        for i, c in enumerate(colors)
    ]
    df = pd.DataFrame(extracted)
    df.to_csv(REFERENCE_PALETTE_FILE, index=False)
    return [(d["r"], d["g"], d["b"]) for d in extracted]

# -------------------------------
# Palette Generators
# -------------------------------
def generate_pastel_palette(k=5):
    return [
        colorsys.hsv_to_rgb(random.random(), random.uniform(0.15, 0.35), random.uniform(0.9, 1.0))
        for _ in range(k)
    ]

def generate_vivid_palette(k=5):
    return [
        colorsys.hsv_to_rgb(random.random(), random.uniform(0.8, 1.0), random.uniform(0.8, 1.0))
        for _ in range(k)
    ]

def generate_mono_palette(k=5, base_h=0.60):
    return [
        colorsys.hsv_to_rgb(
            (base_h + random.uniform(-0.05, 0.05)) % 1.0,
            random.uniform(0.2, 0.8),
            random.uniform(0.5, 1.0),
        )
        for _ in range(k)
    ]

def generate_random_palette(k=5):
    return [(random.random(), random.random(), random.random()) for _ in range(k)]

def get_palette(mode, k=6):
    if mode == "csv":
        return load_palette_from_csv(PALETTE_FILE)
    if mode == "reference":
        return load_palette_from_csv(REFERENCE_PALETTE_FILE)
    if mode == "pastel":
        return generate_pastel_palette(k)
    if mode == "vivid":
        return generate_vivid_palette(k)
    if mode == "mono":
        return generate_mono_palette(k)
    return generate_random_palette(k)

# -------------------------------
# Drawing Helpers
# -------------------------------
def blob(center=(0.5, 0.5), r=0.3, points=200, wobble=0.15):
    """Generate coordinates for a wobbly blob shape."""
    angles = np.linspace(0, 2 * math.pi, points)
    radii = r * (1 + wobble * (np.random.rand(points) - 0.5))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return x, y

def draw_poster(n_layers, wobble, palette_mode, seed):
    """Draw a generative poster and return a Matplotlib Figure."""
    random.seed(seed)
    np.random.seed(seed)

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.axis("off")
    ax.set_facecolor((0.98, 0.98, 0.98))
    palette = get_palette(palette_mode)

    for _ in range(n_layers):
        cx, cy = random.random(), random.random()
        rr = random.uniform(0.05, 0.45)
        x, y = blob(center=(cx, cy), r=rr, wobble=wobble)
        color = random.choice(palette)
        ax.fill(
            x,
            y,
            color=color,
            alpha=random.uniform(0.25, 0.6),
            edgecolor=(0, 0, 0, 0),
        )

    ax.text(
        0.05,
        0.95,
        f"Generative Poster • {palette_mode}",
        fontsize=16,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return fig  # ✅ Return the Figure, not plt

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Generative Poster Studio", layout="wide")
st.title("🎨 Generative Poster Studio")
st.write("Generate algorithmic art using CSV palettes or extracted image palettes!")

# Initialize CSVs
initialize_csv(PALETTE_FILE)
initialize_csv(REFERENCE_PALETTE_FILE)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
palette_mode = st.sidebar.selectbox(
    "Palette Mode", ["csv", "pastel", "vivid", "mono", "random", "reference"]
)
layers = st.sidebar.slider("Number of Layers", 1, 50, 15)
wobble = st.sidebar.slider("Wobble Intensity", 0.01, 2.0, 0.5)
seed = st.sidebar.number_input("Random Seed", 0, 9999, 0)

# --- Image Upload for Reference Palette ---
st.sidebar.subheader("🎨 Extract Colors from Image")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    with st.spinner("Extracting colors..."):
        extracted_palette = extract_palette_from_image(uploaded_file.getvalue(), num_colors=10)
        st.sidebar.success(f"Extracted {len(extracted_palette)} colors into reference.csv")
        st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# --- Palette Visualization ---
st.subheader("Current Palette Preview")
palette_to_show = get_palette(palette_mode)
fig_palette, ax_palette = plt.subplots(figsize=(min(len(palette_to_show), 10), 1))
for i, c in enumerate(palette_to_show):
    ax_palette.fill_between([i, i + 1], 0, 1, color=c)
ax_palette.axis("off")
st.pyplot(fig_palette)

# --- Generate Poster ---
if st.button("🎨 Generate Poster"):
    try:
        fig = draw_poster(layers, wobble, palette_mode, seed)
        st.pyplot(fig)

        # Add a download button
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            label="💾 Download Poster as PNG",
            data=buf,
            file_name=f"poster_{palette_mode}.png",
            mime="image/png",
        )
    except Exception:
        st.error("❌ Error while generating poster:")
        st.text(traceback.format_exc())

# --- CSV Table Display ---
st.subheader("📁 Palette CSV Files")
col1, col2 = st.columns(2)
with col1:
    st.caption("palette.csv")
    st.dataframe(read_csv_palette(PALETTE_FILE))
with col2:
    st.caption("reference.csv")
    st.dataframe(read_csv_palette(REFERENCE_PALETTE_FILE))
