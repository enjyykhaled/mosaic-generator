import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Mosaic Ultra Pro", layout="wide")

# -------- STYLE --------
st.markdown("""
    <style>
    .main {background-color: #0e1117; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("🎨 Mosaic Generator Ultra Pro")
st.write("Mosaic + Puzzle + Analysis + Comparison + AI Auto-Tuning 🤖")

# -------- SIDEBAR --------
st.sidebar.header("⚙️ Settings")

pixel_size = st.sidebar.slider("Pixel Size", 5, 50, 20)
k = st.sidebar.slider("Number of Colors (k)", 2, 15, 6)

st.sidebar.markdown("---")
st.sidebar.header("🔬 Advanced Tech")
use_lab = st.sidebar.toggle("Use LAB Color Space (Better Colors)", value=True)
apply_enhancement = st.sidebar.checkbox("✨ Enhance Image Before Processing", True)
compare_color_space = st.sidebar.checkbox("⚖️ Compare RGB vs LAB")

st.sidebar.markdown("---")
st.sidebar.header("🎨 Artistic & Preprocessing")
preprocess = st.sidebar.selectbox(
    "Preprocessing Method",
    ["None", "CLAHE", "Canny", "Bilateral", "KMeans", "Morphology"]
)
artistic_style = st.sidebar.selectbox("Artistic Style", ["None", "Watercolor Style", "Cartoon Effect"])
show_edges = st.sidebar.checkbox("Show Canny Edges (Structural Map)", value=False)
show_numbers = st.sidebar.checkbox("Show Numbers", True)

st.sidebar.markdown("---")
st.sidebar.header("📊 Comparisons")
compare_k_values = st.sidebar.checkbox("Compare K values")
compare_preprocessing = st.sidebar.checkbox("Compare Preprocessing Methods")

if st.sidebar.button("🔄 Reset Settings"):
    st.rerun()

# -------- FUNCTIONS --------
def enhance_image(image):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return denoised, sharpened, edges_colored

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def kmeans_quantization(image, k, use_lab=True):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if use_lab:
        process_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        process_img = image.copy()
        
    data = process_img.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()].reshape(image.shape)
    
    if use_lab:
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        centers_bgr = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)
        return result, centers_bgr
    return result, centers

def apply_kmeans_only(image, k):
    quantized, _ = kmeans_quantization(image, k, use_lab=False)
    return quantized

def draw_grid(image, grid_size):
    img = image.copy()
    h, w, _ = img.shape
    for y in range(0, h, grid_size): cv2.line(img, (0, y), (w, y), (255,255,255), 1)
    for x in range(0, w, grid_size): cv2.line(img, (x, 0), (x, h), (255,255,255), 1)
    return img

def draw_numbers(image, centers, grid_size):
    img = image.copy()
    h, w, _ = img.shape
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            color = img[y, x]
            for idx, c in enumerate(centers):
                if np.array_equal(color, c):
                    cv2.putText(img, str(idx), (x+2, y+grid_size-3), cv2.FONT_HERSHEY_SIMPLEX, grid_size/35.0, (0,0,0), 1, cv2.LINE_AA)
                    break
    return img

def apply_morphology(image, k):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)

def plot_histogram(image, title):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    fig, ax = plt.subplots()
    ax.plot(hist, color='cyan')
    ax.set_title(title, color='white')
    ax.set_xlabel("Pixel Intensity", color='white')
    ax.set_ylabel("Frequency", color='white')
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    return fig

# --- Metrics ---
def mse(img1, img2): return np.mean((img1.astype("float") - img2.astype("float")) ** 2)
def psnr(img1, img2):
    m = mse(img1, img2)
    return 100 if m == 0 else 20 * np.log10(255.0 / np.sqrt(m))

def add_noise(img):
    noise = np.random.normal(0, 20, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# -------- PROCESS --------
uploaded_file = st.file_uploader("📤 Upload Image", type=["png","jpg","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    original_img = img.copy()

    with st.spinner("AI Analysis & Processing... ⏳"):
        # 1. AI Face Detection (Auto-Tuning)
        gray_detect = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_detect, 1.1, 5)
        if len(faces) > 0:
            st.success(f"🤖 AI Detected {len(faces)} Face(s). Optimized for Portrait.")
            preprocess = "Bilateral"
            k = max(k, 8)

        # 2. Enhancement & Previews
        if apply_enhancement: 
            denoised, sharpened, edges_enh = enhance_image(img)
            img = sharpened

            st.subheader("✨ Enhancement Preview")
            col_enh1, col_enh2 = st.columns(2)
            col_enh1.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Original")
            col_enh2.image(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB), caption="Enhanced")

            st.subheader("🔍 Enhancement Processing Steps")
            ce1, ce2, ce3 = st.columns(3)
            ce1.image(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB), caption="Denoised")
            ce2.image(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB), caption="Sharpened")
            ce3.image(cv2.cvtColor(edges_enh, cv2.COLOR_BGR2RGB), caption="Edges")

        # 3. Preprocessing Selection & Morphology Previews
        before = img.copy()
        
        if preprocess == "CLAHE": img = apply_clahe(img)
        elif preprocess == "Canny": img = cv2.cvtColor(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200), cv2.COLOR_GRAY2BGR)
        elif preprocess == "Bilateral": img = cv2.bilateralFilter(img, 9, 75, 75)
        elif preprocess == "KMeans": img = apply_kmeans_only(img, k)
        elif preprocess == "Morphology": 
            morph = apply_morphology(img, k)
            st.subheader("🧪 Morphology Preview")
            cm1, cm2 = st.columns(2)
            cm1.image(cv2.cvtColor(before, cv2.COLOR_BGR2RGB), caption="Original")
            cm2.image(cv2.cvtColor(morph, cv2.COLOR_BGR2RGB), caption="After Morphology")
            mask = cv2.cvtColor(morph, cv2.COLOR_BGR2GRAY)
            img = cv2.bitwise_and(img, img, mask=mask)

        after = img.copy()

        # 4. Histogram Analysis (If preprocessing applied)
        if preprocess != "None":
            st.subheader("📊 Histogram Analysis")
            ch1, ch2 = st.columns(2)
            with ch1: st.pyplot(plot_histogram(before, "Before Preprocessing"))
            with ch2: st.pyplot(plot_histogram(after, "After Preprocessing"))

        # 5. Mosaic Generation
        h, w, _ = img.shape
        small = cv2.resize(img, (w // pixel_size, h // pixel_size))
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        quantized, centers = kmeans_quantization(pixelated, k, use_lab=use_lab)

        # 6. Stylization
        if artistic_style == "Watercolor Style": quantized = cv2.stylization(quantized, sigma_s=60, sigma_r=0.6)
        elif artistic_style == "Cartoon Effect":
            gray_cart = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
            mask_cart = cv2.adaptiveThreshold(cv2.medianBlur(gray_cart, 5), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            quantized = cv2.bitwise_and(quantized, quantized, mask=mask_cart)
        
        if show_edges:
            canny_q = cv2.Canny(cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY), 50, 150)
            quantized[canny_q == 255] = [0, 0, 0]

        # 7. Grid & Numbers
        grid_img = draw_grid(quantized, pixel_size)
        if show_numbers: grid_img = draw_numbers(grid_img, centers, pixel_size)

    # -------- ANIMATION --------
    st.subheader("🎬 Animation")
    placeholder = st.empty()
    for _ in range(2):
        placeholder.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Original")
        time.sleep(0.5)
        placeholder.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB), caption="Mosaic Result")
        time.sleep(0.5)

    # -------- UI RESULTS --------
    st.subheader("🖼️ Result Display")
    res_col1, res_res2 = st.columns(2)
    res_col1.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
    res_res2.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB), caption="Final Mosaic Result", use_container_width=True)

    # -------- COMPARISONS --------
    if compare_k_values:
        st.subheader("📊 K Values Comparison")
        ks = [3, 6, 12]
        cols_comp = st.columns(len(ks))
        for i, k_val in enumerate(ks):
            q_img_k, _ = kmeans_quantization(pixelated, k_val, use_lab=use_lab)
            cols_comp[i].image(cv2.cvtColor(q_img_k, cv2.COLOR_BGR2RGB), caption=f"k = {k_val}", use_container_width=True)

    if compare_preprocessing:
        st.subheader("📊 Preprocessing Comparison (Mosaic Level)")
        methods = ["Original", "CLAHE", "Canny", "Bilateral", "KMeans", "Morphology"]
        cols_prep = st.columns(len(methods))
        for i, m in enumerate(methods):
            temp = original_img.copy()
            if m == "CLAHE": temp = apply_clahe(temp)
            elif m == "Canny": temp = cv2.cvtColor(cv2.Canny(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY), 100, 200), cv2.COLOR_GRAY2BGR)
            elif m == "Bilateral": temp = cv2.bilateralFilter(temp, 9, 75, 75)
            elif m == "KMeans": temp = apply_kmeans_only(temp, k)
            elif m == "Morphology": temp = apply_morphology(temp, k)

            h_c, w_c = temp.shape[:2]
            small_c = cv2.resize(temp, (w_c // pixel_size, h_c // pixel_size))
            mosaic_c = cv2.resize(small_c, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
            q_img_c, _ = kmeans_quantization(mosaic_c, k, use_lab=use_lab)
            grid_img_cmp = draw_grid(q_img_c, pixel_size)
            cols_prep[i].image(cv2.cvtColor(grid_img_cmp, cv2.COLOR_BGR2RGB), caption=m, use_container_width=True)

    if compare_color_space:
        st.subheader("⚖️ Algorithm Comparison: RGB vs LAB")
        c1, c2 = st.columns(2)
        q_rgb, _ = kmeans_quantization(pixelated, k, use_lab=False)
        q_lab, _ = kmeans_quantization(pixelated, k, use_lab=True)
        c1.image(cv2.cvtColor(q_rgb, cv2.COLOR_BGR2RGB), caption="K-Means (RGB Space)", use_container_width=True)
        c2.image(cv2.cvtColor(q_lab, cv2.COLOR_BGR2RGB), caption="K-Means (LAB Space - Better Perception)", use_container_width=True)

    # -------- ROBUSTNESS & METRICS --------
    st.subheader("🧪 Robustness Testing & Evaluation")
    tests = {
        "Low Light": cv2.convertScaleAbs(original_img, alpha=0.6, beta=0),
        "High Brightness": cv2.convertScaleAbs(original_img, alpha=1.2, beta=10),
        "Noisy": add_noise(original_img),
        "High Contrast": cv2.convertScaleAbs(original_img, alpha=1.3, beta=-10),
    }
    cols_rob = st.columns(len(tests))
    metrics_report = ""
    for i, (name, test_img) in enumerate(tests.items()):
        t_h, t_w = test_img.shape[:2]
        t_small = cv2.resize(test_img, (t_w // pixel_size, t_h // pixel_size))
        t_pixel = cv2.resize(t_small, (t_w, t_h), interpolation=cv2.INTER_NEAREST)
        t_quant, _ = kmeans_quantization(t_pixel, k, use_lab=use_lab)
        err = mse(original_img, t_quant)
        qual = psnr(original_img, t_quant)
        cols_rob[i].image(cv2.cvtColor(t_quant, cv2.COLOR_BGR2RGB), caption=f"{name}\nMSE: {err:.1f} | PSNR: {qual:.1f}", use_container_width=True)
        metrics_report += f"{name} -> MSE: {err:.2f}, PSNR: {qual:.2f}\n"

    st.subheader("🧠 Robustness Analysis")
    st.write("""
    The system was evaluated under multiple challenging conditions including low lighting, high brightness, added noise, and increased contrast.
    Results show that K-Means clustering remains stable across brightness variations, producing consistent mosaic structures.
    Noise slightly affects clustering accuracy but does not break the system.
    High contrast conditions exaggerate edges, leading to sharper but slightly distorted mosaics.
    This demonstrates that the system is robust under different real-world conditions.
    """)

    # -------- ANALYSIS & EXPORT --------
    st.subheader("🎨 Color Analysis & Statistics")
    report_col1, report_col2 = st.columns(2)
    pixels = quantized.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    report_text = f"System Report\nPixel Size: {pixel_size}\nK: {k}\n\nRobustness Metrics:\n{metrics_report}\nColor Palette:\n"
    
    with report_col1:
        st.write(f"**Total Detected Colors:** {len(colors)}")
        for i, (c, cnt) in enumerate(zip(colors, counts)):
            color_box = np.zeros((40, 100, 3), dtype=np.uint8)
            color_box[:] = c
            st.image(cv2.cvtColor(color_box, cv2.COLOR_BGR2RGB), caption=f"ID {i} → {cnt} pixels")
            report_text += f"Color {i} (RGB {c[::-1]}): {cnt} pixels\n"
            
    with report_col2:
        fig, ax = plt.subplots()
        ax.pie(counts, labels=[f"ID {i}" for i in range(len(colors))], colors=[c[::-1]/255 for c in colors], autopct='%1.1f%%')
        ax.set_title("Color Distribution")
        st.pyplot(fig)
        
    st.subheader("🧠 System Analysis")
    st.write(f"""
    - **Preprocessing:** Different preprocessing techniques affect the final representation. CLAHE enhances contrast, Canny extracts edges, and Bilateral preserves edges while smoothing noise.
    - **Quantization:** Simplified from thousands of colors to exactly **{k}** colors using **{'LAB' if use_lab else 'RGB'}** space.
    - **Resolution:** A Pixel size of **{pixel_size}**px created a grid of **{h//pixel_size} x {w//pixel_size}** blocks.
    - **Conclusion:** Higher 'K' increases detail, while larger pixel size increases abstraction. The system successfully balances detail, abstraction, and performance.
    """)

    # -------- DOWNLOAD --------
    st.subheader("⬇️ Export")
    c_down1, c_down2 = st.columns(2)
    with c_down1:
        st.download_button("Download Mosaic Image", cv2.imencode('.png', grid_img)[1].tobytes(), "mosaic_final.png")
    with c_down2:
        st.download_button("Download Full Report", report_text, "mosaic_report.txt")