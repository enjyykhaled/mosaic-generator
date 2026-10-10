# 🎨 Smart Mosaic Generation & AI-Driven Stylization System

**Course:** A1306 – Image Processing

A content-aware pipeline that turns images into "paint-by-numbers" mosaics.
It detects faces (Haar Cascades) and automatically adapts preprocessing and
the number of colors (k) to preserve facial details.

## ✨ Features
- AI auto-tuning: face detection → edge-preserving smoothing + higher k
- Bilateral filtering for noise removal while keeping edges
- K-Means color quantization in **LAB** color space (compared with RGB)
- Numbered palette + color distribution statistics
- Preprocessing options: CLAHE, Canny, Bilateral, KMeans, Morphology
- Artistic styles: Watercolor, Cartoon
- Robustness tests (MSE / PSNR): low light, high brightness, high contrast, noise
- Interactive Streamlit UI

## 🚀 How to Run

```bash
git clone https://github.com/enjyykhaled/mosaic-generator.git
cd mosaic-generator
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Project Structure
| File | Description |
|------|-------------|
| `app.py` | Streamlit web app |
| `mosaic.ipynb` | Full experiments notebook (pipeline, comparisons, robustness tests) |
| `smart_model.pkl` | Model generated in the notebook |
| `Final_Project_Report.pdf` | Full project report |
| `requirements.txt` | Python dependencies |

## 📊 Robustness Results
| Scenario | MSE | PSNR (dB) |
|----------|-----|-----------|
| Low Light | 3967.1 | 12.1 |
| High Brightness | 1552.1 | 16.2 |
| Noisy | 399.7 | 22.1 |
| High Contrast | 1259.3 | 17.1 |

## 🛠️ Tech Stack
Python, OpenCV, NumPy, scikit-learn, TensorFlow, Matplotlib, Streamlit
