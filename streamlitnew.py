import os
import io
import tempfile
import numpy as np
import nibabel as nib
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from sklearn.cluster import DBSCAN

# ================================
# 🎯 Custom Layer: SpatialPyramidPooling3D
# ================================
class SpatialPyramidPooling3D(tf.keras.layers.Layer):
    def __init__(self, pool_list=[1, 2, 4], **kwargs):
        super(SpatialPyramidPooling3D, self).__init__(**kwargs)
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i**3 for i in pool_list])

    def build(self, input_shape):
        self.nb_channels = input_shape[-1]
        super(SpatialPyramidPooling3D, self).build(input_shape)

    def call(self, x):
        input_shape = tf.shape(x)
        outputs = []
        for pool_size in self.pool_list:
            h, w, d = input_shape[1], input_shape[2], input_shape[3]
            ph, pw, pd = tf.maximum(h // pool_size, 1), tf.maximum(w // pool_size, 1), tf.maximum(d // pool_size, 1)
            for i in range(pool_size):
                for j in range(pool_size):
                    for k in range(pool_size):
                        hs, he = i * ph, tf.minimum((i + 1) * ph, h)
                        ws, we = j * pw, tf.minimum((j + 1) * pw, w)
                        ds, de = k * pd, tf.minimum((k + 1) * pd, d)
                        region = x[:, hs:he, ws:we, ds:de, :]
                        outputs.append(tf.reduce_max(region, axis=[1, 2, 3]))
        return tf.concat(outputs, axis=-1)

# Register custom layer globally (brute force fix)
tf.keras.utils.get_custom_objects()["SpatialPyramidPooling3D"] = SpatialPyramidPooling3D

# ================================
# 🚀 Model Loader
# ================================
def load_model_local(path, custom=False):
    if custom:
        with tf.keras.utils.custom_object_scope({'SpatialPyramidPooling3D': SpatialPyramidPooling3D}):
            return tf.keras.models.load_model(path)
    return tf.keras.models.load_model(path)

# ================================
# 📍 Path ke Model Lokal
# ================================
FCN_MODEL_PATH = "fcn_precision_focused_best.h5"
CNN_MODEL_PATH = "stage2_cnn_final.h5"

# ================================
# ⚙️ Konfigurasi
# ================================
FCN_PATCH_SHAPE = (16, 16, 10)
FCN_STRIDE = (4, 4, 2)
FCN_BATCH_SIZE = 1024
FCN_THRESHOLD = 0.85
FCN_MIN_CLUSTER_SIZE = 3

CNN_PATCH_SHAPE = (20, 20, 16)
CNN_PATCH_X, CNN_PATCH_Y, CNN_PATCH_Z = 10, 10, 8
CNN_THRESHOLD = 0.5

# ================================
# 🧠 Pipeline
# ================================
def create_brain_mask(volume):
    mask = binary_fill_holes(volume > 0.1)
    mask = binary_erosion(mask, iterations=2)
    mask = binary_dilation(mask, iterations=4)
    return mask

def fcn_inference(model, volume):
    H, W, D = volume.shape
    pH, pW, pD = FCN_PATCH_SHAPE
    sH, sW, sD = FCN_STRIDE
    score_map = np.zeros_like(volume, dtype=np.float32)
    count_map = np.zeros_like(volume, dtype=np.float32)

    patch_coords = [(x, y, z) for x in range(0, H-pH+1, sH)
                                 for y in range(0, W-pW+1, sW)
                                 for z in range(0, D-pD+1, sD)]

    for i in range(0, len(patch_coords), FCN_BATCH_SIZE):
        batch = patch_coords[i:i+FCN_BATCH_SIZE]
        patches = np.array([volume[x:x+pH, y:y+pW, z:z+pD] for x, y, z in batch])[..., np.newaxis]
        preds = model.predict(patches, verbose=0)
        for j, (x, y, z) in enumerate(batch):
            prob = preds[j, 1]
            cx, cy, cz = x + pH//2, y + pW//2, z + pD//2
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        ax, ay, az = cx+dx, cy+dy, cz+dz
                        if 0 <= ax < H and 0 <= ay < W and 0 <= az < D:
                            weight = 1.0 if (dx, dy, dz) == (0, 0, 0) else 0.3
                            score_map[ax, ay, az] += prob * weight
                            count_map[ax, ay, az] += weight

    return np.divide(score_map, count_map, out=np.zeros_like(score_map), where=count_map>0)

def fcn_clustering(score_map):
    binary_map = score_map > FCN_THRESHOLD
    coords = np.column_stack(np.where(binary_map))
    if len(coords) == 0:
        return []
    clustering = DBSCAN(eps=3, min_samples=FCN_MIN_CLUSTER_SIZE).fit(coords)
    results = []
    for label in set(clustering.labels_):
        if label == -1:
            continue
        mask = clustering.labels_ == label
        cluster_coords = coords[mask]
        cluster_scores = score_map[tuple(cluster_coords.T)]
        center = np.average(cluster_coords, axis=0, weights=cluster_scores)
        results.append({"coordinate": [int(round(c)) for c in center], "score": float(np.max(cluster_scores))})
    return sorted(results, key=lambda x: x["score"], reverse=True)

def extract_cnn_patches(volume, candidates):
    patches, valid = [], []
    for c in candidates:
        cx, cy, cz = c["coordinate"]
        x0, x1 = cx - CNN_PATCH_X, cx + CNN_PATCH_X
        y0, y1 = cy - CNN_PATCH_Y, cy + CNN_PATCH_Y
        z0, z1 = cz - CNN_PATCH_Z, cz + CNN_PATCH_Z
        if x0 >= 0 and x1 <= volume.shape[0] and y0 >= 0 and y1 <= volume.shape[1] and z0 >= 0 and z1 <= volume.shape[2]:
            patch = volume[x0:x1, y0:y1, z0:z1]
            if patch.shape == CNN_PATCH_SHAPE:
                patches.append(patch)
                valid.append(c)
    return np.array(patches)[..., np.newaxis], valid

def cnn_inference(model, patches, candidates):
    if len(patches) == 0:
        return []
    preds = model.predict(patches, batch_size=32, verbose=0)
    results = []
    for i, cand in enumerate(candidates):
        score = preds[i, 1]
        if score >= CNN_THRESHOLD:
            cand["cnn_score"] = float(score)
            results.append(cand)
    return results

def get_detection_slices(detections):
    """Mendapatkan list slice yang memiliki deteksi"""
    if not detections:
        return []
    slices = sorted(list(set([d["coordinate"][2] for d in detections])))
    return slices

def plot_slice(volume, detections, slice_idx):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(np.rot90(volume[:, :, slice_idx]), cmap="gray")
    
    # Hitung jumlah deteksi di slice ini
    detections_in_slice = []
    for i, d in enumerate(detections):
        cx, cy, cz = d["coordinate"]
        if cz == slice_idx:
            detections_in_slice.append(d)
            # Plot deteksi dengan label nomor
            ax.plot(cy, cx, "ro", markersize=8)
            ax.annotate(f'{i+1}', (cy, cx), xytext=(5, 5), 
                       textcoords='offset points', color='yellow', 
                       fontweight='bold', fontsize=12)
    
    ax.set_title(f"Slice {slice_idx} - {len(detections_in_slice)} CMB detected")
    ax.axis("off")
    return fig

# ================================
# 🧠 Streamlit UI
# ================================
st.set_page_config(page_title="CMB Detection", layout="wide")
st.title("🧠 CMB Detection on MRI")

uploaded_file = st.file_uploader("📤 Upload NIfTI file (.nii.gz)", type=["nii.gz"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("📂 Loading and normalizing NIfTI file..."):
            nii = nib.load(tmp_path)
            volume = nii.get_fdata().astype(np.float32)
            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
            st.success(f"✅ Data loaded: {volume.shape}")
    finally:
        os.remove(tmp_path)

    # Run full inference pipeline with spinner
    with st.spinner("💡 Running model inference, please wait..."):
        fcn_model = load_model_local(FCN_MODEL_PATH)
        cnn_model = load_model_local(CNN_MODEL_PATH, custom=True)

        brain_mask = create_brain_mask(volume)
        fcn_map = fcn_inference(fcn_model, volume * brain_mask)
        candidates = fcn_clustering(fcn_map)
        X, valid = extract_cnn_patches(volume, candidates)
        detections = cnn_inference(cnn_model, X, valid)
    st.success("🎉 Inference finished!")

    st.metric("🧠 Jumlah Deteksi", len(detections))

    # Initialize session state for slice index
    if 'current_slice' not in st.session_state:
        st.session_state.current_slice = volume.shape[2] // 2

    # Navigation if detections exist
    if detections:
        st.subheader("🎯 Navigate to Detection Results")
        detection_slices = get_detection_slices(detections)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🏠 First Detection"):
                st.session_state.current_slice = detection_slices[0]
        with col2:
            if st.button("⬅️ Previous Detection"):
                current = st.session_state.current_slice
                prev_slices = [s for s in detection_slices if s < current]
                if prev_slices:
                    st.session_state.current_slice = max(prev_slices)
                else:
                    st.warning("Sudah di deteksi pertama")
        with col3:
            if st.button("➡️ Next Detection"):
                current = st.session_state.current_slice
                next_slices = [s for s in detection_slices if s > current]
                if next_slices:
                    st.session_state.current_slice = min(next_slices)
                else:
                    st.warning("Sudah di deteksi terakhir")
        with col4:
            if st.button("🎯 Last Detection"):
                st.session_state.current_slice = detection_slices[-1]

        # Selectbox untuk jump ke deteksi tertentu
        st.subheader("📋 Jump to Specific Detection")
        detection_options = []
        for i, d in enumerate(detections):
            cx, cy, cz = d["coordinate"]
            score = d.get("cnn_score", d.get("score", 0))
            detection_options.append(f"Detection {i+1}: Slice {cz} (x:{cx}, y:{cy}) - Score: {score:.3f}")

        selected_detection = st.selectbox("Select detection:", detection_options)
        if selected_detection and st.button("🚀 Jump to Selected"):
            slice_num = int(selected_detection.split("Slice ")[1].split(" ")[0])
            st.session_state.current_slice = slice_num

        st.info(f"📊 Slices with detections: {', '.join(map(str, detection_slices))}")

    # Manual navigation
    st.subheader("🔍 Manual Slice Navigation")
    idx = st.slider("Slice Index", 0, volume.shape[2] - 1,
                    value=st.session_state.current_slice, key="slice_slider")
    st.session_state.current_slice = idx

    # Show slice
    fig = plot_slice(volume, detections, idx)
    st.pyplot(fig)

    # Detection info
    current_detections = [d for d in detections if d["coordinate"][2] == idx]
    if current_detections:
        st.success(f"🎯 {len(current_detections)} CMB detected in this slice!")
        for i, d in enumerate(current_detections):
            cx, cy, cz = d["coordinate"]
            score = d.get("cnn_score", d.get("score", 0))
            st.write(f"Detection {detections.index(d)+1}: ({cx}, {cy}, {cz}) - Score: {score:.3f}")
    else:
        st.info("ℹ️ No detections in this slice.")
