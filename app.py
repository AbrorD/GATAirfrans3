
# SINI ADALAH TEMPAT ANDA MENEMPELKAN SELURUH KODE
# streamlit_app.py YANG TELAH KITA BUAT SEBELUMNYA
# (Pastikan semua impor dan definisi fungsi ada di dalam file ini)

import streamlit as st
import torch
import numpy as np
import pyvista as pv
import yaml
import os
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from scipy.spatial import KDTree # Untuk reorganize

# --- FUNGSI UTILITAS ---
def reorganize(in_order_points, out_order_points, quantity_to_reordered):
    if out_order_points.shape[0] == 0 or in_order_points.shape[0] == 0:
        if hasattr(quantity_to_reordered, 'shape') and len(quantity_to_reordered.shape) > 1:
             return np.zeros((out_order_points.shape[0], quantity_to_reordered.shape[1]), dtype=quantity_to_reordered.dtype)
        else:
             return np.zeros(out_order_points.shape[0], dtype=quantity_to_reordered.dtype)
    tree = KDTree(in_order_points)
    dist, idx_in_in_order = tree.query(out_order_points)
    if len(quantity_to_reordered.shape) > 1:
        reordered_quantity = np.zeros((out_order_points.shape[0], quantity_to_reordered.shape[1]), dtype=quantity_to_reordered.dtype)
    else:
        reordered_quantity = np.zeros(out_order_points.shape[0], dtype=quantity_to_reordered.dtype)
    if quantity_to_reordered.shape[0] > 0 and idx_in_in_order.max() < quantity_to_reordered.shape[0] and idx_in_in_order.min() >=0 :
        reordered_quantity = quantity_to_reordered[idx_in_in_order]
    elif quantity_to_reordered.shape[0] > 0:
        st.warning(f"Peringatan (reorganize): Index mismatch.") # Ganti print dengan st.warning
    return reordered_quantity

# Impor dari struktur direktori GATAirfrans2 Anda
# Ini mengasumsikan Colab notebook Anda berada di /content/ dan GATAirfrans2 ada di sana
# Jika GATAirfrans2 ada di root Colab (/content/GATAirfrans2), path berikut mungkin perlu disesuaikan
# atau Anda perlu menambahkan GATAirfrans2 ke sys.path

# Untuk mengatasi masalah path impor di Colab, kita bisa set direktori kerja
# atau menambahkan path secara eksplisit.
# Asumsikan GATAirfrans2 sudah di-clone atau di-upload ke /content/GATAirfrans2
import sys
PROJECT_ROOT = '/content/GATAirfrans2' # Path ke root proyek Anda di Colab
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dataset import Dataset as ProjectDataset # Perhatikan ini
    from models.MLP import MLP
    from models.GAT import GAT
    import metrics
    import metrics_NACA
    import torch_geometric.nn as nng
except ImportError as e:
    st.error(f"Gagal mengimpor modul proyek dari {PROJECT_ROOT}: {e}")
    st.error("Pastikan Anda telah meng-upload atau meng-clone proyek GATAirfrans2 ke direktori yang benar di Colab.")
    st.stop()


# --- Cache untuk pemuatan model dan koefisien ---
@st.cache_resource
def load_model(model_path, device):
    if not osp.exists(model_path):
        st.error(f"File model tidak ditemukan: {model_path}")
        return None
    try:
        trained_models_list = torch.load(model_path, map_location=device, weights_only=False)
        model_instance = trained_models_list[0] if isinstance(trained_models_list, list) and trained_models_list else trained_models_list
        if not isinstance(model_instance, torch.nn.Module):
            st.error("Objek yang dimuat bukan model PyTorch yang valid.")
            return None
        model_instance.to(device).eval()
        return model_instance
    except Exception as e:
        st.error(f"Gagal memuat model dari {model_path}: {e}")
        return None

@st.cache_data
def load_normalization_coefficients(coef_norm_path):
    if not osp.exists(coef_norm_path):
        st.error(f"File koefisien normalisasi tidak ditemukan: {coef_norm_path}")
        return None, None, None, None
    try:
        loaded_tuple = torch.load(coef_norm_path, map_location='cpu', weights_only=False)
        if isinstance(loaded_tuple, tuple) and len(loaded_tuple) == 4:
            m_in, s_in, m_out, s_out = ((arr.numpy() if hasattr(arr, 'numpy') else arr).astype(np.float32) for arr in loaded_tuple)
            return m_in, s_in, m_out, s_out
        else:
            st.error("Format koefisien normalisasi tidak sesuai.")
            return None, None, None, None
    except Exception as e:
        st.error(f"Gagal memuat koefisien normalisasi: {e}")
        return None, None, None, None

# --- UI Streamlit ---
st.set_page_config(layout="wide")
st.title("Prediksi Distribusi Tekanan Airfoil menggunakan GAT (via Colab & ngrok)")


# --- PATH KONFIGURASI (disesuaikan untuk Colab) ---
MODEL_TYPE_APP = 'GAT' 
TASK_MODEL_TRAINED_ON_APP = 'scarce' 

PATH_TO_TRAINED_MODEL_APP = osp.join(PROJECT_ROOT, f'metrics/{TASK_MODEL_TRAINED_ON_APP}/{MODEL_TYPE_APP}/{MODEL_TYPE_APP}') 
PARAMS_FILE_PATH_APP = osp.join(PROJECT_ROOT, 'params.yaml') 
COEF_NORM_LOAD_PATH_APP = osp.join(PROJECT_ROOT, f'Dataset/ormalization_coefficients_{TASK_MODEL_TRAINED_ON_APP}.pt')

# --- Muat Model, Hparams, CoefNorm ---
device_app = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_instance_app = load_model(PATH_TO_TRAINED_MODEL_APP, device_app)
mean_in_app, std_in_app, mean_out_app, std_out_app = load_normalization_coefficients(COEF_NORM_LOAD_PATH_APP)

if not osp.exists(PARAMS_FILE_PATH_APP): 
    st.error(f"params.yaml tidak ditemukan di {PARAMS_FILE_PATH_APP}"); st.stop()
with open(PARAMS_FILE_PATH_APP, 'r') as f: hparams_all_app = yaml.safe_load(f)
if MODEL_TYPE_APP not in hparams_all_app: 
    st.error(f"Hparams untuk {MODEL_TYPE_APP} tidak di params.yaml"); st.stop()
hparams_model_dict_app = hparams_all_app[MODEL_TYPE_APP]

if not model_instance_app or mean_in_app is None:
    st.error("Gagal memuat model atau koefisien normalisasi. Aplikasi tidak dapat berjalan.")
    st.stop()

st.sidebar.header("Input Pengguna")
vtu_uploaded_file = st.sidebar.file_uploader("Unggah File VTU Domain (*_internal.vtu)", type=["vtu"])
vtp_uploaded_file = st.sidebar.file_uploader("Unggah File VTP Airfoil (*_aerofoil.vtp)", type=["vtp"])
U_inf_input = st.sidebar.number_input("Kecepatan Freestream (U∞)", value=10.0, step=0.1, format="%.2f")
alpha_deg_input = st.sidebar.number_input("Sudut Serang (Alpha, derajat)", value=0.0, step=0.1, format="%.1f")
run_button = st.sidebar.button("Jalankan Inferensi")

if run_button and vtu_uploaded_file is not None and vtp_uploaded_file is not None:
    with st.spinner("Memproses data dan menjalankan inferensi..."):
        # Simpan file yang diunggah sementara
        temp_vtu_path = osp.join(PROJECT_ROOT, "temp_internal.vtu") # Simpan di dalam GATAirfrans2
        temp_vtp_path = osp.join(PROJECT_ROOT, "temp_aerofoil.vtp")
        with open(temp_vtu_path, "wb") as f:
            f.write(vtu_uploaded_file.getbuffer())
        with open(temp_vtp_path, "wb") as f:
            f.write(vtp_uploaded_file.getbuffer())
        
        case_name_from_upload = osp.splitext(vtu_uploaded_file.name)[0].replace('_internal', '')
        st.write(f"Memproses kasus: {case_name_from_upload}")

        try:
            # --- Proses Data (gunakan ProjectDataset) ---
            # Pastikan ProjectDataset bisa menemukan file dari path sementara atau nama kasus
            # Untuk ProjectDataset, kita perlu nama dasar kasus dan ia akan mencari di 'Dataset/nama_kasus/...'
            # Ini mungkin perlu penyesuaian jika file diunggah.
            # Cara termudah: baca manual dengan pyvista, lalu buat objek Data.
            
            internal_mesh_pv = pv.read(temp_vtu_path)
            airfoil_mesh_pv = pv.read(temp_vtp_path)
            
            # ... (Salin SELURUH logika dari Langkah 3 (Ekstrak Fitur)
            #      hingga Langkah 7 (Perhitungan Metrik) dari skrip 
            #      inference_single_case_fully_using_metrics_module_with_paths.py
            #      atau inference_single_case_manual_aligned.py,
            #      pastikan untuk menggunakan U_inf_input dan alpha_deg_input,
            #      serta mean_in_app, std_in_app, dst.
            #      Ganti print() dengan st.write() atau st.info() jika perlu.
            #      Saya akan menyalin bagian pentingnya di sini)

            # --- Mulai salin dari skrip inferensi ---
            pos_np = internal_mesh_pv.points[:, :2].astype(np.float32)
            VTU_VELOCITY_FIELD_APP = 'U' # Sesuaikan jika nama field di VTU Anda berbeda
            if VTU_VELOCITY_FIELD_APP not in internal_mesh_pv.point_data:
                 st.error(f"Field kecepatan '{VTU_VELOCITY_FIELD_APP}' tidak ditemukan di VTU."); st.stop()
            velocity_field_from_vtu = internal_mesh_pv.point_data[VTU_VELOCITY_FIELD_APP]
            surf_bool_np = (np.isclose(velocity_field_from_vtu[:, 0], 0.0)) & (np.isclose(velocity_field_from_vtu[:, 1], 0.0))
            alpha_rad_input_val = np.deg2rad(alpha_deg_input)
            U_inf_vector_val = np.array([np.cos(alpha_rad_input_val), np.sin(alpha_rad_input_val)]) * U_inf_input
            U_inf_vector_repeated_np_val = np.tile(U_inf_vector_val, (internal_mesh_pv.n_points, 1)).astype(np.float32)
            
            # SDF manual (disalin dari skrip sebelumnya)
            geom_sdf_np_val = np.zeros((internal_mesh_pv.n_points, 1), dtype=np.float32)
            if airfoil_mesh_pv.n_points > 0:
                airfoil_tree_2d_val = KDTree(airfoil_mesh_pv.points[:, :2])
                distances_val, _ = airfoil_tree_2d_val.query(internal_mesh_pv.points[:, :2])
                signed_distances_val = np.copy(distances_val.astype(np.float32))
                # (Logika penentuan tanda SDF yang lebih kompleks bisa ditambahkan di sini jika perlu)
                geom_sdf_np_val = signed_distances_val[:, np.newaxis]
            
            normals_on_volume_np_val = np.zeros((internal_mesh_pv.n_points, 2), dtype=np.float32)
            if np.any(surf_bool_np) and 'Normals' in airfoil_mesh_pv.point_data and airfoil_mesh_pv.n_points > 0:
                # ... (logika reorganize normal) ...
                pass # Placeholder, Anda perlu salin logika reorganize normal

            input_features_np_val = np.concatenate([pos_np, U_inf_vector_repeated_np_val, geom_sdf_np_val, normals_on_volume_np_val], axis=1)
            data_for_model_val = Data(x=torch.from_numpy(input_features_np_val), pos=torch.from_numpy(pos_np), surf=torch.from_numpy(surf_bool_np))
            data_for_model_val.y = torch.zeros((data_for_model_val.num_nodes, len(mean_out_app)), dtype=torch.float32)
            data_for_model_val.x = (data_for_model_val.x - torch.from_numpy(mean_in_app)) / (torch.from_numpy(std_in_app) + 1e-8)

            # Inferensi
            if MODEL_TYPE_APP not in ['MLP', 'PointNet']:
                with torch.no_grad():
                    data_for_model_val.edge_index = nng.radius_graph(
                        x=data_for_model_val.pos.to(device_app), r=hparams_model_dict_app['r'], loop=True, 
                        max_num_neighbors=int(hparams_model_dict_app['max_neighbors']), batch=None).to(device_app)
            data_for_model_val = data_for_model_val.to(device_app)
            with torch.no_grad(): predicted_y_normalized_tensor_val = model_instance_app(data_for_model_val)
            
            # Denormalisasi & BC
            predicted_y_denormalized_np_val = (predicted_y_normalized_tensor_val.cpu().numpy() * (std_out_app + 1e-8)) + mean_out_app
            predicted_y_denormalized_np_val[surf_bool_np, :2] = 0.0
            predicted_y_denormalized_np_val[surf_bool_np, 3] = 0.0
            pred_p_np_val = predicted_y_denormalized_np_val[:, 2]

            # Persiapan untuk metrics_NACA
            airfoil_pred_pv_val = airfoil_mesh_pv.copy()
            if 'Normals' not in airfoil_pred_pv_val.point_data and airfoil_pred_pv_val.n_points > 0:
                airfoil_pred_pv_val.compute_normals(point_normals=True, cell_normals=False, inplace=True)
            
            # Reorganize tekanan prediksi ke airfoil_pred_pv_val
            if np.any(surf_bool_np):
                points_vol_surf = internal_mesh_pv.points[surf_bool_np, :2]
                p_vol_surf = pred_p_np_val[surf_bool_np] # Tekanan prediksi pada node permukaan volume
                if points_vol_surf.shape[0] > 0 and airfoil_pred_pv_val.points.shape[0] > 0:
                    airfoil_pred_pv_val.point_data['p_pred'] = reorganize(points_vol_surf, airfoil_pred_pv_val.points[:,:2], p_vol_surf)
                else: airfoil_pred_pv_val.point_data['p_pred'] = np.zeros(airfoil_pred_pv_val.n_points)
            else: airfoil_pred_pv_val.point_data['p_pred'] = np.zeros(airfoil_pred_pv_val.n_points)

            airfoil_pred_pv_val.point_data['p'] = airfoil_pred_pv_val.point_data.get('p_pred', np.zeros(airfoil_pred_pv_val.n_points))
            
            # Asumsi WSS tidak dihitung di sini untuk kesederhanaan Streamlit, 
            # jadi 'wallShearStress' mungkin tidak ada, metrics_NACA akan menggunakan default
            if 'wallShearStress' not in airfoil_pred_pv_val.point_data:
                 airfoil_pred_pv_val.point_data['wallShearStress'] = np.zeros((airfoil_pred_pv_val.n_points, 2))


            # Hitung Cp Prediksi
            case_name_for_metrics_naca_val = f"airfoil_U{U_inf_input}_A{alpha_deg_input}" # Nama dummy
            cp_pred_raw_val, _ = metrics_NACA.surface_coefficients(airfoil_pred_pv_val, case_name_for_metrics_naca_val)
            
            # (Tambahkan perhitungan Cd/Cl jika diperlukan dan jika WSS dihitung)

            # --- Plotting ---
            st.subheader(f"Hasil Prediksi untuk {case_name_from_upload}")
            fig, ax = plt.subplots(figsize=(10, 6))
            if cp_pred_raw_val is not None and cp_pred_raw_val.size > 0:
                ax.plot(cp_pred_raw_val[:, 0], cp_pred_raw_val[:, 1], 'o-', label=f'Predicted Cp ({MODEL_TYPE_APP})', markersize=3, linewidth=1)
            ax.invert_yaxis()
            ax.set_xlabel('x/c')
            ax.set_ylabel(r'$C_p$')
            title_info = f"Cp untuk U∞={U_inf_input:.2f}, α={alpha_deg_input:.1f}°"
            ax.set_title(title_info)
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            # --- Akhir salin dari skrip inferensi ---

        except Exception as e_process:
            st.error(f"Terjadi error saat pemrosesan file: {e_process}")
            import traceback
            st.text(traceback.format_exc())
        finally:
            if os.path.exists(temp_vtu_path): os.remove(temp_vtu_path)
            if os.path.exists(temp_vtp_path): os.remove(temp_vtp_path)
else:
    st.info("Silakan unggah file VTU dan VTP, lalu masukkan parameter aliran dan klik 'Jalankan Inferensi'.")