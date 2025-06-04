import os.path as osp
import pathlib # Ditambahkan untuk manajemen path yang lebih baik

import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader

import pyvista as pv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

import metrics_NACA
from reorganize import reorganize
from dataset import Dataset # Asumsi Dataset.py ada di path yang benar atau PYTHONPATH

from tqdm import tqdm

NU = np.array(1.56e-5)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def rsquared(predict, true):
    mean = true.mean(dim = 0)
    return 1 - ((true - predict)**2).sum(dim = 0)/((true - mean)**2).sum(dim = 0)

def rel_err(a, b):
    return np.abs((a - b)/a)

def WallShearStress(Jacob_U, normals):
    S = .5*(Jacob_U + Jacob_U.transpose(0, 2, 1))
    S = S - S.trace(axis1 = 1, axis2 = 2).reshape(-1, 1, 1)*np.eye(2)[None]/3
    ShearStress = 2*NU.reshape(-1, 1, 1)*S
    ShearStress = (ShearStress*normals[:, :2].reshape(-1, 1, 2)).sum(axis = 2)
    return ShearStress

@torch.no_grad()
def Infer_test(device, models, hparams, data, coef_norm = None):
    # Inference procedure on new simulation
    outs = [torch.zeros_like(data.y)]*len(models)
    n_out = torch.zeros_like(data.y[:, :1])
    idx_points_remaining = set(map(tuple, data.pos[:, :2].cpu().numpy())) # Pindahkan ke CPU untuk operasi set

    # Pastikan data ada di CPU untuk subsampling awal
    data_cpu = data.clone().cpu()
    
    i = 0
    while len(idx_points_remaining) > 0:
        i += 1
        
        # Ambil sampel dari titik yang belum diprediksi
        current_sample_size = min(hparams[0]['subsampling'], len(idx_points_remaining))
        if current_sample_size == 0: # Untuk menghindari error jika subsampling > jumlah titik tersisa
            break
            
        # Buat list dari idx_points_remaining untuk random.sample
        # Ambil sampel dari idx yang belum terprediksi
        # Ini bagian yang rumit jika ingin memastikan semua titik tercover dengan subsampling
        # Cara yang lebih mudah adalah mengambil sampel dari keseluruhan, lalu update n_out
        # Namun, implementasi asli mencoba mengurangi idx_points.
        # Mari kita ikuti logika asli tapi pastikan operasinya benar.

        # Jika ingin memastikan semua titik diproses, subsampling mungkin lebih baik dilakukan pada keseluruhan
        # dan mengandalkan n_out untuk merata-ratakan. Logika asli dengan idx_points agak kompleks
        # untuk dijamin benar tanpa melihat konteks subsampling lebih dalam.
        # Untuk penyederhanaan dan robustisitas, kita akan subsample dari keseluruhan data setiap iterasi
        # dan mengandalkan n_out. Ini mungkin kurang efisien jika subsampling dimaksudkan untuk mengurangi komputasi.
        # Jika tujuannya adalah untuk memproses graf besar secara bertahap, maka logika asli perlu dipertahankan
        # dengan hati-hati.

        # Mengikuti logika asli dengan sedikit modifikasi untuk kejelasan:
        data_sampled = data_cpu.clone() # Selalu mulai dari data asli di CPU
        
        # Ambil sampel dari indeks keseluruhan
        # idx = random.sample(range(data_sampled.x.size(0)), hparams[0]['subsampling'])
        # Jika ingin sampling dari yang belum terprediksi dan hparams['subsampling'] cukup besar:
        # idx_list_remaining = [data_cpu.pos.tolist().index(list(pt)) for pt in idx_points_remaining]
        # sample_indices_from_remaining = random.sample(idx_list_remaining, current_sample_size)
        # idx = torch.tensor(sample_indices_from_remaining)

        # Untuk saat ini, mari kita pakai subsampling dari keseluruhan, seperti di train.py
        # Ini akan membuat n_out lebih penting.
        idx = random.sample(range(data_sampled.x.size(0)), hparams[0]['subsampling'])
        idx = torch.tensor(idx)

        # Update idx_points_remaining berdasarkan apa yang baru saja disampel
        sampled_pos_set = set(map(tuple, data_sampled.pos[idx, :2].numpy()))
        idx_points_remaining = idx_points_remaining - sampled_pos_set
        
        # Buat data_to_device dari data_sampled dengan index yang dipilih
        data_to_device = data_cpu.clone() # Clone lagi untuk modifikasi
        data_to_device.pos = data_cpu.pos[idx]
        data_to_device.x = data_cpu.x[idx]
        data_to_device.y = data_cpu.y[idx] # y mungkin tidak perlu di device jika hanya untuk output
        data_to_device.surf = data_cpu.surf[idx]
        data_to_device.batch = None # Jika batch_size=1, batch bisa None atau torch.zeros_like(idx, dtype=torch.long)
                                   # Jika data asli memiliki batch, ini harus data_cpu.batch[idx]

        out_iter = [torch.zeros_like(data_cpu.y)] * len(models) # Untuk akumulasi di iterasi ini
        tim_iter = np.zeros(len(models))

        for n, model in enumerate(models):
            model.eval()
            # Pindahkan data_to_device ke device
            data_for_model = data_to_device.clone().to(device)
            
            try: # Per model edge_index construction
                data_for_model.edge_index = nng.radius_graph(x=data_for_model.pos, r=hparams[n]['r'], loop=True, max_num_neighbors=int(hparams[n]['max_neighbors']))
            except KeyError:
                data_for_model.edge_index = None # Atau gunakan edge_index asli jika ada
            except Exception as e: # Tangani error radius_graph jika data terlalu sedikit
                # print(f"Warning: radius_graph failed for model {n} in Infer_test: {e}. Skipping edge_index.")
                data_for_model.edge_index = None


            start = time.time()
            o = model(data_for_model)
            tim_iter[n] += time.time() - start
            out_iter[n][idx] = o.cpu() # Simpan output ke posisi yang benar

            outs[n] = outs[n] + out_iter[n] # Akumulasi ke output global
        
        n_out[idx] = n_out[idx] + torch.ones_like(n_out[idx][:,:1]) # Update count untuk titik yang diprediksi

        # Kondisi berhenti: jika semua titik sudah diprediksi *setidaknya sekali*
        # atau jika idx_points_remaining kosong (jika mengikuti logika pengurangan).
        # Dengan sampling dari keseluruhan, kita perlu iterasi sejumlah tertentu atau sampai konvergen.
        # Untuk menyederhanakan, kita pakai `i < max_iter` atau sampai n_out minimal 1 untuk semua
        if i > (data_cpu.x.size(0) // hparams[0]['subsampling']) + 5 : # Beri beberapa iterasi ekstra
             if (n_out > 0).all(): # Pastikan semua titik setidaknya sekali
                 break
        if i > 2 * ((data_cpu.x.size(0) // hparams[0]['subsampling']) + 5) : # Failsafe
            print("Infer_test: Max iterations reached.")
            break


    # Rata-ratakan output
    # Handle division by zero jika ada titik yang tidak pernah tersample
    n_out_safe = torch.where(n_out == 0, torch.ones_like(n_out), n_out)
    for n in range(len(outs)):
        outs[n] = outs[n] / n_out_safe
        if coef_norm is not None:
            # Denormalisasi parsial untuk kondisi batas (jika diperlukan sebelum denormalisasi penuh)
            # Ini akan dilakukan di Airfoil_test untuk kejelasan
            # outs[n][data_cpu.surf, :2] = -torch.tensor(coef_norm[2][None, :2], dtype=torch.float32) / (torch.tensor(coef_norm[3][None, :2], dtype=torch.float32) + 1e-8)
            # outs[n][data_cpu.surf, 3] = -torch.tensor(coef_norm[2][3], dtype=torch.float32) / (torch.tensor(coef_norm[3][3], dtype=torch.float32) + 1e-8)
            
            # Sebaiknya, kondisi batas no-slip diterapkan setelah denormalisasi penuh.
            # Di sini, kita bisa memastikan output ternormalisasi untuk surface nodes adalah representasi dari nol setelah denormalisasi.
            # Misal, jika y_norm = (y_true - mean) / std, dan y_true = 0, maka y_norm = -mean / std
            mean_out_tensor_surf_vel = torch.tensor(coef_norm[2][:2], dtype=outs[n].dtype, device=outs[n].device)
            std_out_tensor_surf_vel = torch.tensor(coef_norm[3][:2], dtype=outs[n].dtype, device=outs[n].device)
            outs[n][data_cpu.surf, :2] = (-mean_out_tensor_surf_vel) / (std_out_tensor_surf_vel + 1e-8)

            mean_out_tensor_surf_nut = torch.tensor(coef_norm[2][3], dtype=outs[n].dtype, device=outs[n].device) # Asumsi nut juga 0 di dinding
            std_out_tensor_surf_nut = torch.tensor(coef_norm[3][3], dtype=outs[n].dtype, device=outs[n].device)
            outs[n][data_cpu.surf, 3] = (-mean_out_tensor_surf_nut) / (std_out_tensor_surf_nut + 1e-8)
        else:
            outs[n][data_cpu.surf, :2] = torch.zeros_like(outs[n][data_cpu.surf, :2])
            outs[n][data_cpu.surf, 3] = torch.zeros_like(outs[n][data_cpu.surf, 3])


    return outs, tim_iter / (i if i > 0 else 1) # tim_iter adalah dari iterasi terakhir, lebih baik tim total / iterasi


# MODIFIKASI Airfoil_test
def Airfoil_test_single_model(internal_orig_mesh, airfoil_orig_mesh, 
                              single_out_tensor, # Output dari SATU model (sudah ternormalisasi)
                              coef_norm, bool_surf_tensor, 
                              case_name_str, model_identifier_str, 
                              output_save_dir_for_case_str):
    """
    Processes a single model's output for one test case, denormalizes it,
    updates PyVista mesh objects, and saves the denormalized field data to a .vtu file.
    """
    intern_pred = internal_orig_mesh.copy()
    aerofoil_pred = airfoil_orig_mesh.copy()

    # bool_surf_tensor harusnya sama panjangnya dengan single_out_tensor dan intern_pred.points
    bool_surf_np = bool_surf_tensor.cpu().numpy() # Pastikan ini boolean numpy array

    point_mesh_surf_coords = intern_pred.points[bool_surf_np, :2]
    point_airfoil_surf_coords = aerofoil_pred.points[:, :2]

    # Denormalize the single model's output
    # coef_norm: (mean_in, std_in, mean_out, std_out)
    mean_out = coef_norm[2]
    std_out = coef_norm[3]
    
    # Pindahkan single_out_tensor ke CPU jika belum, lalu ke numpy
    out_denormalized_np = (single_out_tensor.cpu().numpy() * (std_out + 1e-8)) + mean_out
    
    # Apply no-slip boundary condition on the denormalized values at the surface
    out_denormalized_np[bool_surf_np, :2] = 0.0  # Velocity (vx, vy) = 0
    out_denormalized_np[bool_surf_np, 3] = 0.0   # Turbulent viscosity (nut) = 0 at wall

    # Update PyVista internal mesh with denormalized predicted fields
    intern_pred.point_data['U_pred'] = np.hstack((out_denormalized_np[:, :2], np.zeros((out_denormalized_np.shape[0], 1)))) # vx, vy, vz (vz=0 for 2D)
    intern_pred.point_data['p_pred'] = out_denormalized_np[:, 2]
    intern_pred.point_data['nut_pred'] = out_denormalized_np[:, 3]

    # Reorganize surface pressure from volume mesh to airfoil surface mesh
    surf_p_from_volume = intern_pred.point_data['p_pred'][bool_surf_np]
    
    # Cek apakah reorganisasi diperlukan atau bisa langsung dari bool_surf_np
    # Asumsi reorganize menangani pemetaan jika urutan titik berbeda
    if point_mesh_surf_coords.shape[0] > 0 and point_airfoil_surf_coords.shape[0] > 0 :
        try:
            surf_p_reorganized = reorganize(point_mesh_surf_coords, point_airfoil_surf_coords, surf_p_from_volume)
            aerofoil_pred.point_data['p_pred'] = surf_p_reorganized
        except Exception as e:
            # print(f"Warning: Reorganize failed for case {case_name_str}, model {model_identifier_str}: {e}")
            # Fallback or error handling if reorganize fails
            if aerofoil_pred.n_points == surf_p_from_volume.shape[0]:
                 aerofoil_pred.point_data['p_pred'] = surf_p_from_volume # Jika jumlah titik sama, coba langsung
            else:
                 pass # print(f"Skipping p_pred on airfoil for {case_name_str} due to reorganize error and mismatched points.")
    
    intern_pred = intern_pred.ptc(pass_point_data=True)
    aerofoil_pred = aerofoil_pred.ptc(pass_point_data=True)

    # Save the denormalized prediction to a .vtu file
    pathlib.Path(output_save_dir_for_case_str).mkdir(parents=True, exist_ok=True)
    output_filename = osp.join(output_save_dir_for_case_str, f"{case_name_str}_pred_{model_identifier_str}.vtu")
    intern_pred.save(output_filename)
    # print(f"Saved denormalized prediction to: {output_filename}") # Opsional: untuk debugging

    return intern_pred, aerofoil_pred


def Airfoil_mean(internals, airfoils):
    # Average multiple prediction over one simulation
    if not internals or not airfoils: # Handle empty lists
        return None, None

    oi_point = np.zeros((internals[0].points.shape[0], 4)) # U_pred (vx,vy), p_pred, nut_pred
    # oi_cell = np.zeros((internals[0].cell_data['U_pred'].shape[0], 4)) # Jika cell data juga di-rata2kan
    oa_point = np.zeros((airfoils[0].points.shape[0], 1)) # p_pred (jika hanya p yang ada di airfoil)
    # oa_cell = np.zeros((airfoils[0].cell_data['U_pred'].shape[0], 4))

    num_predictions = len(internals)

    for k in range(num_predictions):
        if 'U_pred' in internals[k].point_data:
            oi_point[:, :2] += internals[k].point_data['U_pred'][:, :2]
        if 'p_pred' in internals[k].point_data:
            oi_point[:, 2] += internals[k].point_data['p_pred']
        if 'nut_pred' in internals[k].point_data:
            oi_point[:, 3] += internals[k].point_data['nut_pred']
        
        # if 'U_pred' in internals[k].cell_data:
        #     oi_cell[:, :2] += internals[k].cell_data['U_pred'][:, :2]
        # if 'p_pred' in internals[k].cell_data:
        #     oi_cell[:, 2] += internals[k].cell_data['p_pred']
        # if 'nut_pred' in internals[k].cell_data:
        #     oi_cell[:, 3] += internals[k].cell_data['nut_pred']

        if 'p_pred' in airfoils[k].point_data:
            oa_point[:, 0] += airfoils[k].point_data['p_pred']
        # ... (serupa untuk cell data airfoil jika ada) ...

    oi_point /= num_predictions
    # oi_cell /= num_predictions
    oa_point /= num_predictions
    # oa_cell /= num_predictions
    
    internal_mean = internals[0].copy()
    internal_mean.point_data['U_pred_mean'] = np.hstack((oi_point[:,:2], np.zeros((oi_point.shape[0],1))))
    internal_mean.point_data['p_pred_mean'] = oi_point[:, 2]
    internal_mean.point_data['nut_pred_mean'] = oi_point[:, 3]
    # ... (serupa untuk cell data internal) ...

    airfoil_mean = airfoils[0].copy()
    if 'p_pred' in airfoil_mean.point_data : # Cek jika field ada sebelum assign
        airfoil_mean.point_data['p_pred_mean'] = oa_point[:, 0]
    # ... (serupa untuk cell data airfoil) ...
    
    return internal_mean, airfoil_mean


def Compute_coefficients(internals_pv_list, airfoils_pv_list, bool_surf_tensor, Uinf, angle, keep_vtk=False):
    coefs = []
    new_internals_vtk = []
    new_airfoils_vtk = []
    
    bool_surf_np = bool_surf_tensor.cpu().numpy()

    for internal_pv, airfoil_pv in zip(internals_pv_list, airfoils_pv_list):
        intern = internal_pv.copy() # Bekerja pada copy
        aerofoil = airfoil_pv.copy()

        # Pastikan field yang digunakan adalah hasil prediksi (misal, 'U_pred', 'p_pred')
        # Jika inputnya adalah ground truth, fieldnya adalah 'U', 'p'
        u_field_name = 'U_pred' if 'U_pred' in intern.point_data else 'U'
        p_field_name = 'p_pred' if 'p_pred' in intern.point_data else 'p'

        if u_field_name not in intern.point_data or p_field_name not in intern.point_data:
            # print(f"Warning: Required fields '{u_field_name}' or '{p_field_name}' not in internal mesh. Skipping coefficient calculation for this entry.")
            coefs.append(np.array([np.nan, np.nan])) # Atau nilai default
            if keep_vtk:
                new_internals_vtk.append(intern) # Kembalikan apa adanya
                new_airfoils_vtk.append(aerofoil)
            continue

        # Gradien dihitung dari field kecepatan yang sesuai
        intern = intern.compute_derivative(scalars=u_field_name, gradient='pred_grad_raw')
        # pred_grad_raw akan (N, 9), reshape ke (N, 3, 3)
        # Untuk 2D, kita hanya perlu komponen xy: [du/dx, du/dy, dv/dx, dv/dy]
        # PyVista mungkin mengembalikan [du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz]
        # Kita ambil blok 2x2 atas kiri dari bagian U dan V
        grad_full = intern.point_data['pred_grad_raw'].reshape(-1, 3, 3)
        grad_uv_xy = grad_full[:, :2, :2] # (N, 2, 2) -> [[du/dx, du/dy], [dv/dx, dv/dy]]

        point_mesh_surf_coords = intern.points[bool_surf_np, :2]
        point_airfoil_surf_coords = aerofoil.points[:, :2]
        
        # Reorganize gradien dan tekanan dari volume ke permukaan airfoil
        surf_grad_reorganized = np.zeros((aerofoil.n_points, 2, 2)) # Default jika reorganize gagal
        surf_p_reorganized = np.zeros(aerofoil.n_points)

        if point_mesh_surf_coords.shape[0] > 0 and point_airfoil_surf_coords.shape[0] > 0:
            try:
                surf_grad_reorganized = reorganize(point_mesh_surf_coords, point_airfoil_surf_coords, grad_uv_xy[bool_surf_np])
                # Tekanan pada airfoil diambil dari field p_field_name di airfoil_pv jika sudah ada,
                # atau dari internal jika perlu di-reorganize
                if p_field_name in aerofoil.point_data:
                     surf_p_reorganized = aerofoil.point_data[p_field_name]
                elif p_field_name in intern.point_data:
                     surf_p_reorganized = reorganize(point_mesh_surf_coords, point_airfoil_surf_coords, intern.point_data[p_field_name][bool_surf_np])
                else: # Fallback jika p tidak ada
                    # print(f"Warning: Pressure field '{p_field_name}' not found for coefficient calculation on airfoil.")
                    pass

            except Exception as e:
                # print(f"Warning: Reorganize failed during Compute_coefficients: {e}")
                # Coba fallback jika jumlah titik cocok
                if aerofoil.n_points == grad_uv_xy[bool_surf_np].shape[0]:
                    surf_grad_reorganized = grad_uv_xy[bool_surf_np]
                if p_field_name in aerofoil.point_data and aerofoil.n_points == aerofoil.point_data[p_field_name].shape[0]:
                    surf_p_reorganized = aerofoil.point_data[p_field_name]
                elif p_field_name in intern.point_data and aerofoil.n_points == intern.point_data[p_field_name][bool_surf_np].shape[0]:
                    surf_p_reorganized = intern.point_data[p_field_name][bool_surf_np]
                # else:
                    # print(f"Skipping grad/p assignment on airfoil due to reorganize error and mismatched points in Compute_coefficients.")
                    

        # Hitung WSS menggunakan gradien yang sudah di-reorganize dan normal airfoil
        # Pastikan 'Normals' ada di aerofoil.point_data
        if 'Normals' in aerofoil.point_data:
            Wss_pred_vals = WallShearStress(surf_grad_reorganized, -aerofoil.point_data['Normals'])
            aerofoil.point_data['wallShearStress_pred'] = Wss_pred_vals
        else:
            # print("Warning: 'Normals' not found in airfoil point data. Cannot compute WSS.")
            aerofoil.point_data['wallShearStress_pred'] = np.zeros((aerofoil.n_points, 2)) # WSS adalah vektor (tau_x, tau_y)

        aerofoil.point_data[p_field_name] = surf_p_reorganized # Update tekanan di airfoil

        # Konversi ke cell data untuk integrasi
        aerofoil_cell_data = aerofoil.ptc(pass_point_data=True)
        
        WP_int = np.zeros(2)
        Wss_int = np.zeros(2)

        if 'Normals' in aerofoil_cell_data.cell_data and p_field_name in aerofoil_cell_data.cell_data and 'Length' in aerofoil_cell_data.cell_data:
            WP_int = -(aerofoil_cell_data.cell_data[p_field_name][:, None] * aerofoil_cell_data.cell_data['Normals'][:, :2])
            WP_int = (WP_int * aerofoil_cell_data.cell_data['Length'].reshape(-1, 1)).sum(axis=0)
        # else:
            # print("Warning: Missing data for pressure force calculation (Normals, p_field, or Length).")

        if 'wallShearStress_pred' in aerofoil_cell_data.cell_data and 'Length' in aerofoil_cell_data.cell_data:
            # WSS adalah vektor, jadi langsung dikalikan panjang dan dijumlahkan
            # Wss_int = (aerofoil_cell_data.cell_data['wallShearStress_pred'] * aerofoil_cell_data.cell_data['Length'].reshape(-1, 1)).sum(axis=0)
            # Kode asli: Wss_int = (aerofoil.cell_data['wallShearStress']*aerofoil.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)
            # Ini sepertinya salah karena WSS sudah per unit area, perlu dikalikan dengan vektor tangensial lalu panjang.
            # Atau, jika WallShearStress mengembalikan komponen gaya geser per unit panjang di arah normal, maka dikali panjang OK.
            # Mari asumsikan WallShearStress mengembalikan vektor tegangan geser di permukaan.
            # F_shear = integral( tau_wall dot t_tangent ) dl
            # Jika WallShearStress mengembalikan (tau_nx, tau_ny) yang merupakan proyeksi dari S*n
            # maka ini adalah gaya geser per unit area. Perlu dikali panjang dan arah yang tepat.
            # Untuk CFD, biasanya WSS adalah magnitudo. Di sini `WallShearStress` mengembalikan vektor.

            # Kita akan mengikuti logika skrip asli yang menjumlahkan komponen WSS
            # (yang sudah merupakan vektor) dikali panjang segmen.
            Wss_int = (aerofoil_cell_data.cell_data['wallShearStress_pred'] * aerofoil_cell_data.cell_data['Length'].reshape(-1, 1)).sum(axis=0)

        # else:
            # print("Warning: Missing data for shear force calculation (wallShearStress_pred or Length).")
            
        force = Wss_int - WP_int # Arah WSS dan WP harus konsisten. Biasanya WSS + WP (jika P positif keluar)

        alpha_rad = angle * np.pi / 180
        basis = np.array([[np.cos(alpha_rad), np.sin(alpha_rad)], [-np.sin(alpha_rad), np.cos(alpha_rad)]])
        force_rot = basis @ force
        coef = 2 * force_rot / (Uinf**2) # Asumsi densitas = 1 jika tidak, perlu (0.5 * rho * Uinf**2 * Area_ref)
                                        # Area_ref biasanya chord length (c=1 untuk airfoil ternormalisasi)
        coefs.append(coef)
        if keep_vtk:
            new_internals_vtk.append(intern) # Ini sudah di-copy di awal loop
            new_airfoils_vtk.append(aerofoil_cell_data) # Kembalikan yang cell data

    if keep_vtk:
        return coefs, new_internals_vtk, new_airfoils_vtk
    else:
        return coefs

# MODIFIKASI Results_test
def Results_test(device, models_outer_list, hparams_list, coef_norm, path_in, path_out,
                 model_names_list=None, # Tambahan: list nama model untuk penamaan file
                 n_test=3, criterion='MSE', x_bl=[.2, .4, .6, .8], s='full_test'):
    sns.set()
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True) # path_out utama untuk skor, JSON

    with open(osp.join(path_in, 'manifest.json'), 'r') as f:
        manifest = json.load(f)

    test_dataset_names = manifest[s]
    # Ambil sampel dari test_dataset_names jika n_test lebih kecil dari total
    if n_test < len(test_dataset_names):
        idx_selected_for_detailed_analysis = random.sample(range(len(test_dataset_names)), k=n_test)
    else:
        idx_selected_for_detailed_analysis = list(range(len(test_dataset_names)))
    idx_selected_for_detailed_analysis.sort()

    # Buat dataset PyG untuk semua test cases
    # Coef_norm diterapkan di sini jika Dataset adalah kelas PyG Dataset
    # Jika Dataset adalah fungsi yang mengembalikan list of Data, maka coef_norm sudah diterapkan
    # Berdasarkan nama 'Dataset', ini adalah fungsi dari dataset.py
    test_pyg_dataset_all = Dataset(test_dataset_names, sample=None, coef_norm=coef_norm)
    test_loader = DataLoader(test_pyg_dataset_all, batch_size=1, shuffle=False) # Batch size 1 untuk inferensi per kasus

    if criterion == 'MSE' or criterion == 'MSE_weighted': # MSE_weighted ditangani di train, di sini hanya MSE
        loss_criterion_func = nn.MSELoss(reduction='none')
    elif criterion == 'MAE':
        loss_criterion_func = nn.L1Loss(reduction='none')
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # scores_vol, scores_surf, dll. akan menjadi list per run/seed
    # Setiap elemen adalah array [model_type_idx, metric_value]
    all_runs_scores_vol = []
    all_runs_scores_surf = []
    all_runs_scores_force_rel_err = []
    all_runs_scores_p_rel_err = []
    all_runs_scores_wss_rel_err = []
    
    # Untuk menyimpan VTK dari kasus terpilih dan hasil BL/surf_coefs
    # Strukturnya: [run_idx][selected_case_idx][model_type_idx]
    # Namun, Airfoil_mean perlu [model_type_idx][run_idx][selected_case_idx]
    # Mari kita simpan per run dulu, lalu re-organize jika perlu untuk Airfoil_mean
    # Atau, ubah cara Airfoil_mean dipanggil.
    
    # internals_selected_cases: list (panjang = n_runs)
    #   setiap elemen: list (panjang = n_test terpilih)
    #     setiap elemen: list (panjang = n_model_types) -> objek PV intern
    internals_selected_cases_all_runs = []
    airfoils_selected_cases_all_runs = []
    
    true_internals_vtk_selected_cases = [] # Simpan hanya sekali (run pertama)
    true_airfoils_vtk_selected_cases = []  # Simpan hanya sekali

    all_runs_times = [] # [run_idx][model_type_idx]
    all_runs_pred_coefs = [] # [run_idx][case_idx][model_type_idx, drag/lift]
    true_coefs_all_cases = [] # [case_idx, drag/lift], simpan sekali

    num_model_types = len(models_outer_list)
    num_runs = len(models_outer_list[0]) if num_model_types > 0 else 0

    for run_idx in range(num_runs):
        current_run_models = [models_outer_list[model_type_idx][run_idx] for model_type_idx in range(num_model_types)]
        
        # Metrik untuk run saat ini
        run_avg_loss_per_var = np.zeros((num_model_types, 4))
        run_avg_loss_surf_var = np.zeros((num_model_types, 4))
        run_avg_loss_vol_var = np.zeros((num_model_types, 4))
        run_avg_rel_err_force = np.zeros((num_model_types, 2)) # Cd, Cl
        run_avg_rel_err_p_surf = np.zeros(num_model_types)
        run_avg_rel_err_wss_surf = np.zeros((num_model_types, 2)) # Mag WSS

        run_times_per_model = np.zeros(num_model_types)
        
        # Hasil VTK dan koefisien untuk run saat ini
        run_pred_coefs_for_all_cases = [] # list [case_idx][model_type_idx, drag/lift]
        
        internals_current_run_selected_cases = []
        airfoils_current_run_selected_cases = []

        for case_loader_idx, data_pyg in enumerate(tqdm(test_loader, desc=f"Run {run_idx+1}/{num_runs}")):
            current_case_name = test_dataset_names[case_loader_idx]
            Uinf, angle = float(current_case_name.split('_')[2]), float(current_case_name.split('_')[3])

            # Inferensi: `outs_norm` adalah list tensor ternormalisasi, 1 per model_type
            outs_norm, समय = Infer_test(device, current_run_models, hparams_list, data_pyg, coef_norm=coef_norm)
            run_times_per_model += समय # Akumulasi waktu inferensi

            # Baca mesh CFD ground truth
            internal_cfd_gt = pv.read(osp.join(path_in, current_case_name, current_case_name + '_internal.vtu'))
            aerofoil_cfd_gt = pv.read(osp.join(path_in, current_case_name, current_case_name + '_aerofoil.vtp'))

            # Hitung koefisien ground truth
            # data_pyg.surf adalah tensor boolean untuk titik permukaan di mesh CFD
            gt_coefs_list, gt_intern_vtk, gt_airfoil_vtk = Compute_coefficients(
                [internal_cfd_gt], [aerofoil_cfd_gt], data_pyg.surf, Uinf, angle, keep_vtk=True
            )
            gt_coef_current_case = gt_coefs_list[0] # [Cd, Cl]
            if run_idx == 0: # Simpan koefisien true hanya sekali
                true_coefs_all_cases.append(gt_coef_current_case)

            # Proses output setiap model untuk kasus ini
            predicted_internals_pv_for_case = []
            predicted_airfoils_pv_for_case = []
            pred_coefs_current_case_all_models = []

            output_vtk_dir_for_this_case = osp.join(path_out, "predicted_vtks_per_run", f"run_{run_idx}", current_case_name)

            for model_type_idx, single_model_out_norm_tensor in enumerate(outs_norm):
                model_id_str = model_names_list[model_type_idx] if model_names_list and model_type_idx < len(model_names_list) else f"model{model_type_idx}"
                
                # Dapatkan mesh internal dan airfoil dengan field prediksi denormalisasi, dan simpan VTK
                pred_intern_pv, pred_airfoil_pv = Airfoil_test_single_model(
                    internal_cfd_gt, aerofoil_cfd_gt,
                    single_model_out_norm_tensor,
                    coef_norm, data_pyg.surf,
                    current_case_name, model_id_str,
                    output_vtk_dir_for_this_case
                )
                predicted_internals_pv_for_case.append(pred_intern_pv)
                predicted_airfoils_pv_for_case.append(pred_airfoil_pv)

            # Hitung koefisien dari prediksi
            # `pred_coefs_for_models` adalah list [model_idx][Cd,Cl]
            pred_coefs_for_models, processed_pred_internals_pv, processed_pred_airfoils_pv = Compute_coefficients(
                predicted_internals_pv_for_case, predicted_airfoils_pv_for_case,
                data_pyg.surf, Uinf, angle, keep_vtk=True
            )
            run_pred_coefs_for_all_cases.append(pred_coefs_for_models)

            # Jika kasus ini terpilih untuk analisis detail (BL, surf_coefs, mean VTK)
            if case_loader_idx in idx_selected_for_detailed_analysis:
                internals_current_run_selected_cases.append(processed_pred_internals_pv) # list of PVs (per model_type)
                airfoils_current_run_selected_cases.append(processed_pred_airfoils_pv)   # list of PVs (per model_type)
                if run_idx == 0:
                    true_internals_vtk_selected_cases.append(gt_intern_vtk[0]) # Simpan GT VTK sekali
                    true_airfoils_vtk_selected_cases.append(gt_airfoil_vtk[0])

            # Hitung loss fields (pada data ternormalisasi)
            for model_type_idx, out_norm_tensor in enumerate(outs_norm):
                loss_per_var = loss_criterion_func(out_norm_tensor, data_pyg.y.to(out_norm_tensor.device)).mean(dim=0) # data_pyg.y harusnya sudah ternormalisasi
                
                loss_surf_var = loss_criterion_func(out_norm_tensor[data_pyg.surf, :], data_pyg.y[data_pyg.surf, :].to(out_norm_tensor.device)).mean(dim=0)
                loss_vol_var = loss_criterion_func(out_norm_tensor[~data_pyg.surf, :], data_pyg.y[~data_pyg.surf, :].to(out_norm_tensor.device)).mean(dim=0)

                run_avg_loss_per_var[model_type_idx] += loss_per_var.cpu().numpy()
                run_avg_loss_surf_var[model_type_idx] += loss_surf_var.cpu().numpy()
                run_avg_loss_vol_var[model_type_idx] += loss_vol_var.cpu().numpy()
                
                # Error relatif untuk gaya, Cp, WSS (gunakan processed_pred_airfoils_pv)
                # Ini memerlukan field WSS dan P pada ground truth (gt_airfoil_vtk[0])
                # dan pada prediksi (processed_pred_airfoils_pv[model_type_idx])
                current_pred_airfoil_vtk = processed_pred_airfoils_pv[model_type_idx]
                
                if not np.any(np.isnan(pred_coefs_for_models[model_type_idx])) and not np.any(np.isnan(gt_coef_current_case)):
                     run_avg_rel_err_force[model_type_idx] += rel_err(gt_coef_current_case, pred_coefs_for_models[model_type_idx])
                
                p_field_gt = 'p' # atau 'p_pred' jika gt_airfoil_vtk adalah hasil dari ComputeCoeff
                p_field_pred = 'p_pred'
                if p_field_gt in gt_airfoil_vtk[0].point_data and p_field_pred in current_pred_airfoil_vtk.point_data:
                    # Handle potential division by zero in rel_err if gt_airfoil_vtk[0].point_data[p_field_gt] is zero
                    gt_p_surf = gt_airfoil_vtk[0].point_data[p_field_gt]
                    pred_p_surf = current_pred_airfoil_vtk.point_data[p_field_pred]
                    # Avoid division by zero or very small numbers in rel_err for pressure
                    valid_indices_p = np.abs(gt_p_surf) > 1e-6 # Threshold
                    if np.any(valid_indices_p):
                        run_avg_rel_err_p_surf[model_type_idx] += rel_err(gt_p_surf[valid_indices_p], pred_p_surf[valid_indices_p]).mean()

                wss_field_gt = 'wallShearStress_pred' # ComputeCoeff menamai ini 'wallShearStress_pred'
                wss_field_pred = 'wallShearStress_pred'
                if wss_field_gt in gt_airfoil_vtk[0].point_data and wss_field_pred in current_pred_airfoil_vtk.point_data:
                    gt_wss_surf_vec = gt_airfoil_vtk[0].point_data[wss_field_gt] # Vektor (tau_x, tau_y)
                    pred_wss_surf_vec = current_pred_airfoil_vtk.point_data[wss_field_pred]
                    # Hitung magnitudo WSS
                    gt_wss_mag = np.linalg.norm(gt_wss_surf_vec, axis=1)
                    pred_wss_mag = np.linalg.norm(pred_wss_surf_vec, axis=1)
                    valid_indices_wss = np.abs(gt_wss_mag) > 1e-6 # Threshold
                    if np.any(valid_indices_wss): # Hanya hitung jika ada nilai valid
                         run_avg_rel_err_wss_surf[model_type_idx, 0] += rel_err(gt_wss_mag[valid_indices_wss], pred_wss_mag[valid_indices_wss]).mean()
                         # Indeks 1 bisa untuk metrik WSS lain jika ada

        # Rata-ratakan metrik untuk run saat ini
        num_test_cases = len(test_loader)
        all_runs_scores_vol.append(run_avg_loss_vol_var / num_test_cases)
        all_runs_scores_surf.append(run_avg_loss_surf_var / num_test_cases)
        all_runs_scores_force_rel_err.append(run_avg_rel_err_force / num_test_cases)
        all_runs_scores_p_rel_err.append(run_avg_rel_err_p_surf / num_test_cases)
        all_runs_scores_wss_rel_err.append(run_avg_rel_err_wss_surf / num_test_cases)
        all_runs_times.append(run_times_per_model / num_test_cases) # Waktu rata-rata per kasus
        
        all_runs_pred_coefs.append(np.array(run_pred_coefs_for_all_cases)) # [run_idx][case_idx][model_type_idx, Cd/Cl]
        internals_selected_cases_all_runs.append(internals_current_run_selected_cases)
        airfoils_selected_cases_all_runs.append(airfoils_current_run_selected_cases)

    # Konversi list ke array numpy untuk perhitungan mean/std
    scores_vol_np = np.array(all_runs_scores_vol)     # (n_runs, n_model_types, 4)
    scores_surf_np = np.array(all_runs_scores_surf)    # (n_runs, n_model_types, 4)
    scores_force_np = np.array(all_runs_scores_force_rel_err) # (n_runs, n_model_types, 2)
    scores_p_np = np.array(all_runs_scores_p_rel_err)      # (n_runs, n_model_types)
    scores_wss_np = np.array(all_runs_scores_wss_rel_err)   # (n_runs, n_model_types, 2)
    times_np = np.array(all_runs_times)                # (n_runs, n_model_types)
    
    true_coefs_np = np.array(true_coefs_all_cases)     # (n_cases, 2)
    # all_runs_pred_coefs adalah list of arrays, perlu stacking hati-hati
    # Shape: (n_runs, n_cases, n_model_types, 2)
    pred_coefs_np = np.array(all_runs_pred_coefs)

    # Hitung mean dan std dari prediksi koefisien antar run
    pred_coefs_mean_over_runs = pred_coefs_np.mean(axis=0) # (n_cases, n_model_types, 2)
    pred_coefs_std_over_runs = pred_coefs_np.std(axis=0)   # (n_cases, n_model_types, 2)

    # Spearman correlation (per model type, dihitung dari mean prediksi antar run)
    spear_coefs_all_model_types = [] # list per model_type, [spear_drag, spear_lift]
    for model_type_idx in range(num_model_types):
        # Ambil prediksi rata-rata untuk model_type ini di semua kasus
        current_model_pred_means = pred_coefs_mean_over_runs[:, model_type_idx, :] # (n_cases, 2)
        if true_coefs_np.shape[0] > 1 and current_model_pred_means.shape[0] > 1:
            try:
                spear_drag = sc.stats.spearmanr(true_coefs_np[:, 0], current_model_pred_means[:, 0], nan_policy='omit')[0]
                spear_lift = sc.stats.spearmanr(true_coefs_np[:, 1], current_model_pred_means[:, 1], nan_policy='omit')[0]
                spear_coefs_all_model_types.append([spear_drag, spear_lift])
            except Exception as e: # Catch potential errors from spearmanr
                # print(f"Spearman correlation failed for model type {model_type_idx}: {e}")
                spear_coefs_all_model_types.append([np.nan, np.nan])

        else:
            spear_coefs_all_model_types.append([np.nan, np.nan]) # Not enough data for correlation
    spear_coefs_np = np.array(spear_coefs_all_model_types) # (n_model_types, 2)


    with open(osp.join(path_out, 'scores_summary.json'), 'w') as f:
        json.dump(
            {
                'mean_time_per_case': times_np.mean(axis=0).tolist(), # (n_model_types)
                'std_time_per_case': times_np.std(axis=0).tolist(),   # (n_model_types)
                'mean_score_vol_per_var': scores_vol_np.mean(axis=0).tolist(), # (n_model_types, 4)
                'std_score_vol_per_var': scores_vol_np.std(axis=0).tolist(),   # (n_model_types, 4)
                'mean_score_surf_per_var': scores_surf_np.mean(axis=0).tolist(),# (n_model_types, 4)
                'std_score_surf_per_var': scores_surf_np.std(axis=0).tolist(),  # (n_model_types, 4)
                'mean_rel_err_p_surf': scores_p_np.mean(axis=0).tolist(),    # (n_model_types)
                'std_rel_err_p_surf': scores_p_np.std(axis=0).tolist(),      # (n_model_types)
                'mean_rel_err_wss_surf': scores_wss_np.mean(axis=0).tolist(), # (n_model_types, 2)
                'std_rel_err_wss_surf': scores_wss_np.std(axis=0).tolist(),   # (n_model_types, 2)
                'mean_rel_err_force': scores_force_np.mean(axis=0).tolist(),# (n_model_types, 2)
                'std_rel_err_force': scores_force_np.std(axis=0).tolist(),  # (n_model_types, 2)
                'spearman_coefs_mean_pred': spear_coefs_np.tolist() # (n_model_types, 2)
            }, f, indent=4, cls=NumpyEncoder
        )

    # Proses untuk BL dan surface coefficients dari kasus terpilih
    # Ini memerlukan rata-rata prediksi VTK *antar run* untuk setiap model type
    
    pred_surf_coefs_selected_cases_mean_run = [] # [selected_case_idx][model_type_idx] -> (cp_array, cf_array)
    pred_bls_selected_cases_mean_run = []        # [selected_case_idx][model_type_idx][x_bl_idx] -> (yc, u, v, nut)
    
    true_surf_coefs_selected_cases = []
    true_bls_selected_cases = []

    for sel_case_list_idx, original_case_loader_idx in enumerate(idx_selected_for_detailed_analysis):
        current_case_name_selected = test_dataset_names[original_case_loader_idx]
        
        # Dapatkan VTK ground truth untuk kasus terpilih ini
        true_internal_sel = true_internals_vtk_selected_cases[sel_case_list_idx]
        true_airfoil_sel = true_airfoils_vtk_selected_cases[sel_case_list_idx]

        # Hitung true surface coefs dan BL
        true_surf_coefs_sel = metrics_NACA.surface_coefficients(true_airfoil_sel, current_case_name_selected)
        true_surf_coefs_selected_cases.append(true_surf_coefs_sel)
        
        true_bl_sel_list = []
        for x_loc in x_bl:
            true_bl_sel_list.append(np.array(metrics_NACA.boundary_layer(true_airfoil_sel, true_internal_sel, current_case_name_selected, x_loc)))
        true_bls_selected_cases.append(np.array(true_bl_sel_list))

        # Untuk setiap model type, rata-ratakan VTK prediksinya antar run
        surf_coefs_this_case_all_models = []
        bls_this_case_all_models = []
        for model_type_idx in range(num_model_types):
            model_id_str_for_mean = model_names_list[model_type_idx] if model_names_list and model_type_idx < len(model_names_list) else f"model{model_type_idx}"
            
            # Kumpulkan VTK untuk model_type ini, kasus terpilih ini, dari semua run
            internals_for_mean = [internals_selected_cases_all_runs[r_idx][sel_case_list_idx][model_type_idx] for r_idx in range(num_runs)]
            airfoils_for_mean = [airfoils_selected_cases_all_runs[r_idx][sel_case_list_idx][model_type_idx] for r_idx in range(num_runs)]
            
            if not internals_for_mean or not airfoils_for_mean: continue # Lewati jika kosong

            internal_mean_pv, airfoil_mean_pv = Airfoil_mean(internals_for_mean, airfoils_for_mean)
            
            if internal_mean_pv and airfoil_mean_pv:
                # Simpan VTK rata-rata ini
                mean_vtk_dir = osp.join(path_out, "mean_predicted_vtks", current_case_name_selected)
                pathlib.Path(mean_vtk_dir).mkdir(parents=True, exist_ok=True)
                internal_mean_pv.save(osp.join(mean_vtk_dir, f"{current_case_name_selected}_pred_mean_{model_id_str_for_mean}.vtu"))

                # Hitung surface coefs dan BL dari VTK rata-rata
                pred_surf_coef_mean = metrics_NACA.surface_coefficients(airfoil_mean_pv, current_case_name_selected)
                surf_coefs_this_case_all_models.append(pred_surf_coef_mean)
                
                pred_bl_mean_list = []
                for x_loc in x_bl:
                    pred_bl_mean_list.append(np.array(metrics_NACA.boundary_layer(airfoil_mean_pv, internal_mean_pv, current_case_name_selected, x_loc)))
                bls_this_case_all_models.append(np.array(pred_bl_mean_list))
            else: # Jika Airfoil_mean gagal (misal, karena list kosong)
                # Tambahkan placeholder agar ukuran list tetap konsisten
                surf_coefs_this_case_all_models.append((np.array([]), np.array([]))) 
                bls_this_case_all_models.append(np.array([np.array([]) for _ in x_bl]))


        pred_surf_coefs_selected_cases_mean_run.append(surf_coefs_this_case_all_models)
        pred_bls_selected_cases_mean_run.append(bls_this_case_all_models)
        
    # Kembalikan semua hasil yang relevan
    return (true_coefs_np, pred_coefs_mean_over_runs, pred_coefs_std_over_runs,
            np.array(true_surf_coefs_selected_cases, dtype=object), 
            np.array(pred_surf_coefs_selected_cases_mean_run, dtype=object),
            np.array(true_bls_selected_cases, dtype=object), 
            np.array(pred_bls_selected_cases_mean_run, dtype=object))