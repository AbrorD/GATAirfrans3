# get_results.py
import yaml, json
import torch
import metrics 
from dataset import Dataset 
import os.path as osp
import pathlib 
import os 
import numpy as np

use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

root_dir = "." 
tasks = ['full', 'scarce', 'reynolds', 'aoa']

for task in tasks:
    print(f'Generating results for task {task}...')
    
    s_test_split_name = task + '_test' if task != 'scarce' else 'full_test'
    s_train_split_name = task + '_train'

    data_dir = osp.join(root_dir, 'Dataset') 
    manifest_path = osp.join(data_dir, 'manifest.json')
    if not osp.exists(manifest_path):
        print(f"Error: manifest.json not found at {manifest_path}")
        continue 

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    if s_train_split_name not in manifest:
        print(f"Error: Training split '{s_train_split_name}' not found in manifest.json for task '{task}'. Skipping.")
        continue
        
    manifest_train_list = manifest[s_train_split_name]
    
    # >>> MODIFIKASI DIMULAI DI SINI <<<
    # Coba muat coef_norm dari file yang disimpan oleh main.py
    coef_norm = None
    COEF_NORM_LOAD_DIR = 'Dataset'
    coef_norm_load_path = osp.join(COEF_NORM_LOAD_DIR, f'normalization_coefficients_{task}.pt')

    if osp.exists(coef_norm_load_path):
        try:
            coef_norm = torch.load(coef_norm_load_path)
            # Pastikan formatnya adalah tuple numpy array
            if isinstance(coef_norm, tuple) and len(coef_norm) == 4 and \
               all(isinstance(arr, np.ndarray) for arr in coef_norm):
                print(f"Normalization coefficients for task '{task}' loaded from {coef_norm_load_path}")
            else: # Jika formatnya tuple tensor atau tidak sesuai, coba konversi atau error
                try:
                    coef_norm = tuple(c.numpy() if hasattr(c, 'numpy') else c for c in coef_norm)
                    if not (isinstance(coef_norm, tuple) and len(coef_norm) == 4 and \
                            all(isinstance(arr, np.ndarray) for arr in coef_norm)):
                        raise TypeError("Converted coef_norm is not a tuple of 4 numpy arrays.")
                    print(f"Normalization coefficients for task '{task}' loaded from {coef_norm_load_path} and ensured numpy format.")
                except Exception as e_conv:
                    print(f"Warning: Loaded normalization_coefficients from {coef_norm_load_path} are not in the expected format (tuple of numpy arrays/tensors). Error: {e_conv}. Will attempt to recalculate.")
                    coef_norm = None # Set None agar dihitung ulang
        except Exception as e:
            print(f"Warning: Could not load normalization_coefficients from {coef_norm_load_path}. Error: {e}. Will attempt to recalculate.")
            coef_norm = None # Set None agar dihitung ulang
    else:
        print(f"Warning: Normalization coefficient file {coef_norm_load_path} not found. Will attempt to recalculate for task '{task}'.")

    if coef_norm is None: # Jika gagal dimuat atau file tidak ada
        if not manifest_train_list:
            print(f"Error: Training dataset for task '{task}' is empty AND normalization file not found. Cannot proceed.")
            # Tentukan dimensi default jika perlu, atau lewati task
            default_dim_in = 7 # Sesuaikan jika fitur input Anda berubah
            default_dim_out = 4 # Sesuaikan jika fitur output Anda berubah
            coef_norm = (np.zeros(default_dim_in, dtype=np.float32), np.ones(default_dim_in, dtype=np.float32),
                         np.zeros(default_dim_out, dtype=np.float32), np.ones(default_dim_out, dtype=np.float32))
            print("Using placeholder normalization coefficients due to missing file and empty train list.")
        else:
            print(f"Recalculating normalization coefficients for task '{task}' from training data.")
            try:
                _, coef_norm = Dataset(manifest_train_list, norm=True, sample=None) 
            except Exception as e_calc:
                print(f"Error recalculating normalization coefficients for task '{task}': {e_calc}")
                continue # Lewati task ini jika tidak bisa mendapatkan coef_norm
    # >>> MODIFIKASI SELESAI DI SINI <<<

    model_names = ['MLP', 'GraphSAGE', 'PointNet', 'GUNet', 'GAT'] 
    loaded_models_outer_list = [] 
    hparams_list_per_model_type = []
    processed_model_names_for_results_test = [] # Untuk menyimpan nama model yang berhasil di-load

    all_models_fully_loaded = True
    for model_name in model_names:
        model_file_path = osp.join(root_dir, 'metrics', task, model_name, model_name)
        
        if not osp.exists(model_file_path):
            print(f"Info: Model file for {model_name} on task {task} not found at {model_file_path}. Skipping this model type.")
            continue
            
        current_model_runs_list = None
        try:
            current_model_runs_list = torch.load(model_file_path, map_location=device)
            if not isinstance(current_model_runs_list, list) or not all(isinstance(m, torch.nn.Module) for m in current_model_runs_list):
                print(f"Warning: Model file {model_file_path} for {model_name} does not contain a valid list of PyTorch models. Skipping.")
                continue
        except Exception as e:
            print(f"Error loading model file {model_file_path} for {model_name}: {e}. Skipping this model type.")
            continue

        params_yaml_path = osp.join(root_dir, 'params.yaml') 
        if not osp.exists(params_yaml_path):
            print(f"Error: params.yaml not found at {params_yaml_path}. Cannot load hparams. Aborting for task {task}.")
            all_models_fully_loaded = False; break 
        
        current_hparams_model = None
        with open(params_yaml_path, 'r') as f:
            hparam_all_models_yaml = yaml.safe_load(f)
            if model_name not in hparam_all_models_yaml:
                print(f"Warning: Hyperparameters for {model_name} not found in params.yaml. Skipping this model type.")
                continue # Lewati model ini jika hparams tidak ada
            current_hparams_model = hparam_all_models_yaml[model_name]
        
        # Jika sampai sini, model dan hparams berhasil dimuat untuk model_name ini
        loaded_models_outer_list.append([m.to(device) for m in current_model_runs_list])
        hparams_list_per_model_type.append(current_hparams_model)
        processed_model_names_for_results_test.append(model_name) # Tambahkan nama model yang berhasil
    
    if not all_models_fully_loaded: 
        print(f"Skipping result generation for task {task} due to issues loading all hparams.")
        continue
        
    if not loaded_models_outer_list: 
        print(f"No models were successfully loaded for evaluation for task {task}. Skipping.")
        continue

    results_dir_for_task = osp.join(root_dir, 'scores', task)
    os.makedirs(results_dir_for_task, exist_ok=True)

    if s_test_split_name not in manifest:
        print(f"Error: Test split '{s_test_split_name}' not found in manifest.json for task '{task}'. Skipping result computation.")
        continue

    print(f"Running Results_test for task {task} with test split {s_test_split_name}...")
    try:
        results_tuple = metrics.Results_test(
            device,
            loaded_models_outer_list, 
            hparams_list_per_model_type, 
            coef_norm, # Gunakan coef_norm yang sudah dimuat/dihitung
            data_dir, 
            results_dir_for_task, 
            model_names_list=processed_model_names_for_results_test, # Gunakan list nama model yang sudah diproses
            n_test=3, 
            criterion='MSE',
            s=s_test_split_name
        )
    except Exception as e:
        print(f"Error during metrics.Results_test for task {task}: {e}")
        import traceback
        traceback.print_exc()
        continue

    (true_coefs_np, pred_coefs_mean_np, pred_coefs_std_np,
     true_surf_coefs_np_obj, pred_surf_coefs_np_obj,
     true_bls_np_obj, pred_bls_np_obj) = results_tuple

    np.save(osp.join(results_dir_for_task, 'true_force_coeffs.npy'), true_coefs_np)
    np.save(osp.join(results_dir_for_task, 'pred_force_coeffs_mean_over_runs.npy'), pred_coefs_mean_np)
    np.save(osp.join(results_dir_for_task, 'pred_force_coeffs_std_over_runs.npy'), pred_coefs_std_np)
    
    np.save(osp.join(results_dir_for_task, 'true_surf_coeffs_selected_cases.npy'), true_surf_coefs_np_obj)
    np.save(osp.join(results_dir_for_task, 'pred_surf_coeffs_selected_cases_mean_runs.npy'), pred_surf_coefs_np_obj)
    np.save(osp.join(results_dir_for_task, 'true_boundary_layers_selected_cases.npy'), true_bls_np_obj)
    np.save(osp.join(results_dir_for_task, 'pred_boundary_layers_selected_cases_mean_runs.npy'), pred_bls_np_obj)

    print(f"Results for task {task} saved in {results_dir_for_task}")

print("All tasks processed.")