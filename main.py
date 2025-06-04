# main.py
import argparse, yaml, os, json, glob
import torch
import train, metrics
from dataset import Dataset
import os.path as osp

import numpy as np # Pastikan numpy diimpor jika belum

parser = argparse.ArgumentParser()
parser.add_argument('model', help = 'The model you want to train, choose between MLP, GraphSAGE, PointNet, GUNet, GAT', type = str)
parser.add_argument('-n', '--nmodel', help = 'Number of trained models for standard deviation estimation (default: 1)', default = 1, type = int)
parser.add_argument('-w', '--weight', help = 'Weight in front of the surface loss (default: 1)', default = 1, type = float)
parser.add_argument('-t', '--task', help = 'Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)', default = 'full', type = str)
parser.add_argument('-s', '--score', help = 'If you want to compute the score of the models on the associated test set. (default: 0)', default = 0, type = int)
args = parser.parse_args()

manifest_file = osp.join('Dataset', 'manifest.json')
if not osp.exists(manifest_file):
    print(f"Error: {manifest_file} not found. Please ensure the path is correct.")
    exit()

with open(manifest_file, 'r') as f:
    manifest = json.load(f)

if args.task + '_train' not in manifest or \
   (args.task + '_test' not in manifest and args.task != 'scarce') or \
   ('full_test' not in manifest and args.task == 'scarce'):
    print(f"Error: Task '{args.task}' splits not found in {manifest_file}. Available keys: {list(manifest.keys())}")
    exit()
    
manifest_train = manifest[args.task + '_train']
test_dataset_names = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
n = int(.1*len(manifest_train))
train_dataset_names = manifest_train[:-n]
val_dataset_names = manifest_train[-n:]

if not train_dataset_names:
    print(f"Error: Training dataset for task '{args.task}' is empty.")
    exit()

# Hitung coef_norm dari train_dataset_names
train_dataset_list, coef_norm = Dataset(train_dataset_names, norm = True, sample = None)

# >>> MODIFIKASI DIMULAI DI SINI <<<
# Simpan coef_norm ke file.
# Nama file akan menyertakan nama task agar unik jika coef_norm berbeda per task.
COEF_NORM_SAVE_DIR = 'Dataset' # Simpan di direktori Dataset
os.makedirs(COEF_NORM_SAVE_DIR, exist_ok=True) # Buat direktori jika belum ada
coef_norm_save_path = osp.join(COEF_NORM_SAVE_DIR, f'normalization_coefficients_{args.task}.pt')

# coef_norm adalah tuple dari numpy array. torch.save bisa menanganinya.
try:
    torch.save(coef_norm, coef_norm_save_path)
    print(f"Normalization coefficients for task '{args.task}' saved to {coef_norm_save_path}")
except Exception as e:
    print(f"Error saving normalization coefficients: {e}")
# >>> MODIFIKASI SELESAI DI SINI <<<

if val_dataset_names:
    val_dataset_list = Dataset(val_dataset_names, sample = None, coef_norm = coef_norm)
else:
    val_dataset_list = [] 
    print(f"Warning: Validation dataset for task '{args.task}' is empty.")


# Cuda
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

params_file = 'params.yaml'
if not osp.exists(params_file):
    print(f"Error: {params_file} not found. Please ensure the path is correct.")
    exit()
with open(params_file, 'r') as f: 
    hparams_all = yaml.safe_load(f)
    if args.model not in hparams_all:
        print(f"Error: Hyperparameters for model '{args.model}' not found in {params_file}.")
        exit()
    hparams = hparams_all[args.model]


from models.MLP import MLP 
models_list = [] 
for i in range(args.nmodel):
    if 'encoder' not in hparams or 'decoder' not in hparams:
        print(f"Error: Encoder/Decoder hyperparameters missing for model '{args.model}' in {params_file}.")
        exit()
        
    encoder = MLP(hparams['encoder'], batch_norm = False) 
    decoder = MLP(hparams['decoder'], batch_norm = False)

    model_instance = None # Inisialisasi model_instance
    if args.model == 'GraphSAGE':
        from models.GraphSAGE import GraphSAGE
        model_instance = GraphSAGE(hparams, encoder, decoder)
    elif args.model == 'PointNet':
        from models.PointNet import PointNet
        model_instance = PointNet(hparams, encoder, decoder)
    elif args.model == 'MLP':
        from models.NN import NN 
        model_instance = NN(hparams, encoder, decoder)
    elif args.model == 'GUNet':
        from models.GUNet import GUNet
        model_instance = GUNet(hparams, encoder, decoder)
    elif args.model == 'GAT': 
        from models.GAT import GAT 
        model_instance = GAT(hparams, encoder, decoder)   
    else: 
        print(f"Error: Model '{args.model}' is not recognized. Choose from MLP, GraphSAGE, PointNet, GUNet, GAT.")
        exit()

    log_path = osp.join('metrics', args.task, args.model) 
    trained_model = train.main(device, train_dataset_list, val_dataset_list, model_instance, hparams, log_path, 
                criterion = 'MSE_weighted', val_iter = 10, reg = args.weight, name_mod = args.model, val_sample = True)
    models_list.append(trained_model)

model_save_dir = osp.join('metrics', args.task, args.model)
os.makedirs(model_save_dir, exist_ok=True)
torch.save(models_list, osp.join(model_save_dir, args.model)) 

if bool(args.score):
    score_task_dir = osp.join('scores', args.task)
    os.makedirs(score_task_dir, exist_ok=True)
    
    s_test_split = args.task + '_test' if args.task != 'scarce' else 'full_test'
    
    if s_test_split not in manifest:
        print(f"Error: Test split '{s_test_split}' for task '{args.task}' not found in manifest.json for scoring.")
    else:
        # Untuk scoring, kita juga perlu coef_norm yang sama dengan training.
        # coef_norm sudah dihitung di atas dari train_dataset_names.
        print(f"Running scoring for task {args.task} on test split {s_test_split} using saved/calculated normalization...")
        try:
            coefs = metrics.Results_test(
                device, 
                [models_list], 
                [hparams],     
                coef_norm, # Gunakan coef_norm yang sudah dihitung/dimuat
                path_in='Dataset', 
                path_out=score_task_dir, 
                model_names_list=[args.model], 
                n_test=3, 
                criterion='MSE', 
                s=s_test_split
            )
            
            (true_coefs_np, pred_coefs_mean_np, pred_coefs_std_np,
             true_surf_coefs_np_obj, pred_surf_coefs_np_obj,
             true_bls_np_obj, pred_bls_np_obj) = coefs

            np.save(osp.join(score_task_dir, 'true_force_coeffs.npy'), true_coefs_np)
            np.save(osp.join(score_task_dir, 'pred_force_coeffs_mean_over_runs.npy'), pred_coefs_mean_np)
            np.save(osp.join(score_task_dir, 'pred_force_coeffs_std_over_runs.npy'), pred_coefs_std_np)
            
            np.save(osp.join(score_task_dir, 'true_surf_coeffs_selected_cases.npy'), true_surf_coefs_np_obj)
            np.save(osp.join(score_task_dir, 'pred_surf_coeffs_selected_cases_mean_runs.npy'), pred_surf_coefs_np_obj)
            np.save(osp.join(score_task_dir, 'true_boundary_layers_selected_cases.npy'), true_bls_np_obj)
            np.save(osp.join(score_task_dir, 'pred_boundary_layers_selected_cases_mean_runs.npy'), pred_bls_np_obj)
            print(f"Scoring results saved in {score_task_dir}")

        except Exception as e:
            print(f"Error during scoring for task {args.task}: {e}")
            import traceback
            traceback.print_exc()