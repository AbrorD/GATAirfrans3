import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from naca_generator import camber_line

sns.set()

# Properties of air at sea level and 293.15K
RHO = 1.184
NU = 1.56e-5
C = 346.1
P_ref = 1.013e5

# Di dalam metrics_NACA.py

# P_ref = 1.013e5 # Anda mungkin perlu ini jika tekanan adalah absolut

def surface_coefficients(airfoil, aero_name, compressible=False, extrado=False, u_inf_val=None, p_ref_val=0.0): # Tambah argumen u_inf_val dan p_ref_val
    if u_inf_val is None: # Jika u_inf_val tidak diberikan, coba parsing dari nama
        try:
            # Coba parsing Uinf dari format "Name_U<nilai>_A<nilai>"
            parts = aero_name.split('_')
            u_inf_str = None
            for part in parts:
                if part.startswith('U') and len(part) > 1:
                    u_inf_str = part[1:] # Ambil angka setelah 'U'
                    break
            if u_inf_str is None: # Fallback ke parsing lama jika format 'U<nilai>' tidak ada
                 u_inf_str = parts[parts.index('U') + 1] if 'U' in parts else parts[2] # Parsing lama atau posisi default

            u_inf = float(u_inf_str)
        except (ValueError, IndexError, TypeError) as e:
            raise ValueError(f"Gagal mem-parsing U_inf dari aero_name='{aero_name}'. Error: {e}. Harap berikan u_inf_val secara eksplisit.")
    else:
        u_inf = u_inf_val # Gunakan nilai yang diberikan

    digits = [] # Parsing digits mungkin tidak lagi relevan jika camber line tidak digunakan di sini
    # ... (logika parsing digits jika masih perlu) ...

    # RHO dan C tetap global atau bisa jadi argumen juga
    # Untuk qInf, pastikan rho digunakan dengan benar
    rho_val = RHO if compressible else 1.0 # Jika incompressible, rho sering diasumsikan 1 dalam qInf non-dimensi
                                          # Jika p adalah p/(rho U^2), maka qInf harus 0.5
                                          # Jika p adalah tekanan fisik, qInf harus 0.5 * rho * U^2
    
    qInf = 0.5 * rho_val * u_inf**2 
    if qInf == 0: qInf = 1e-9 # Hindari pembagian dengan nol

    if extrado:
        # ... (logika extrado, pastikan digits ada jika perlu) ...
        pass

    points = airfoil.points[:, 0]
    if 'p' not in airfoil.point_data:
        raise KeyError("'p' field not found in airfoil point_data for Cp calculation.")
    pressure = airfoil.point_data['p'] # Ini adalah p_static lokal

    # Definisi Cp standar: (p_lokal - p_referensi_jauh) / q_inf
    # Jika pressure Anda adalah p_gauge relatif terhadap p_jauh = 0, maka p_ref_val = 0
    # Jika pressure Anda adalah p_absolut, maka p_ref_val harus p_absolut_jauh
    
    cp_values = (pressure - p_ref_val) / qInf
    c_p = np.concatenate([points[:, None], cp_values[:, None]], axis=1)

    # Perhitungan Cf (Skin Friction Coefficient)
    cf_values = np.array([]) # Default jika tidak ada WSS
    if 'wallShearStress' in airfoil.point_data:
        wss_mag = np.linalg.norm(airfoil.point_data['wallShearStress'][:, :2], axis=1)
        cf_values = wss_mag / qInf
        c_f = np.concatenate([points[:, None], cf_values[:, None]], axis=1)
    else:
        print("Warning (metrics_NACA): 'wallShearStress' not found in airfoil point_data. Cf not calculated.")
        c_f = np.concatenate([points[:, None], np.full_like(points[:, None], np.nan)], axis=1) # Cf diisi NaN


    if extrado and 'idx_extrado' in locals(): # Pastikan idx_extrado ada
        return c_p, c_f, 
    else:
        return c_p, c_f
    
def boundary_layer(airfoil, internal, aero_name, x, y = 1e-3, resolution = int(1e3), direction = 'normals', rotation = False, extrado = True):
    u_inf = float(aero_name.split('_')[2])
    digits = list(map(float, aero_name.split('_')[4:-1]))
    camber = camber_line(digits, airfoil.points[:, 0])[0]
    idx_extrado = (airfoil.points[:, 1] > camber)

    if extrado:
        arg = np.argmin(np.abs(airfoil.points[idx_extrado, 0] - x)) + 1
        arg = np.argwhere(idx_extrado.cumsum() == arg).min()
    else:
        arg = np.argmin(np.abs(airfoil.points[~idx_extrado, 0] - x)) + 1
        arg = np.argwhere((~idx_extrado).cumsum() == arg).min()

    if direction == 'normals':
        normals = -airfoil.point_data['Normals'][arg]
    
    elif direction == 'y':
        normals = np.array([0, 2*int(extrado) - 1, 0])
    
    a, b = airfoil.points[arg], airfoil.points[arg] + y*normals
    bl = internal.sample_over_line(a, b, resolution = resolution)
    
    if rotation:
        rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        u = (bl.point_data['U']*(rot@normals)).sum(axis = 1)
        v = (bl.point_data['U']*normals).sum(axis = 1)
    else:
        u = bl.point_data['U'][:, 0]
        v = bl.point_data['U'][:, 1]
    
    nut = bl.point_data['nut']
    yc = bl.points[:, 1] - a[1]

    return yc, u/u_inf, v/u_inf, nut/NU

def compare_boundary_layer(coefs1, coefs2, ylim = .1, path = None, ylog = False):
    yc1, u1, v1, nut1 = coefs1
    yc2, u2, v2, nut2 = coefs2

    fig, ax = plt.subplots(1, 3, figsize = (30, 10))
    ax[0].scatter(u1, yc1, label = 'Experiment 1')
    ax[0].scatter(u2, yc2, label = 'Experiment 2', color = 'r', marker = 'x')
    ax[0].set_xlabel(r'$u/U_\infty$')
    ax[0].set_ylabel(r'$(y-y_0)/c$')
    # ax[0].set_xlim([-0.2, 1.4])
    # ax[0].set_ylim([0, ylim])
    ax[0].legend(loc = 'best')

    ax[1].scatter(v1, yc1, label = 'Experiment 1')
    ax[1].scatter(v2, yc2, label = 'Experiment 2', color = 'r', marker = 'x')
    ax[1].set_xlabel(r'$v/U_\infty$')
    ax[1].set_ylabel(r'$(y-y_0)/c$')
    # ax[1].set_xlim([-0.2, 0.2])
    # ax[1].set_ylim([0, ylim])
    ax[1].legend(loc = 'best')

    ax[2].scatter(nut1, yc1, label = 'Experience 1')
    ax[2].scatter(nut2, yc2, label = 'Experience 2', color = 'r', marker = 'x')
    # ax[2].set_ylim([0, ylim])
    ax[2].set_xlabel(r'$\nu_t/\nu$')
    ax[2].set_ylabel(r'$(y-y_0)/c$')
    ax[2].legend(loc = 'best')

    if ylog:
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[2].set_yscale('log')

    if path != None:
        fig.savefig(path + 'boundary_layer.png', bbox_inches = 'tight', dpi = 150)

def plot_residuals(path, params):
    datas = dict()
    if params['turbulence'] == 'SA':
        fields = ['Ux', 'Uy', 'p', 'nuTilda']
    elif params['turbulence'] == 'SST':
        fields = ['Ux', 'Uy', 'p', 'k', 'omega']
    for field in fields:
        data = np.loadtxt(path + 'logs/' + field +'_0')[:, 1]
        datas[field] = data

    if params['turbulence'] == 'SA':
        fig, ax = plt.subplots(2, 2, figsize = (20, 20))
        ax[1, 1].plot(datas['nuTilda'])
        ax[1, 1].set_yscale('log')
        ax[1, 1].set_title('nuTilda residual')
        ax[1, 1].set_xlabel('Number of iterations')

    elif params['turbulence'] == 'SST':
        fig, ax = plt.subplots(3, 2, figsize = (30, 20))
        ax[1, 1].plot(datas['k'])
        ax[1, 1].set_yscale('log')
        ax[1, 1].set_title('k residual')
        ax[1, 1].set_xlabel('Number of iterations')

        ax[2, 0].plot(datas['omega'])
        ax[2, 0].set_yscale('log')
        ax[2, 0].set_title('omega residual')
        ax[2, 0].set_xlabel('Number of iterations');
    
    ax[0, 0].plot(datas['Ux'])
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_title('Ux residual')

    ax[0, 1].plot(datas['Uy'])
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_title('Uy residual')

    ax[1, 0].plot(datas['p'])
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_title('p residual')
    ax[1, 0].set_xlabel('Number of iterations');

    fig.savefig(path + 'residuals.png', bbox_inches = 'tight', dpi = 150)

    return datas

def plot_coef_convergence(path, params):
    datas = dict()
    datas['c_d'] = np.loadtxt(path + 'postProcessing/forceCoeffs1/0/coefficient.dat')[:, 1]
    datas['c_l'] = np.loadtxt(path + 'postProcessing/forceCoeffs1/0/coefficient.dat')[:, 3]
    c_d, c_l = datas['c_d'][-1], datas['c_l'][-1]

    fig, ax = plt.subplots(2, figsize = (30, 15))
    ax[0].plot(datas['c_d'])
    ax[0].set_ylim([.5*c_d, 1.5*c_d])
    ax[0].set_title('Drag coefficient')
    ax[0].set_xlabel('Number of iterations')
    ax[0].set_ylabel(r'$C_D$')

    ax[1].plot(datas['c_l'])
    ax[1].set_title('Lift coefficient')
    ax[1].set_ylim([.5*c_l, 1.5*c_l])
    ax[1].set_ylabel(r'$C_L$')
    ax[1].set_xlabel('Number of iterations');

    print('Drag coefficient: {0:.5}, lift coefficient: {1:.5}'.format(c_d, c_l))

    fig.savefig(path + 'coef_convergence.png', bbox_inches = 'tight', dpi = 150)

    return datas, c_d, c_l