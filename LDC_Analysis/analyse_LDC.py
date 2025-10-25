"""
SPT Load-Deflection Curve Analysis

Extracts characteristic features from SPT load-deflection curves (LDC) and
analyzes them according to EN 10371 and ASTM E3205-20 standards.

IMPORTANT:
    - These functions assume execution inside an Abaqus Python environment
      (Abaqus/CAE or abaqus python) with access to `mdb`, `session`, `odbAccess`
    - Abaqus typically runs Python 2.7

Key features extracted:
    - Inflection points (yield, elastic-plastic transition, plastic instability)
    - Slope values for different deformation stages
    - Force values at specific deflections (EN 10371, ASTM E3205-20)

Author: Saleem Lubbad
Original comments and annotations were cleaned by ChatGPT, under Oxford University license.
"""



def analyse_LDC_(
        deflection, force, thickness,
        Slope_IV_deviation, save_dir, prefix, plot):
    """
    Analyse SPT load–deflection curves (simulation) and extract key points/slopes.

    Parameters
    ----------
    deflection : array-like
        Deflection arrays per simulation (shape: N x M or list of arrays).
    force : array-like
        Force arrays per simulation (shape: N x M or list of arrays).
    thickness : float
        Sample thickness (used for stage boundaries).
    Slope_IV_deviation : float
        Threshold (relative) to detect onset of Stage IV deviation.
    save_dir : str or None
        Directory to save figures; if None or empty, figures are not saved.
    prefix : str
        Prefix used to label each simulated sample (e.g., material name).
    plot : bool
        If True, generate and display/save plots.

    Returns
    -------
    dict
        Dictionary with extracted metrics:
        Fm, um, Fy, Fy_curv, uy, FB, uB, Fe, Fep, Fy_10, Fy_100, ue,
        Fi, Fi_std, ui, slope_I, slope_II, slope_III, slope_IV, slope_0
    """
    Fy, uy, Fy_crv = [], [], []
    FB, uB = [], []
    Fe, ue = [], []
    Fep = []
    Fy_10 = []
    Fy_100 = []
    Fi, Fi_standard, ui = [], [], []
    Fm, um = [], []
    us, Fs = [], []

    ui_std = 0.552  # Table C.1 prEN 10371:2019 (E) – SPT standard (reference deflection)
    Slope_0, Slope_I, Slope_II, Slope_III, Slope_IV = [], [], [], [], []
    anaomolies_indx = []

    try:
        no_simulations = force.shape[1]
    except AttributeError:
        no_simulations = len(force)

    print(f'Number of Experiments = {no_simulations}')

    for i in range(no_simulations):
        material_name = f'{prefix}-{i+1}'

        # Maximum deflection and force
        uB_val = thickness
        print(f'Sample Name: {i+1}')
        try:
            defl = deflection[:, i]
            f = abs(force[:, i])
        except TypeError:
            defl = np.array(deflection[i])
            f = abs(np.array(force[i]))

        def_max = np.max(defl)

        Fmi = np.nanmax(f)
        Fm.append(Fmi)
        max_indices = np.where(f == Fmi)[0][0]
        umi = defl[max_indices]
        um.append(umi)

        try:
            uB_indx = np.where(defl >= thickness)[0][0]
            end_indx = np.where(defl >= uB_val * 1.5)[0][0]
            FB_val = f[uB_indx]
            FB.append(FB_val)
            uB.append(uB_val)
        except IndexError:
            print(f'Experiment {i+1} does not go beyond sample thickness.')
            FB_val = f[-1]
            uB_val = defl[-1]
            uB_indx = np.where(defl >= uB_val)[0][0]
            end_indx = np.where(defl == max(defl))[0][0]
            FB.append(FB_val)
            uB.append(uB_val)

        # Stage I: Elastic–plastic transition (Fe)
        x_data_1 = defl[:uB_indx]
        y_data_1 = f[:uB_indx]

        initial_guess = [uB_val / 10, FB_val / 2]
        bounds = [(0, uB_val / 2), (0, FB_val)]

        ue_val, Fa_val = optimise_bilinear(
            x_data_1, y_data_1,
            uB_val, FB_val,
            initial_guess, bounds)

        fitted_force_1 = bilinear_model(
            np.linspace(0, max(x_data_1), 100), ue_val,
            Fa_val, FB_val, uB_val)

        Fe_val = np.interp(ue_val, x_data_1, y_data_1)
        Fi_std = np.interp(ui_std, defl, f)
        Fi_standard.append(Fi_std)
        ue.append(ue_val)
        Fep.append(Fe_val)
        Fe.append(Fa_val)

        initial_slope = Fe_val / ue_val
        Slope_I.append(initial_slope)
        slope_II = (FB_val - Fe_val) / (uB_val - ue_val)
        Slope_II.append(slope_II)

        # Stage 0: Yield onset (Fy) before ue
        ue_indx = np.where(x_data_1 >= ue_val)[0][0]
        x_data_0 = defl[:ue_indx]
        y_data_0 = f[:ue_indx]

        initial_guess = [ue_val / 2, Fe_val / 2]
        bounds = [(0, Fe_val), (0, Fe_val)]

        uy_val, Fy_val = optimise_bilinear(
            x_data_0, y_data_0,
            ue_val, Fe_val,
            initial_guess, bounds)

        fitted_force_0 = bilinear_model(
            np.linspace(0, max(x_data_0), 100), uy_val, Fy_val,
            Fe_val, ue_val)

        Fy_val = np.interp(uy_val, x_data_0, y_data_0)
        Fy.append(Fy_val)
        uy.append(uy_val)
        initial_slope = Fy_val / uy_val
        Slope_0.append(initial_slope)

        # t/10 offset yield estimate
        projection_x = x_data_1
        projection_y = np.where(
            x_data_1 < thickness / 10, 0,
            initial_slope * (x_data_1 - thickness / 10))

        y1 = projection_y[np.where(projection_x >= thickness / 10)[0][0]:]
        y2 = y_data_1[np.where(x_data_1 >= thickness / 10)[0][0]:]
        difference = np.array(abs(y1 - y2))
        min_difference = np.min(difference)
        min_diff_indx = np.where(difference == min_difference)
        min_diff_indx = min_diff_indx + np.where(x_data_1 >= thickness / 10)[0][0]

        intersection_x = x_data_1[min_diff_indx][0][0]
        intersection_y = y_data_1[min_diff_indx][0][0]
        Fy_10.append(intersection_y)

        if plot:
            print(f"Intersection by grid search t/10: index = {min_diff_indx} , x = {intersection_x}, y = {intersection_y}")

        if defl[-1] > uB_val:
            # Stage III: Plastic instability (Fi) + Slope_III
            if defl[-1] > uB_val * 1.5:
                zoneIII_limit = 1.75 * thickness
                if defl[-1] > zoneIII_limit:
                    zoneIII_limit_indx = np.where(defl >= zoneIII_limit)[0][0]
                    x_data_3 = defl[:zoneIII_limit_indx]
                    y_data_3 = f[:zoneIII_limit_indx]
                else:
                    x_data_3 = defl
                    y_data_3 = f

                initial_guess = [ue_val, thickness * 1.1]
                bounds = [(0.99 * ue_val, 1.01 * ue_val), (thickness, thickness * 1.2)]

                u_e, f_e, ui_val, Fi_val = optimise_trilinear(
                    x_data_3, y_data_3, initial_guess, bounds)

                fitted_force_2 = trilinear_model(
                    x_data_3, u_e, f_e, ui_val, Fi_val, x_data_3[-1], y_data_3[-1])

                ui.append(ui_val)
                Fi.append(Fi_val)

                slope_III = (y_data_3[-1] - Fi_val) / (x_data_3[-1] - ui_val)
                Slope_III.append(slope_III)

            # Stage IV: Detect deviation from projected Slope_III
            start_indx = np.where(defl >= ui_val)[0][0]
            end_indx = np.where(defl == umi)[0][0]

            f_II = f[start_indx:end_indx]
            defl_II = defl[start_indx:end_indx]
            f_proj = Fi_val + slope_III * (defl_II - ui_val)

            deviation = np.abs(f_II - f_proj) / np.maximum(f_proj, 1e-6)
            threshold = Slope_IV_deviation
            min_consecutive = int(len(deviation) * 0.5)

            y = deviation
            x = defl_II
            above_thresh = y > threshold

            for i_w in range(len(above_thresh) - min_consecutive + 1):
                if np.all(above_thresh[i_w:i_w + min_consecutive]):
                    sustained_start_idx = i_w
                    sustained_x = x[sustained_start_idx]
                    sustained_y = y[sustained_start_idx]
                    if plot:
                        print(f"Sustained deviation starts at deflection = {sustained_x:.4f}, deviation = {sustained_y:.4f}")
                    break
            else:
                print("No sustained deviation found above the threshold.")

            if plot:
                plt.scatter(x, y, label=f'Slope IV Deviation_{material_name}', s=1)
                if 'sustained_start_idx' in locals():
                    plt.axvline(x[sustained_start_idx], color='red', linestyle='--', label="Start of sustained deviation")
                    plt.scatter([x[sustained_start_idx]], [y[sustained_start_idx]], color='red')
                plt.title("Sustained Deviation Detection")
                plt.xlabel("Deflection")
                plt.ylabel("Deviation")
                plt.legend()
                plt.grid(True)
                plt.show()

            us_indx = sustained_start_idx
            us_val = defl_II[us_indx]
            Fs_val = f_II[us_indx]

            us.append(us_val)
            Fs.append(Fs_val)

            delta_u = us_val * 1.2 - us_val
            delta_f = f[np.where(defl >= us_val * 1.2)[0][0]] - Fs_val
            slope_IV = delta_f / delta_u if delta_u != 0 else np.nan
            Slope_IV.append(slope_IV)

        else:
            print('Experiment does not go beyond sample thickness')

            ui_val = uB_val
            us_val = uB_val
            Fi_val = FB_val
            Fs_val = FB_val

            ui.append(uB_val)
            Fi.append(uB_val)

            slope_III = slope_II
            Slope_III.append(slope_III)

            us.append(uB_val)
            Fs.append(uB_val)

            slope_IV = slope_II
            Slope_IV.append(slope_IV)

        if plot:
            plt.figure(figsize=(10, 8))
            max_def_idx = np.where(defl <= umi)[0]
            defl_plot = defl[max_def_idx]
            f_plot = f[max_def_idx]
            plt.plot(defl_plot, f_plot, linewidth=0.75, color="blue", label=f'Simulation: {material_name}')

            # Slope 0
            slope_line_u = np.linspace(0, ue_val, 100)
            slope_line_f = initial_slope * (slope_line_u - 0)
            plt.plot(slope_line_u, slope_line_f, linestyle="--", color="green", label="$Slope_{0}$")

            # Bilinear fit (EN 10371)
            x_fit = np.linspace(0, max(x_data_1), 100)
            plt.plot(x_fit, fitted_force_1, color="red", label="Bilinear Fit [EN 10371]")

            # Slope II
            slope_line_u = np.linspace(uB_val, ui_val * 1.5, 100)
            slope_line_f = Fe_val + slope_II * (slope_line_u - ue_val)
            plt.plot(slope_line_u, slope_line_f, linestyle="--", color="red", label="$Slope_{II}$")

            # Slope III
            if not np.isnan(ui_val) and not np.isnan(Fi_val):
                u_line = np.linspace(0.75 * ui_val, umi, 50)
                f_line = Fi_val + slope_III * (u_line - ui_val)
                plt.plot(u_line, f_line, linestyle="--", linewidth=1.0, color='purple', label="$Slope_{III}$")

            # Slope IV
            if not np.isnan(us_val) and not np.isnan(Fs_val):
                u_line = np.linspace(ui_val, umi, 50)
                f_line = Fs_val + slope_IV * (u_line - us_val)
                plt.plot(u_line, f_line, linestyle="--", linewidth=1.0, color='black', label="$Slope_{IV}$")

            # t/10 Offset
            offset_start = np.where(projection_x >= thickness / 10)[0][0]
            offset_end = np.where(projection_x >= intersection_x)[0][0]
            plt.plot(projection_x[offset_start:offset_end], projection_y[offset_start:offset_end],
                     linewidth=0.75, linestyle="--", color="blue", label="t/10 Offset Method")

            # Yield (t/10)
            plt.scatter(intersection_x, intersection_y, color="blue", marker="*", label="$F_{y,t/10}$")
            plt.hlines(intersection_y, 0, intersection_x, linestyle="dotted", color="blue")
            plt.annotate(f'$F_{{y,t/10}} = {intersection_y:.0f}$', xy=(0, intersection_y),
                         xytext=(-0.1, intersection_y * 1.02), fontsize=10, color='blue')

            # Fe (EN 10371)
            plt.scatter([ue_val], [Fa_val], color="red", marker="*", label="$F_e$ (EN 10371)")
            plt.vlines(ue_val, 0, Fa_val, linestyle="dotted", color="red")
            plt.hlines(Fa_val, 0, ue_val, linestyle="dotted", color="red")
            plt.annotate(f'$F_e = {Fa_val:.0f}$', xy=(0, Fa_val), xytext=(-0.1, Fa_val * 1.02), fontsize=10, color='red')
            plt.annotate(f'$u_e = {ue_val:.2f}$', xy=(ue_val, 0), xytext=(ue_val * 1.01, -20),
                         fontsize=10, color='red', rotation=90)

            # Fep (ASTM)
            plt.scatter([ue_val], [Fe_val], color="green", marker="*", label="$F_{ep}$ (ASTM E3205-20)")
            plt.hlines(Fe_val, 0, ue_val, linestyle="dotted", color="green")
            plt.annotate(f'$F_{{ep}} = {Fe_val:.0f}$', xy=(0, Fe_val), xytext=(-0.1, Fe_val * 0.8),
                         fontsize=10, color='green')

            # Fi_std (EN 10371)
            plt.scatter([ui_std], [Fi_std], color="black", marker="*", label="$F_i^{std}$ (EN 10371)")
            plt.vlines(ui_std, Fi_std, plt.ylim()[1], linestyle="dotted", color="black")
            plt.hlines(Fi_std, plt.xlim()[1] * 0.85, ui_std, linestyle="dotted", color="black")
            plt.annotate(f'$F_i^{{std}} = {Fi_std:.0f}$', xy=(ui_std, Fi_std),
                         xytext=(plt.xlim()[1] * 0.8, Fi_std * 1.05), fontsize=10, color='black')
            plt.annotate(f'$u_i^{{std}} = {ui_std:.3f}$', xy=(ui_std, 0),
                         xytext=(ui_std * 1.01, plt.ylim()[1] * 0.8),
                         fontsize=10, color='black', rotation=90, ha='left')

            # Fi detected
            if not (np.isnan(ui_val) or np.isnan(Fi_val)):
                plt.scatter([ui_val], [Fi_val], color="purple", marker="*", label="$F_i$ ($2^{nd}$ Inflexion Point)")
                plt.vlines(ui_val, 0, Fi_val, linestyle="dotted", color="purple")
                plt.hlines(Fi_val, 0, ui_val, linestyle="dotted", color="purple")
                plt.annotate(f'$F_i = {Fi_val:.0f}$', xy=(0, Fi_val),
                             xytext=(-0.1, Fi_val * 0.97), fontsize=10, color='purple')
                plt.annotate(f'$u_i = {ui_val:.2f}$', xy=(ui_val, 0),
                             xytext=(ui_val * 1.01, -20), fontsize=10, color='purple', rotation=90)

            # Fs detected
            if not (np.isnan(us_val) or np.isnan(Fs_val)):
                plt.scatter([us_val], [Fs_val], color="#D55E00", marker="*", label="$F_s$ ($3^{rd}$ Inflexion Point)")
                plt.vlines(us_val, 0, Fs_val, linestyle="dotted", color="#D55E00")
                plt.hlines(Fs_val, 0, us_val, linestyle="dotted", color="#D55E00")
                plt.annotate(f'$F_s = {Fs_val:.0f}$', xy=(0, Fs_val),
                             xytext=(-0.1, Fs_val * 0.97), fontsize=10, color='#D55E00')
                plt.annotate(f'$u_s = {us_val:.2f}$', xy=(us_val, 0),
                             xytext=(us_val * 1.01, -20), fontsize=10, color='#D55E00', rotation=90)

            # Final formatting
            plt.title("SPT Load-Deflection Curve Analysis", fontsize=14)
            plt.xlabel("Deflection [mm]", fontsize=12)
            plt.ylabel("Force [N]", fontsize=12)
            plt.grid(True)

            # Legend ordering
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [labels.index(l) for l in sorted(labels, key=lambda x: (
                0 if 'Experimental' in x else
                1 if 'Bilinear' in x else
                2 if '$F' in x else
                3 if 'Slope' in x else
                4))]
            handles = [handles[i] for i in order]
            labels = [labels[i] for i in order]
            plt.legend(handles, labels, loc="best", fontsize=9)

            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/{material_name}_LDC_full_analysis.png")
                print(f'Plot saved to {save_dir}')
            else:
                print('Plot not saved')
            plt.show()

    LDC_data = {
        'Fm': Fm,
        'um': um,
        'Fy': Fy,
        'Fy_curv': Fy_crv,
        'uy': uy,
        'FB': FB,
        'uB': uB,
        'Fe': Fe,
        'Fep': Fep,
        'Fy_10': Fy_10,
        'Fy_100': Fy_100,
        'ue': ue,
        'Fi': Fi,
        'Fi_std': Fi_standard,
        'ui': ui,
        'slope_I': Slope_I,
        'slope_II': Slope_II,
        'slope_III': Slope_III,
        'slope_IV': Slope_IV,
        'slope_0': Slope_0
    }

    return LDC_data
