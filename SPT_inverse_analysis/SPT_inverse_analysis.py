"""
SPT Inverse Analysis Module

Automates inverse material parameter identification for the Small Punch Test (SPT)
using Abaqus finite-element simulations and optimization methods.

IMPORTANT:
    - These functions assume execution inside an Abaqus Python environment
      (Abaqus/CAE or abaqus python) with access to `mdb`, `session`, `odbAccess`
    - Abaqus typically runs Python 2.7
    - Some functions depend on FE_utility/selected_FE_utility_functions.py

Supported optimization methods:
    - Nelder-Mead (validated)
    - L-BFGS-B
    - Bayesian (GP) minimization

Core routines include:
    - Parameter scaling and unscaling
    - Objective function evaluation with regularization and bounds penalties
    - Automatic handling of Abaqus jobs to enable FEMU (Finite Element Model Updating)

Author: Saleem Lubbad
Original comments and annotations were cleaned by ChatGPT, under Oxford University license.
"""

# ===== Required Global Variables =====
# These must be defined before calling the objective functions:
# - parameters: str, one of 'Elastic', 'Elastic and Plastic', 'Plastic', 'Hardening Parameters'
# - thickness: float, sample thickness in mm
# - linearsiation: bool, whether to use linear fitting (optional, defaults to False)
# - NORMALIZATION_FACTORS: dict, parameter scaling factors (optional)

# ===== Imports =====
# Note: run_simulation should be imported from FE_utility.selected_FE_utility_functions
# However, signature mismatch exists - see function definitions for details

# ===== Parameter Scaling Helpers =====
def scale_parameters(params):
    """ Scale parameters using the defined NORMALIZATION_FACTORS dictionary in the global scope.
        If NORMALIZATION_FACTORS is not defined, returns params unchanged.
    """
    global NORMALIZATION_FACTORS
    try:
        return {key: params[key] / NORMALIZATION_FACTORS[key] for key in params}
    except Exception:
        # Return params unchanged if scaling not possible
        return params


def unscale_parameters(params):
    """ Unscale parameters using the defined NORMALIZATION_FACTORS dictionary in the global scope.
        If NORMALIZATION_FACTORS is not defined, returns params unchanged.
    """
    global NORMALIZATION_FACTORS
    try:
        return {key: params[key] * NORMALIZATION_FACTORS[key] for key in params}
    except Exception:
        return params


# ===== Inverse Analysis Orchestrator =====
def run_inverse_analysis(file_path, filter_data, regularization_factor, method, initial_guess, lower_bounds,
    upper_bounds, working_dir, save_dir_path, job_name, material_name):
    """
    Execute inverse analysis workflow for SPT material parameter identification.

    Parameters:
      - file_path: experimental csv path
      - filter_data: bool, whether to filter exp data
      - regularization_factor: float
      - method: string optimizer
      - initial_guess: list
      - lower_bounds, upper_bounds: dicts of parameter bounds
      - working_dir: path used during FE runs
      - save_dir_path: path to store results/outputs
      - job_name, material_name: forwarded to run_simulation and job creation

    Workflow:
      - Prepares directories
      - Loads and optionally filters experimental data
      - Selects optimization routine (Nelder-Mead, L-BFGS-B, or Bayesian)
      - Runs optimization loop with objective function evaluations
      - Saves results, plots, and parameter traces
    """
    # prepare directories
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        print('New path created: ' + working_dir)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
        print('New path created: ' + save_dir_path)

    os.chdir(working_dir)
    print('Working Directory set to: ' + working_dir)
    print('Iteration and plots values are saved to:: ' + working_dir)

    start_time = system_time.time()

    # read experimental data
    threshold = 0
    experimental_force, experimental_def, experimental_time, experimental_dis = read_exp_fv_data(file_path, threshold)

    experimental_force = np.abs(np.array(experimental_force))
    experimental_def = np.abs(np.array(experimental_def))

    if filter_data:
        try:
            from scipy.signal import savgol_filter
            experimental_force = savgol_filter(experimental_force, 101, 3)
        except Exception:
            print("Warning: savgol_filter not available or failed. Using raw data.")

    # prepare bounds for scipy minimize (if applicable)
    # Form bounds list from lower_bounds and upper_bounds dictionaries
    try:
        if isinstance(lower_bounds, dict) and isinstance(upper_bounds, dict):
            param_keys = list(lower_bounds.keys())
            bounds = [ (lower_bounds[k], upper_bounds[k]) for k in param_keys ]
        else:
            # if lower_bounds provided as sequence already use it
            bounds = lower_bounds
            param_keys = None
    except Exception:
        bounds = None
        param_keys = None

    # Choose optimization method
    if method == 'Bayesian':
        # Use Bayesian optimization if skopt is available
        try:
            from skopt import gp_minimize
            use_bayesian = True
        except Exception:
            use_bayesian = False
            print("skopt not available, falling back to SciPy optimize.")
    else:
        use_bayesian = False

    # callback list containers for plotting
    sigma_y0_values, E_values, v_values, R_inf_values, alpha_values, n_values, m_values = [], [], [], [], [], [], []

    # Callback function for optimizer to log parameter iterations
    def callbackF(x):
        # Write parameter traces to CSV and collect values for plotting
        try:
            # Save all parameter values to CSV for this optimization method
            csv_path = os.path.join(working_dir, 'params_trace_{}.csv'.format(method))
            with open(csv_path, "ab") as f:
                writer = csv.writer(f)
                writer.writerow(x)

            # Extract individual parameter values for plotting
            if len(x) >= 1:
                sigma_y0_values.append(x[0])
            if len(x) >= 2:
                E_values.append(x[1])
            if len(x) >= 3:
                R_inf_values.append(x[2])
            if len(x) >= 4:
                alpha_values.append(x[3])
            if len(x) >= 5:
                n_values.append(x[4])
            if len(x) >= 6:
                m_values.append(x[5])
        except Exception:
            # non-fatal
            traceback.print_exc()

    # Dispatch to appropriate optimization method
    try:
        if use_bayesian:
            # Call gp_minimize with objective function for Bayesian optimization
            from skopt import gp_minimize

            def _to_opt(x):
                # Construct params dict keyed by parameter names (e.g., 'sigma_y0', 'E', etc.)
                if param_keys is None:
                    # Use generic parameter names if keys not available
                    params = {f'p{i}': xi for i, xi in enumerate(x)}
                else:
                    params = {k: x[i] for i, k in enumerate(param_keys)}
                return objective_function(params, experimental_force, experimental_def, experimental_time, filter_data, lower_bounds, upper_bounds, regularization_factor, working_dir, save_dir_path, job_name, material_name)

            # Convert bounds to skopt-compatible dimensions list
            if isinstance(bounds, list):
                space = bounds
            else:
                # Use default unit hypercube search space
                space = [(0.0, 1.0)] * len(initial_guess)

            result = gp_minimize(_to_opt, space, x0=initial_guess, n_calls=50, n_random_starts=10, random_state=42)
            opt_result = result
        else:
            from scipy.optimize import minimize
            def _obj_vec(x):
                if param_keys is None:
                    params = {f'p{i}': x[i] for i in range(len(x))}
                else:
                    params = {param_keys[i]: x[i] for i in range(len(x))}
                return objective_function(params, experimental_force, experimental_def, experimental_time, filter_data, lower_bounds, upper_bounds, regularization_factor, working_dir, save_dir_path, job_name, material_name)

            if method == 'L-BFGS-B' and bounds is not None:
                res = minimize(lambda x: _obj_vec(x), x0=np.array(initial_guess, dtype=float), method='L-BFGS-B', bounds=bounds, callback=callbackF, options={'maxiter':1000})
            else:
                res = minimize(lambda x: _obj_vec(x), x0=np.array(initial_guess, dtype=float), method='Nelder-Mead', callback=callbackF, options={'maxiter':1000, 'disp': True})

            opt_result = res

    except Exception as e:
        print("Inverse analysis optimizer failed: {}".format(e))
        traceback.print_exc()
        opt_result = None

    elapsed = system_time.time() - start_time
    print("Time to completion: {:.2f} seconds".format(elapsed))

    # Save optimization history plots
    try:
        plot_optimization_results(sigma_y0_values, E_values, R_inf_values, alpha_values, n_values, save_dir_path)
    except Exception:
        # Non-fatal: continue if plotting fails
        traceback.print_exc()

    # Save optimizer result summary
    try:
        summary_path = os.path.join(save_dir_path, 'inverse_summary_{}.txt'.format(job_name))
        with open(summary_path, 'w') as f:
            f.write("Optimizer result:\n")
            f.write(str(opt_result))
            f.write("\nElapsed seconds: {}\n".format(elapsed))
    except Exception:
        pass

    return opt_result


# ===== Penalized Objective Function =====
def penalized_objective_function(params, experimental_force, experimental_def, experimental_time, filter_data, regularization_factor, dir_path, job_name, material_name):
    """
    Objective function with parameter bounds penalties.
    
    Steps:
      - Checks parameter bounds and applies penalties
      - Scales parameters using NORMALIZATION_FACTORS
      - Runs FE simulation via run_simulation
      - Computes NRMSE between simulation and experiment
      - Returns cost (NRMSE + penalties)
    
    Warning: May be redundant with objective_function(). Consider consolidating.
    Warning: Calls run_simulation() which has signature mismatch - needs fixing.
    """
    penalty = 0
    # Access global bounds if defined
    try:
        global bounds
    except Exception:
        bounds = None

    # Parameter bounds penalty (if bounds provided as sequence of (lb,ub))
    try:
        if bounds is not None and isinstance(bounds, (list, tuple)):
            # assume params is sequence
            for i, (lower, upper) in enumerate(bounds):
                pval = float(params[i])
                if pval < lower:
                    penalty += (lower - pval) ** 2
                elif pval > upper:
                    penalty += (pval - upper) ** 2
    except Exception:
        pass

    # scale parameters if helper exists
    try:
        scaled_params = scale_parameters(params)
    except Exception:
        scaled_params = params

    # run FE simulation
    try:
        simulation_force, simulation_def, simulation_time, simulation_PE, time_PE = run_simulation(dir_path, job_name, material_name, scaled_params)
    except Exception:
        traceback.print_exc()
        # return large penalty
        return 1e6 + penalty

    if parameters == 'Elastic':
        try:
            Elastic_limit_indx_PE = np.where(simulation_PE > 0)[0][0]
        except Exception:
            Elastic_limit_indx_PE = len(simulation_PE)
        divergence_time = time_PE[Elastic_limit_indx_PE] if len(time_PE) > 0 else simulation_time[-1]
        try:
            Elastic_limit_indx_def = np.where(simulation_time >= divergence_time)[0][0]
        except Exception:
            Elastic_limit_indx_def = len(simulation_time) - 1
        limit_indx = -1
        exp_limit_indx = -1

        simulation_force = simulation_force[:limit_indx]
        simulation_def = simulation_def[:limit_indx]
        simulation_time = simulation_time[:limit_indx]

        experimental_force = experimental_force[:exp_limit_indx]
        experimental_def = experimental_def[:exp_limit_indx]
        experimental_time = experimental_time[:exp_limit_indx]

    elif parameters == 'Elastic and Plastic':
        # Use thickness global to locate slice end
        try:
            exp_def_limi_indx = np.where(np.abs(experimental_def) >= thickness * 1.7)[0][0]
            exp_time_limit = experimental_time[exp_def_limi_indx]
            sim_time_limit_indx = np.where(np.abs(simulation_time) >= exp_time_limit)[0][0]
            exp_limit_indx = exp_def_limi_indx
            limit_indx = sim_time_limit_indx
        except Exception:
            exp_limit_indx = None
            limit_indx = None

        if limit_indx is not None:
            simulation_force = simulation_force[:limit_indx]
            simulation_def = simulation_def[:limit_indx]
            simulation_time = simulation_time[:limit_indx]
        if exp_limit_indx is not None:
            experimental_force = experimental_force[:exp_limit_indx]
            experimental_def = experimental_def[:exp_limit_indx]
            experimental_time = experimental_time[:exp_limit_indx]

    elif parameters in ('Plastic', 'Hardening Parameters'):
        try:
            Elastic_limit_indx_PE = np.where(simulation_PE > 0)[0][0]
        except Exception:
            Elastic_limit_indx_PE = len(simulation_PE)
        divergence_time = time_PE[Elastic_limit_indx_PE] if len(time_PE) > 0 else simulation_time[-1]
        Elastic_limit_indx_def = np.where(simulation_time >= divergence_time)[0][0]
        plastic_start_indx = Elastic_limit_indx_def
        try:
            exp_plastic_start_indx = np.where(experimental_time >= divergence_time)[0][0]
        except Exception:
            exp_plastic_start_indx = 0

        try:
            exp_def_limi_indx = np.where(np.abs(experimental_def) >= thickness * 1.7)[0][0]
            exp_time_limit = experimental_time[exp_def_limi_indx]
            sim_time_limit_indx = np.where(np.abs(simulation_time) >= exp_time_limit)[0][0]
            exp_plastic_end_indx = exp_def_limi_indx
            plastic_end_indx = sim_time_limit_indx
        except Exception:
            exp_plastic_end_indx = len(experimental_def)
            plastic_end_indx = len(simulation_def)

        simulation_force = simulation_force[plastic_start_indx:plastic_end_indx]
        simulation_def = simulation_def[plastic_start_indx:plastic_end_indx]
        simulation_time = simulation_time[plastic_start_indx:plastic_end_indx]

        experimental_force = experimental_force[exp_plastic_start_indx:exp_plastic_end_indx]
        experimental_def = experimental_def[exp_plastic_start_indx:exp_plastic_end_indx]
        experimental_time = experimental_time[exp_plastic_start_indx:exp_plastic_end_indx]

    # Interpolate and compute NRMSE
    try:
        sim_force_interp = np.interp(experimental_def, simulation_def, simulation_force)
    except Exception:
        sim_force_interp = simulation_force
    try:
        rmse_sim = np.sqrt(np.mean((sim_force_interp - experimental_force) ** 2))
    except Exception:
        rmse_sim = 1e6

    nrmse = 100 * (rmse_sim / (np.sum(experimental_force) if np.sum(experimental_force) != 0 else 1.0))
    normalized_difference = nrmse

    # Plot simulation vs experiment if plotting function available
    try:
        plot_simulation_vs_experiment(simulation_def, simulation_force, simulation_def, experimental_force, normalized_difference, params, dir_path)
    except Exception:
        # ignore plotting failures
        pass

    return normalized_difference + penalty


# ===== Main Objective Function =====
def objective_function(params, experimental_force, experimental_def, experimental_time, filter_data, lower_bounds, upper_bounds, regularization_factor, working_dir, save_dir_path, job_name, material_name):
    """
    Main objective function used by the optimizer.
    - Expects params as dict keyed by parameter names (e.g. 'sigma_y0','E','Rinf',...)
    - Calls run_simulation and compares simulation vs experiment using RMSE-like metrics.
    - Returns a scalar cost (nrmse + penalties).
    """
    try:
        # Attempt to scale parameters if helper exists
        try:
            scaled_params = scale_parameters(params)
        except Exception:
            scaled_params = params

        # Scale bounds if they are dicts
        try:
            lower_b = scale_parameters(lower_bounds) if isinstance(lower_bounds, dict) else lower_bounds
            upper_b = scale_parameters(upper_bounds) if isinstance(upper_bounds, dict) else upper_bounds
        except Exception:
            lower_b = lower_bounds
            upper_b = upper_bounds

        # Run finite element simulation with current parameter values
        simulation_force, simulation_def, simulation_time, simulation_PE, time_PE = run_simulation(
            working_dir, job_name, material_name, scaled_params)

        # Print parameter values for debugging
        print('Inverse Analysis Parameters:' + str(parameters))
        print('Parameters Iteration Values _scaled' + str(scaled_params))
        print('Parameters Iteration Values _original' + str(unscale_parameters(scaled_params)))

        # Optional linearisation - if global linearsiation True will fit lines and compare
        if linearsiation:
            from numpy.polynomial.polynomial import Polynomial
            p_exp = Polynomial.fit(experimental_def, experimental_force, 1)
            x_fit = np.linspace(0, np.max(experimental_def), len(experimental_def))
            experimental_force = p_exp(x_fit)
            # Match array lengths for simulation data
            simulation_force = simulation_force[:len(simulation_def)]
            simulation_def = simulation_def[:len(simulation_force)]
            p_sim = Polynomial.fit(simulation_def, simulation_force, 1)
            sim_force_interp = p_sim(x_fit)
            experimental_def = x_fit
        else:
            # Interpolate simulation onto experimental deflection grid
            try:
                sim_force_interp = np.interp(experimental_def, simulation_def, simulation_force)
            except Exception:
                sim_force_interp = simulation_force

        # Compute Euclidean RMSE between simulation and experiment
        def euclidean_rmse(sim_force, sim_def, exp_force, exp_def):
            if len(sim_force) != len(exp_force):
                # Resample both arrays to minimum length
                min_len = min(len(sim_force), len(exp_force))
                sim_def_resampled = np.linspace(sim_def[0], sim_def[-1], min_len)
                sim_force_resampled = np.interp(sim_def_resampled, sim_def, sim_force)
                exp_def_resampled = np.linspace(exp_def[0], exp_def[-1], min_len)
                exp_force_resampled = np.interp(exp_def_resampled, exp_def, exp_force)
                dist_squared = (sim_force_resampled - exp_force_resampled) ** 2 + (sim_def_resampled - exp_def_resampled) ** 2
                return np.sqrt(np.mean(dist_squared))
            else:
                dist_squared = (sim_force - exp_force) ** 2 + (sim_def - exp_def) ** 2
                return np.sqrt(np.mean(dist_squared))

        # Build interpolated arrays
        try:
            sim_def_interp = np.linspace(simulation_def[0], simulation_def[-1], len(experimental_def))
            sim_force_interp = np.interp(sim_def_interp, simulation_def, simulation_force)
        except Exception:
            sim_def_interp = simulation_def
            sim_force_interp = simulation_force

        # Indices for thickness-based splits (uses global thickness variable)
        try:
            uB_indx_sim = np.where(sim_def_interp >= thickness)[0][0]
            uB_indx_exp = np.where(experimental_def >= thickness)[0][0]
        except Exception:
            uB_indx_sim = 0
            uB_indx_exp = 0

        N = len(sim_force_interp)
        weights = np.linspace(5, 1, N)

        rmse_euc = euclidean_rmse(sim_force_interp, sim_def_interp, experimental_force, experimental_def)

        # Helper function to resample arrays and compute RMSE
        def match_length_and_rmse(sim_force, sim_def, exp_force, exp_def):
            min_len = min(len(sim_force), len(exp_force))
            if len(sim_force) > min_len:
                sim_def_resampled = np.linspace(sim_def[0], sim_def[-1], min_len)
                sim_force_resampled = np.interp(sim_def_resampled, sim_def, sim_force)
            else:
                sim_def_resampled = sim_def
                sim_force_resampled = sim_force

            if len(exp_force) > min_len:
                exp_def_resampled = np.linspace(exp_def[0], exp_def[-1], min_len)
                exp_force_resampled = np.interp(exp_def_resampled, exp_def, exp_force)
            else:
                exp_def_resampled = exp_def
                exp_force_resampled = exp_force

            dist_squared = (sim_force_resampled - exp_force_resampled) ** 2 + (sim_def_resampled - exp_def_resampled) ** 2
            return np.sqrt(np.mean(dist_squared))

        try:
            first_weighting_limit = np.where(experimental_def >= thickness / 5)[0][0]
        except Exception:
            first_weighting_limit = 0

        try:
            rmse_euc_elastic = match_length_and_rmse(
                sim_force_interp[:np.where(sim_def_interp >= thickness / 5)[0][0]],
                sim_def_interp[:np.where(sim_def_interp >= thickness / 5)[0][0]],
                experimental_force[:np.where(experimental_def >= thickness / 5)[0][0]],
                experimental_def[:np.where(experimental_def >= thickness / 5)[0][0]])
        except Exception:
            rmse_euc_elastic = rmse_euc

        try:
            rmse_euc_hardening = match_length_and_rmse(
                sim_force_interp[uB_indx_sim:], sim_def_interp[uB_indx_sim:],
                experimental_force[uB_indx_exp:], experimental_def[uB_indx_exp:])
        except Exception:
            rmse_euc_hardening = 0

        # weights depending on parameters selection
        if parameters == 'Elastic and Plastic':
            w_whole, w_elastic, w_hardening = 1, 1, 1
        elif parameters == 'Elastic':
            w_whole, w_elastic = 0, 2
            w_hardening = 0
        else:
            w_whole, w_elastic, w_hardening = 1, 1, 1

        try:
            rmse_euc_weighted = (w_whole * rmse_euc) + (w_elastic * rmse_euc_elastic) + (w_hardening * rmse_euc_hardening)
        except Exception:
            rmse_euc_weighted = rmse_euc

        rmse = rmse_euc_weighted
        nrmse = 100 * (rmse / (np.max(experimental_force) if np.max(experimental_force) != 0 else 1.0))
        # enforce parameter bounds penalties (strong)
        penalty = 0
        try:
            for key in scaled_params.keys():
                param = float(scaled_params[key])
                lower = float(lower_b[key]) if isinstance(lower_b, dict) and key in lower_b else -np.inf
                upper = float(upper_b[key]) if isinstance(upper_b, dict) and key in upper_b else np.inf
                if param < lower:
                    penalty += (lower - param) ** 200
                elif param > upper:
                    penalty += (param - upper) ** 200
        except Exception:
            pass

        total_cost = nrmse + penalty

        print('Parameters: ' + str(scaled_params))
        print('Penalty: ' + str(penalty))

        # call plotting helper if available
        try:
            plot_simulation_vs_experiment(sim_def_interp, sim_force_interp, experimental_def, experimental_force, first_weighting_limit, uB_indx_exp, total_cost, scaled_params, save_dir_path)
        except Exception:
            pass

        return total_cost

    except Exception as e:
        print("Error in objective_function:", str(e))
        with open("objective_function_error.log", "a") as f:
            f.write("=== ERROR in objective_function ===\n")
            traceback.print_exc(file=f)
            f.write("\n")
        return 100


# ===== Alternative Objective Function (SS) =====
def objective_function_ss(params, experimental_force, experimental_def, experimental_time, filter_data,
                       lower_bounds, upper_bounds, regularization_factor, working_dir,
                       save_dir_path, job_name, material_name):
    """
    Alternative objective function using simplified metric calculation.
    
    Differences from objective_function():
      - Simplified RMSE calculation without multiple weighting zones
      - Uses elastic and hardening region RMSEs separately
    
    Warning: May be redundant with objective_function(). Consider consolidating.
    Warning: Calls run_simulation() which has signature mismatch - needs fixing.
    """
    try:
        scaled_params = scale_parameters(params)
    except Exception:
        scaled_params = params

    # run FE
    try:
        simulation_force, simulation_def, simulation_time, simulation_PE, time_PE = run_simulation(
            working_dir, job_name, material_name, scaled_params)
    except Exception:
        traceback.print_exc()
        return 1e6

    # make arrays and compute interpolated fields
    try:
        sim_def_interp = np.linspace(simulation_def[0], simulation_def[-1], len(experimental_def))
        sim_force_interp = np.interp(sim_def_interp, simulation_def, simulation_force)
    except Exception:
        sim_def_interp = simulation_def
        sim_force_interp = simulation_force

    try:
        uB_indx_sim = np.where(sim_def_interp >= thickness)[0][0]
        uB_indx_exp = np.where(experimental_def >= thickness)[0][0]
    except Exception:
        uB_indx_sim = 0
        uB_indx_exp = 0

    # helper to match length and compute euclidean RMSE
    def match_length_and_rmse(sim_force, sim_def, exp_force, exp_def):
        min_len = min(len(sim_force), len(exp_force))
        if len(sim_force) > min_len:
            sim_def_resampled = np.linspace(sim_def[0], sim_def[-1], min_len)
            sim_force_resampled = np.interp(sim_def_resampled, sim_def, sim_force)
        else:
            sim_def_resampled = sim_def
            sim_force_resampled = sim_force

        if len(exp_force) > min_len:
            exp_def_resampled = np.linspace(exp_def[0], exp_def[-1], min_len)
            exp_force_resampled = np.interp(exp_def_resampled, exp_def, exp_force)
        else:
            exp_def_resampled = exp_def
            exp_force_resampled = exp_force

        return np.sqrt(np.mean((sim_force_resampled - exp_force_resampled) ** 2 + (sim_def_resampled - exp_def_resampled) ** 2))

    try:
        first_weighting_limit = np.where(experimental_def >= thickness / 5)[0][0]
    except Exception:
        first_weighting_limit = 0

    rmse_euc_elastic = match_length_and_rmse(
            sim_force_interp[:np.where(sim_def_interp >= thickness / 5)[0][0]] if np.any(sim_def_interp >= thickness / 5) else sim_force_interp,
            sim_def_interp[:np.where(sim_def_interp >= thickness / 5)[0][0]] if np.any(sim_def_interp >= thickness / 5) else sim_def_interp,
            experimental_force[:np.where(experimental_def >= thickness / 5)[0][0]] if np.any(experimental_def >= thickness / 5) else experimental_force,
            experimental_def[:np.where(experimental_def >= thickness / 5)[0][0]] if np.any(experimental_def >= thickness / 5) else experimental_def)

    rmse_euc_hardening = match_length_and_rmse(
            sim_force_interp[uB_indx_sim:], sim_def_interp[uB_indx_sim:], experimental_force[uB_indx_exp:], experimental_def[uB_indx_exp:])

    if parameters == 'Elastic and Plastic':
        w_whole, w_elastic, w_hardening = 1, 2, 2
    elif parameters == 'Elastic':
        w_whole, w_elastic = 0, 2
        w_hardening = 0
    else:
        w_whole, w_elastic, w_hardening = 1, 1, 1

    rmse = rmse_euc_elastic + rmse_euc_hardening
    nrmse = 100 * rmse / (np.max(np.sqrt(experimental_force ** 2 + experimental_def ** 2)) if np.max(np.sqrt(experimental_force ** 2 + experimental_def ** 2)) != 0 else 1.0)

    # penalty for bounds
    penalty = 0
    try:
        for key in scaled_params.keys():
            param = float(scaled_params[key])
            lower = float(lower_bounds[key]) if isinstance(lower_bounds, dict) and key in lower_bounds else -np.inf
            upper = float(upper_bounds[key]) if isinstance(upper_bounds, dict) and key in upper_bounds else np.inf
            if param < lower:
                penalty += ((lower - param) ** 2) * 100
            elif param > upper:
                penalty += ((param - upper) ** 2) * 100
    except Exception:
        pass

    total_cost = nrmse + penalty

    try:
        plot_simulation_vs_experiment(sim_def_interp, sim_force_interp, experimental_def, experimental_force, 0, uB_indx_exp, total_cost, scaled_params, save_dir_path)
    except Exception:
        pass

    return total_cost


# ===== General Utilities =====
# Note: delete_all_xyData() is available in FE_utility/selected_FE_utility_functions.py
# Import from there instead of duplicating:
#     from FE_utility.selected_FE_utility_functions import delete_all_xyData


# ===== Plotting Helpers =====
def plot_simulation_vs_experiment(simulation_def, simulation_force,
                                  experimental_def, experimental_force,
                                  exp_time_limi_indx, uB_indx_exp,
                                  difference, params, dir_path):
    """
    Plot simulation vs experiment comparison with cost annotation.
    
    Attempts to import external plotting function from basic_functions module.
    Falls back to simple built-in plot if import fails.
    """
    try:
        # Import external plotting function if available
        from basic_functions import plot_simulation_vs_experiment as external_plot
        return external_plot(simulation_def, simulation_force, experimental_def, experimental_force,
                             exp_time_limi_indx, uB_indx_exp, difference, params, dir_path)
    except Exception:
        # Use built-in plotting if external function unavailable
        try:
            params_unscaled = unscale_parameters(params) if isinstance(params, dict) else params
        except Exception:
            params_unscaled = params

        plt.figure(figsize=(8, 5))
        plt.plot(simulation_def, simulation_force, label='FE Simulation')
        plt.plot(experimental_def, experimental_force, label='Experiment', linestyle='--')
        try:
            if exp_time_limi_indx is not None and exp_time_limi_indx < len(experimental_def):
                plt.axvline(experimental_def[exp_time_limi_indx], color='r', linestyle='--', label='Exp limit')
            if uB_indx_exp is not None and uB_indx_exp < len(experimental_def):
                plt.axvline(experimental_def[uB_indx_exp], color='b', linestyle='--', label='Thickness')
        except Exception:
            pass
        plt.xlabel('Deflection [mm]')
        plt.ylabel('Force [N]')
        plt.title('Simulation vs Experiment (cost={:.2f})'.format(difference))
        plt.legend()
        os.makedirs(dir_path, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fname = os.path.join(dir_path, "Sim_Exp_{}.png".format(timestamp_str))
        plt.savefig(fname, dpi=300)
        plt.close()
        return fname


def plot_optimization_results(sigma_y0_values, E_values, R_inf_values, alpha_values, n_values, dir_path):
    """
    Plot parameter evolution over optimization iterations.
    
    Attempts to import external plotting function from basic_functions module.
    Falls back to simple built-in plot if import fails.
    """
    try:
        from basic_functions import plot_optimization_results as external_plot
        return external_plot(sigma_y0_values, E_values, R_inf_values, alpha_values, n_values, dir_path)
    except Exception:
        try:
            os.makedirs(dir_dir if 'dir_dir' in locals() else dir_path, exist_ok=True)
            iterations = range(max(1, len(sigma_y0_values)))
            plt.figure(figsize=(10, 6))
            if len(sigma_y0_values) > 0:
                plt.subplot(211)
                plt.plot(iterations[:len(sigma_y0_values)], sigma_y0_values, '-o')
                plt.title('Sigma_y0 over iterations')
            if len(E_values) > 0:
                plt.subplot(212)
                plt.plot(iterations[:len(E_values)], E_values, '-d')
                plt.title('E over iterations')
            timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            fname = os.path.join(dir_path, "Parameter_optimisation_{}.png".format(timestamp_str))
            plt.tight_layout()
            plt.savefig(fname, dpi=300)
            plt.close()
            return fname
        except Exception:
            traceback.print_exc()
            return None

# End of SPT_inverse_analysis.py
