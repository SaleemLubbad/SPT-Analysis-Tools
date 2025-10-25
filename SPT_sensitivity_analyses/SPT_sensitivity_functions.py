
"""
SPT Sensitivity Analysis Functions

Performs friction and mesh sensitivity analyses for the Small Punch Test (SPT)
using Abaqus finite-element simulations.

IMPORTANT:
    - These functions assume execution inside an Abaqus Python environment
      (Abaqus/CAE or abaqus python) with access to `mdb`, `session`, `odbAccess`
    - Abaqus typically runs Python 2.7
    - Some functions depend on FE_utility/selected_FE_utility_functions.py

Core routines include:
    - Automated updating of contact friction properties
    - Abaqus job creation and submission
    - Data export and interpolation

Author: Saleem Lubbad
Original comments and annotations were cleaned by ChatGPT, under Oxford University license.
"""
# ===== Friction Analysis Functions =====

def update_friction(friction_coef):
    """
    Update friction coefficients for all contact interactions in the SPT model.

    Parameters
    ----------
    friction_coef : float
        Friction coefficient to apply to all contact interactions.
    """
    mdb.models['Model-1'].interactionProperties['LowerDie-spec'].tangentialBehavior.setValues(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0,
        table=((friction_coef, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION,
        fraction=0.005, elasticSlipStiffness=None)
    mdb.models['Model-1'].interactionProperties['Punch-Spec'].tangentialBehavior.setValues(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0,
        table=((friction_coef, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION,
        fraction=0.005, elasticSlipStiffness=None)
    mdb.models['Model-1'].interactionProperties['UpperDie-Spec'].tangentialBehavior.setValues(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0,
        table=((friction_coef, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION,
        fraction=0.005, elasticSlipStiffness=None)

def friction_sensitivity_analysis(friction_coef_range, thickness, end_def, dir_path, Run_Job):
    """
    Run sensitivity study for friction coefficient. Saves figures and CSVs to dir_path.

    Parameters
    ----------
    friction_coef_range : iterable
        List/array of friction coefficient values to test (e.g. [0.05, 0.1, 0.15]).
    thickness : float
        Sample thickness used for region-of-interest selection and plotting.
    end_def : float
        End deflection for visual region-of-interest shading (unused internally here
        but kept for API compatibility).
    dir_path : str
        Directory to store output files and plots.
    Run_Job : bool
        If True, create and submit Abaqus jobs; otherwise expects results files already present.
    """
    set_dir_and_working_path(dir_path)

    force_exp, def_exp, time_exp, dis_exp = read_exp_fv_data(file_path, threshold)
    plt.figure(figsize=(10, 6))
    plt.plot(def_exp, force_exp, label='Experiment - Eurofer 97')

    Err = []

    for i, friction_coef in enumerate(friction_coef_range):

        update_friction(friction_coef)
        Job_name = 'Friction_sen_' + str(int(friction_coef * 100))
        if Run_Job:
            Create_job(Job_name, Model_name)
            Submit_Job(Job_name)
            mdb.jobs[Job_name].waitForCompletion()
            print(Job_name + ' completed')
        delete_all_xyData()
        xy1, xy2, xy3, PE = export_fvData_btm(dir_path+'/', Job_name, Job_name)
        simulation_time, force_sim = zip(*xy2)
        simulation_time, def_sim = zip(*xy1)

        force_sim = np.abs(force_sim)
        def_sim = np.abs(def_sim)

        np.savetxt(str(Job_name) + '_def.csv', xy1, delimiter=',')
        np.savetxt(str(Job_name) + '_force.csv', xy2, delimiter=',')

        label = "$\\mu$= {}".format(friction_coef)
        plt.plot(def_sim, force_sim, label=label)
        plt.savefig(os.path.join(dir_path, "Progress" + str(int(friction_coef * 100)) + ".png"), dpi=400)

        # compute indices for slices
        end_indx = int(np.where(force_sim == max(force_sim))[0][0])
        start_indx = 0
        global def_max
        def_max = end_indx

        end_indx_exp = int(np.where(force_exp == max(force_exp))[0][0])

        # experimental slice
        y2 = force_exp[:end_indx_exp]
        x2 = def_exp[:end_indx_exp]

        # simulated slice
        y1 = force_sim[start_indx:end_indx]
        x1 = def_sim[start_indx:end_indx]

        # interpolate simulation onto experimental grid
        f1 = interp1d(x1, y1, kind='linear', fill_value="extrapolate")
        y1_int = f1(x2)

        v_true = x2
        F_true = y2

        v_pred = np.linspace(min(x1), max(x1), len(x2))
        F_pred = y1_int

        # scale to kN for error calculation (optional)
        F_true_scaled = F_true / 1000.0
        F_pred_scaled = F_pred / 1000.0

        MSE, RMSE, NRMSE = Error(v_true, F_true_scaled, v_pred, F_pred_scaled)
        Err.append(NRMSE * 100)

        plt.xlabel("Deflection [mm]")
        plt.ylabel("Force [N]")
        plt.title("Friction Coefficient Sensitivity Study")
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(dir_path, "Friction Coefficient Sensitivity Analysis.png"), dpi=1200)
        print("plot save :: " + os.path.join(dir_path, "Friction Coefficient Sensitivity Analysis.png"))

    # plot summary NRMSE vs friction
    plt.figure(figsize=(10, 6))
    plt.plot(friction_coef_range, Err, 'r', linestyle='--', marker='*', markerfacecolor='blue')
    plt.xlabel("Friction Coefficient mu")
    plt.ylabel("Normalised Root Mean Squared Error NRMSE [%]")
    plt.title("Friction Sensitivity Study")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'Friction sensitivity study NRMSE.png'), dpi=1200)
    print("plot save :: " + os.path.join(dir_path, 'Friction sensitivity study NRMSE.png'))


# ===== Elastic Modulus Sensitivity Analysis =====

def Sensitivity_Analysis_E(dir_path, Model_name, job_name, Material_name, Sy0, v, E, Rinf, alpha, n, Rho):
    """
    Perform elastic modulus sensitivity study.

    Parameters
    ----------
    dir_path : str
        Directory path for saving results.
    Model_name : str
        Name of the Abaqus model.
    job_name : str
        Name for the Abaqus job.
    Material_name : str
        Name of the material.
    Sy0 : float
        Initial yield stress (MPa).
    v : float
        Poisson's ratio.
    E : array-like
        List of elastic moduli values to test (GPa).
    Rinf : float
        Saturation stress (MPa).
    alpha : float
        Linear hardening coefficient.
    n : float
        Exponential hardening coefficient.
    Rho : float
        Material density (kg/m³).

    Notes
    -----
    Expects external helper functions: read_exp_fv_data, update_section_namterial,
    Submit_Job, get_simulation_results, get_analysis_info.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('New path created: ' + dir_path)

    os.chdir(dir_path)
    print('Working Directory set to: ' + dir_path)

    update_section_namterial(Model_name, section_name, Material_name)

    experimental_force, experimental_def, experimental_time, experimental_displacement = read_exp_fv_data(file_path, threshold)

    if Material_name in mdb.models[Model_name].materials.keys():
        print("Material '{}' already exists in model '{}'".format(Material_name, Model_name))
    else:
        print("Creating new material '{}'".format(Material_name))
        mdb.models[Model_name].Material(name=Material_name)

    if job_name in mdb.jobs.keys():
        print("Job '{}' already exists".format(job_name))
    else:
        print("Creating new job '{}'".format(job_name))
        mdb.Job(name=job_name, model=Model_name, description=job_name)

    epsilon_pl = np.linspace(0, 0.5, 100)
    sigma = Sy0 + Rinf * (alpha * epsilon_pl + 1 - np.exp(-n * epsilon_pl))
    stress_strain_table = np.column_stack((sigma, epsilon_pl))
    mdb.models[Model_name].materials[Material_name].Plastic(table=stress_strain_table)
    mdb.models[Model_name].materials[Material_name].Density(table=((Rho, ), ))
    fig, axs = plt.subplots(figsize=(12, 6))
    for Ei in E:
        print(Ei)
        mdb.models[Model_name].materials[Material_name].Elastic(table=((Ei*1000, v), ))
        Submit_Job(job_name)
        mdb.jobs[job_name].waitForCompletion()
        simulation_force, simulation_def, simulation_time, simulation_PE, time_PE = get_simulation_results(job_name)
        axs.plot(simulation_def[:len(simulation_def)], simulation_force[:len(simulation_def)], label='E = {} GPa'.format(Ei))

    exp_limit = np.where(experimental_def >= 0.1)[0][0]
    axs.plot(experimental_def[:exp_limit], experimental_force[:exp_limit], label='Experimental: E = {} GPa'.format(210))
    axs.set_title('Elastic Modulus Sensitivity Analysis - Load-Deflection Curve')
    axs.set_xlabel('Deflection [mm]')
    axs.set_ylabel('Force [N]')
    axs.legend(loc='lower right')

    mesh_element_type, depth_mesh_size, inner_mesh_size, outer_mesh_size, second_order, integration, Punch_dis_rate, dt = get_analysis_info(Model_name, part_name)
    text_str = (
        'Mesh Element Type : {}\n'
        'Inner Mesh size = {:.2f} μm\n'
        'Outer Mesh size = {:.2f} μm\n'
        'Depth Mesh size = {:.2f} μm\n'
        'Second order accuracy : {}\n'
        '{}\n'
        'Punch displacement rate = {:.4f} mm/s\n'
        'Target Time Increment = {:.6f} s'
    ).format(mesh_element_type, inner_mesh_size, outer_mesh_size, depth_mesh_size, second_order, integration, Punch_dis_rate, dt)
    axs.text(0.05, 0.95, text_str, horizontalalignment='left', verticalalignment='top', transform=axs.transAxes,
             bbox={'facecolor': 'white', 'alpha': 0.25, 'pad': 10, 'edgecolor': 'none'})

    axs.grid(True)
    plt.tight_layout()

    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = os.path.join(dir_path, "Elastic Modulus Sensititivy Analysis_{}.png".format(timestamp_str))
    plt.savefig(filename, dpi=600)
    plt.close()

