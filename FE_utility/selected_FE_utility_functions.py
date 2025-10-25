"""
Abaqus FE Utility Functions for SPT Analysis

Utility functions for Abaqus finite element simulations of Small Punch Tests.
Includes helpers for model manipulation, job submission, results extraction,
and data processing.

IMPORTANT:
    - These functions assume execution inside an Abaqus Python environment
      (Abaqus/CAE or abaqus python) with access to `mdb`, `session`, `odbAccess`
    - Abaqus typically runs Python 2.7

Author: Saleem Lubbad
Original comments and annotations were cleaned by ChatGPT, under Oxford University license.
"""

# ===== General Utilities =====

def set_dir_and_working_path(dir_path):
    """Ensure dir exists and set it as working directory."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    os.chdir(dir_path)
    print('Working Directory set to:', dir_path)

def convert_txt_to_csv(directory):
    """
    Convert .txt files in `directory` to .csv files by copying content.
    Returns: list of created csv file paths.
    """
    created = []
    try:
        txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        for txt_file in txt_files:
            txt_path = os.path.join(directory, txt_file)
            csv_path = os.path.join(directory, txt_file.replace('.txt', '.csv'))
            with open(txt_path, 'r') as src, open(csv_path, 'w') as dst:
                dst.write(src.read())
            created.append(csv_path)
        return created
    except Exception as e:
        raise
# ===== Abaqus Session Helpers =====
def delete_all_xyData():
    """Delete all session.xyDataObjects (Abaqus session)."""
    if session is None:
        raise RuntimeError("Abaqus session not available.")
    for key in list(session.xyDataObjects.keys()):
        try:
            del session.xyDataObjects[key]
        except Exception:
            pass

def delete_all_jobs():
    """Delete all jobs from current mdb (Abaqus)."""
    if mdb is None:
        raise RuntimeError("Abaqus mdb not available.")
    for j in list(mdb.jobs.keys()):
        try:
            del mdb.jobs[j]
        except Exception:
            pass

def delete_all_materials(model_name):
    """Delete all materials in given model."""
    if mdb is None:
        raise RuntimeError("Abaqus mdb not available.")
    model = mdb.models[model_name]
    for mat in list(model.materials.keys()):
        try:
            del model.materials[mat]
        except Exception:
            pass

def close_all_odb_and_remove_locks_and_kill_jobs():
    """
    Close all open ODBs, terminate running jobs, and remove .lck files
    in current directory. Useful for cleanup after failed batch runs.
    """
    if session is None or mdb is None:
        raise RuntimeError("Abaqus environment not available.")
    # Close ODBs
    for odb_name in list(session.odbs.keys()):
        try:
            session.odbs[odb_name].close()
            print("Closed ODB:", odb_name)
        except Exception:
            pass

    # Try to terminate running jobs
    for job_name in list(mdb.jobs.keys()):
        try:
            mdb.jobs[job_name].terminate()
            print("Terminated job:", job_name)
        except Exception:
            pass

    # Remove .lck files in cwd
    for fname in os.listdir(os.getcwd()):
        if fname.endswith('.lck'):
            try:
                os.remove(fname)
                print("Removed lock:", fname)
            except Exception:
                pass


# ===== Model Enquiries =====
def get_analysis_info(model_name, part_name):
    """
    Return a small summary string and numeric mesh info for the named
    model & part (requires Abaqus objects to exist).
    """
    if mdb is None:
        raise RuntimeError("Abaqus mdb not available.")
    model = mdb.models[model_name]
    part = model.parts[part_name]

    # Extract element type from mesh
    try:
        mesh_element_type = part.elements[0].type
    except Exception:
        mesh_element_type = "Unknown"

    # Extract representative edge sizes from specific edge masks
    def _edge_size_from_mask(mask):
        try:
            edges = part.edges.getSequenceFromMask(mask=(mask,), )
            if len(edges) > 0:
                e0 = edges[0]
                # Calculate element size from edge length and node count
                nodes_count = len(e0.getNodes())
                size = e0.getSize() / max(1, nodes_count - 1)
                return float(np.round(size, 6))
        except Exception:
            pass
        return None

    inner_mesh_size = _edge_size_from_mask('[#4302 ]')
    outer_mesh_size = _edge_size_from_mask('[#8848 ]')
    depth_mesh_size = _edge_size_from_mask('[#10000 ]')

    # Extract time increment from explicit step if available
    dt = None
    try:
        step = model.steps.values()[0]
        try:
            dt = step.massScaling[0].dt
        except Exception:
            dt = None
    except Exception:
        dt = None

    text_str = (
        "Mesh Element Type: {}\n"
        "Inner mesh size = {}\n"
        "Outer mesh size = {}\n"
        "Depth mesh size = {}\n"
    ).format(mesh_element_type, inner_mesh_size, outer_mesh_size, depth_mesh_size)

    return text_str, mesh_element_type, depth_mesh_size, inner_mesh_size, outer_mesh_size, dt

# ===== Material Creation Helpers =====
def Create_material_VL(model_name, material_name, Sy0, E, v, Rinf, n, alpha, Rho):
    """
    Create a material with a Voce-like hardening table (elastic + plastic).
    E is expected in GPa and will be scaled to MPa for Abaqus.
    """
    if mdb is None:
        raise RuntimeError("Abaqus mdb not available.")
    model = mdb.models[model_name]

    if material_name not in model.materials.keys():
        model.Material(name=material_name)

    epsilon_pl = np.linspace(0, 0.5, 100)
    sigma = Sy0 + Rinf * (alpha * epsilon_pl + 1 - np.exp(-n * epsilon_pl))
    stress_strain_table = np.column_stack((sigma, epsilon_pl)).tolist()

    model.materials[material_name].Plastic(table=stress_strain_table)
    model.materials[material_name].Density(table=((Rho,),))
    model.materials[material_name].Elastic(table=((E * 1000.0, v),))

    print("Material created/updated:", material_name)
    return stress_strain_table

def Create_material_JC(model_name, material_name, Sy0, E, v, B, n, Rho, m, Tmelt, Tr):
    """
    Create a Johnson-Cook material entry using Abaqus JC parameter format.
    Note: Parameter ordering depends on Abaqus version.
    """
    if mdb is None:
        raise RuntimeError("Abaqus mdb not available.")
    model = mdb.models[model_name]

    if material_name not in model.materials.keys():
        model.Material(name=material_name)

    # Define Johnson-Cook plasticity with material parameters
    model.materials[material_name].Plastic(
        hardening=JOHNSON_COOK, scaleStress=None,
        table=((Sy0, B, n, m, Tmelt, Tr), )
    )
    model.materials[material_name].Density(table=((Rho,),))
    model.materials[material_name].Elastic(table=((E * 1000.0, v),))

    print("JC material created/updated:", material_name)

def Create_material_Combined_JC_VL(model_name, material_name, Sy0, E, v, Rinf, n, alpha, Rho, m, T, Tr, Tmelt):
    """
    Create material with combined Voce hardening and temperature-dependent softening.
    Produces a table of (stress, plastic_strain) and assigns to material.
    """
    if mdb is None:
        raise RuntimeError("Abaqus mdb not available.")
    model = mdb.models[model_name]

    if material_name not in model.materials.keys():
        model.Material(name=material_name)

    epsilon_pl = np.linspace(0, 1.0, 300)
    sigma_plastic = Sy0 + Rinf * (alpha * epsilon_pl + 1 - np.exp(-n * epsilon_pl))
    T_ref = (T - Tr) / float(Tmelt - Tr) if (Tmelt - Tr) != 0 else 0.0
    sigma = sigma_plastic * (1.0 - (T_ref ** m))
    stress_strain_table = np.column_stack((sigma, epsilon_pl)).tolist()

    model.materials[material_name].Plastic(table=stress_strain_table)
    model.materials[material_name].Density(table=((Rho,),))
    model.materials[material_name].Elastic(table=((E * 1000.0, v),))

    print("Combined material created/updated:", material_name)
    return stress_strain_table


# ===== Job Submission & Run Helpers =====
def Submit_Job(job_name, model_name):
    """
    Create and submit a job if it doesn't exist; wait for completion.
    Job definition parameters (cores, memory) should be configured before calling.
    """
    if mdb is None:
        raise RuntimeError("Abaqus mdb not available.")
    if job_name in mdb.jobs.keys():
        print("Job '{}' already exists".format(job_name))
    else:
        print("Creating new job '{}'".format(job_name))
        mdb.Job(name=job_name, model=model_name, description=job_name)

    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()
    print(job_name, "Completed")


def run_simulation(job_name, model_name):
    """
    Submit job and return the main result vectors via get_simulation_results.
    This function is intentionally small: material updates & model configuration
    should be done before calling it.
    """
    Submit_Job(job_name, model_name)
    # Brief pause to ensure output files are written
    time.sleep(1)
    return get_simulation_results(job_name)


# ===== ODB Extraction & Plotting =====
def get_simulation_results(job_name):
    """
    Open the ODB corresponding to job_name and extract:
      - punch force (nodal reaction)
      - sample deflection (nodal displacement)
      - PEEQ (integration point average)
    Returns: simulation_force, simulation_def, time_array, simulation_PE, time_PE
    If any extraction fails, returns empty arrays and prints a warning.
    """
    if session is None:
        raise RuntimeError("Abaqus session not available.")
    odb_path = job_name + ".odb"
    if not os.path.exists(odb_path):
        # Locate ODB in session if not found on disk
        print("ODB not found on disk:", odb_path)
        # Use currently opened ODB from session
        if session.odbs:
            odb = list(session.odbs.values())[0]
        else:
            raise RuntimeError("No open ODB found and {} missing.".format(odb_path))
    else:
        odb = openOdb(path=odb_path)

    try:
        delete_all_xyData()

        session.viewports['Viewport: 1'].setValues(displayedObject=odb)

        # Punch force (nodal RF component RF2)
        session.xyDataListFromField(odb=odb, outputPosition=NODAL,
                                    variable=(('RF', NODAL, ((COMPONENT, 'RF2'),)),),
                                    nodeLabels=(('PUNCH-1', ('1',)),))
        # Retrieve most recently created XYData object
        force_key = list(session.xyDataObjects.keys())[-1]
        xy_force = session.xyDataObjects[force_key]
        time_force, simulation_force = zip(*xy_force)

        # Sample displacement (U2) from a named node set
        session.xyDataListFromField(odb=odb, outputPosition=NODAL,
                                    variable=(('U', NODAL, ((COMPONENT, 'U2'),)),),
                                    nodeSets=('SAMPLE-1.SAMPLE_CNTRBTM',))
        def_key = list(session.xyDataObjects.keys())[-1]
        xy_def = session.xyDataObjects[def_key]
        time_def, simulation_def = zip(*xy_def)

        # PEEQ average (integr. point)
        session.xyDataListFromField(odb=odb, outputPosition=INTEGRATION_POINT,
                                    variable=(('PEEQ', INTEGRATION_POINT),),
                                    nodeSets=("SAMPLE-1.SAMPLE_CNTRBTM",))
        pe_key = list(session.xyDataObjects.keys())[-1]
        xy_pe = session.xyDataObjects[pe_key]
        time_pe, simulation_pe = zip(*xy_pe)

    except Exception as e:
        print("Error extracting results:", e)
        try:
            odb.close()
        except Exception:
            pass
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    try:
        odb.close()
    except Exception:
        pass

    simulation_force = np.abs(np.array(simulation_force))
    simulation_def = np.abs(np.array(simulation_def))
    simulation_pe = np.abs(np.array(simulation_pe))
    time_arr = np.array(time_force)

    return simulation_force, simulation_def, time_arr, simulation_pe, np.array(time_pe)


def export_fvData_btm(odb_path_or_job, rename_prefix=None):
    """
    Lightweight wrapper to extract force/deflection/PEEQ and return the XYData objects
    (or raise if ODB not available). Expects the job's ODB file name or an open ODB path.
    Returns (xy_def, xy_force, combined_xy, pe_xy) as session xyDataObjects.
    """
    if session is None:
        raise RuntimeError("Abaqus session not available.")

    # open odb if path given
    if isinstance(odb_path_or_job, str) and odb_path_or_job.endswith('.odb') and os.path.exists(odb_path_or_job):
        odb = openOdb(path=odb_path_or_job)
        session.viewports['Viewport: 1'].setValues(displayedObject=odb)
    else:
        # Use currently displayed/opened ODB
        if session.odbs:
            odb = list(session.odbs.values())[0]
            session.viewports['Viewport: 1'].setValues(displayedObject=odb)
        else:
            raise RuntimeError("No ODB available")

    # Extract force, displacement, and PEEQ as XYData objects
    session.xyDataListFromField(odb=odb, outputPosition=NODAL,
                                variable=(('RF', NODAL, ((COMPONENT, 'RF2'),)),),
                                nodeLabels=(('PUNCH-1', ('1',)),))
    session.xyDataListFromField(odb=odb, outputPosition=NODAL,
                                variable=(('U', NODAL, ((COMPONENT, 'U2'),)),),
                                nodeSets=('SAMPLE-1.SAMPLE_CNTRBTM',))
    session.xyDataListFromField(odb=odb, outputPosition=INTEGRATION_POINT,
                                variable=(('PEEQ', INTEGRATION_POINT),),
                                nodeSets=("SAMPLE-1.SAMPLE_CNTRBTM",))

    # Retrieve the three most recently created XYData objects
    keys = list(session.xyDataObjects.keys())
    xy_force = session.xyDataObjects[keys[-3]]
    xy_def = session.xyDataObjects[keys[-2]]
    xy_pe = session.xyDataObjects[keys[-1]]

    # Combine absolute deflection and force into single XYData
    combined = combine(abs(xy_def), abs(xy_force))
    combined.setValues(sourceDescription='combined(def,force)')

    if rename_prefix:
        try:
            session.xyDataObjects.changeKey(fromName=xy_force.name, toName=rename_prefix + '_Punch_force')
        except Exception:
            pass

    return xy_def, xy_force, combined, xy_pe


def save_results_to_csv(job_name, out_dir, result_label='simulation'):
    """
    Save simulation vectors (force, deflection, PEEQ) to CSV and write a simple info.txt.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sim_force, sim_def, time_arr, sim_pe, time_pe = get_simulation_results(job_name)
    csv_path = os.path.join(out_dir, result_label + '_simulation_data.csv')
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Deflection', 'Force', 'PEEQ'])
        for t, d, fr, p in zip(time_arr, sim_def, sim_force, sim_pe):
            writer.writerow([t, d, fr, p])

    # Save analysis metadata
    info_path = os.path.join(out_dir, result_label + '_info.txt')
    try:
        txt, *_ = get_analysis_info(mdb.models.keys()[0], mdb.models[mdb.models.keys()[0]].parts.keys()[0])
    except Exception:
        txt = "Analysis info not available."
    with open(info_path, 'w') as f:
        f.write(txt)
    print("Saved:", csv_path, info_path)


def plot_job_results(job_name, out_dir, result_label, deflection_exp=None, force_exp=None):
    """
    Plot simulation vs optional experimental curves; expects CSV written by save_results_to_csv.
    """
    csv_path = os.path.join(out_dir, result_label + '_simulation_data.csv')
    if not os.path.exists(csv_path):
        raise RuntimeError("CSV not found: " + csv_path)

    sim_def, sim_force = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sim_def.append(float(row['Deflection']))
            sim_force.append(float(row['Force']))

    plt.figure(figsize=(8, 6))
    plt.plot(sim_def, sim_force, label='FE Simulation')
    if deflection_exp is not None and force_exp is not None:
        plt.plot(deflection_exp, force_exp, '--', label='Experiment')

    info_txt = ""
    info_path = os.path.join(out_dir, result_label + '_info.txt')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info_txt = f.read()

    plt.text(0.5, 0.35, info_txt, transform=plt.gca().transAxes, fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    plt.xlabel('Deflection [mm]')
    plt.ylabel('Force [N]')
    plt.title('SPT: Simulation vs Experiment')
    plt.grid(True)
    plt.legend()
    png_path = os.path.join(out_dir, result_label + '.png')
    plt.savefig(png_path)
    plt.close()
    print("Saved plot:", png_path)


# ===== Experimental Data Reader =====
def read_exp_fv_data(file_path, threshold=0.0):
    """
    Read experimental force-deflection CSV assumed format:
    Time, Displacement, Deflection, Force, ...
    Expected format: columns 0-3 contain time, displacement, deflection, force.
    Returns: force, deflection, time, displacement  (numpy arrays)
    """
    force = []
    deflection = []
    displacement = []
    time_arr = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            # Extract data from first 4 columns: time, displacement, deflection, force
            try:
                t = float(row[0].strip())
                disp = float(row[1].strip())
                d = float(row[2].strip())
                fval = float(row[3].strip())
            except Exception:
                # Skip row if data cannot be parsed
                continue
            time_arr.append(t)
            displacement.append(disp)
            deflection.append(d)
            force.append(abs(fval))

    force = np.array(force)
    deflection = np.array(deflection)
    displacement = np.array(displacement)
    time_arr = np.array(time_arr)

    if force.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if force.max() < 50:
        # Force values below 50 assumed to be in kN; convert to N
        force *= 1000.0

    mask = force > threshold
    force = force[mask]
    deflection = deflection[mask]
    displacement = displacement[mask]
    time_arr = time_arr[mask]
    # Normalize all measurements to start at zero
    deflection = deflection - deflection[0]
    displacement = displacement - displacement[0]
    time_arr = time_arr - time_arr[0]
    force = force - force[0]

    return force, deflection, time_arr, displacement
