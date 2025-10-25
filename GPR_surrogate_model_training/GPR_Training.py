"""
Gaussian Process Regression (GPR) Surrogate Model Training

Train GPR models with multiple kernels and generate learning curves showing
prediction error and confidence evolution as training data size increases.

Key features:
    - Trains GPRs incrementally using subsets of data to form learning curves
    - Evaluates performance across multiple kernels using RMSE and confidence width
    - Optionally validates against experimental inputs
    - Saves plots for error, confidence, and loss evolution
    - Supports both single-task and multi-task GP regression

Returns
-------
model : GPyTorch model
    Final trained model (last kernel loop).
least_MSE : float
    Best (lowest) test RMSE across kernels/increments.
least_MSE_kernel : kernel object
    Kernel achieving least_MSE.
least_MSE_training : float
    Lowest train RMSE across kernels/increments.
exp_y_o, lower_exp_o, upper_exp_o : array-like
    Denormalized experimental predictions + bounds (if any).
exp_y, lower_exp, upper_exp : array-like
    Normalized experimental predictions + bounds (if any).
exp_mse_percentage : float
    RMSE on experimental truth (normalized), if provided.
least_test_confidence : float
    Lowest average (upper-lower) confidence.
least_confidence_kernel : kernel object
    Kernel achieving lowest confidence.

Example
-------
    kernels = [
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)),
    ]

    model, least_MSE, least_MSE_kernel, *_ = train_GPR_model_with_learning_curves(
        folder_path="work/",
        results_path="work/results",
        kernels=kernels,
        X=X_norm, y=y_norm, y0=y_orig,
        test_data_portion=0.2,
        Noise_level=1e-6,
        exp_file_path="experiments/",
        exp_prediction_path=True,
        exp_x=exp_inputs,
        exp_y_truth=exp_truth_dict,
        exp_y_label=exp_labels,
        denormalize=inv_scale,
        scaler=per_output_scalers,
        norm_method_name="Min-max",
        input_features=["x1"] if X_norm.shape[1]==1 else ["x1","x2"],
        output_features=["Y"] if y_norm.shape[1]==1 else ["Y1","Y2"],
        no_training_increments=max(1, X_norm.shape[0]//10),
        training_iter=500,
        lr=0.05,
        early_stopping_tolerance=1e-5,
        Progress_Plot=True
    )
    print("Best kernel:", least_MSE_kernel)
    print("Best test RMSE:", least_MSE)

Author: Saleem Lubbad
Original comments and annotations were cleaned by ChatGPT, under Oxford University license.
"""

# ===== Main Training Function =====

def train_GPR_model_with_learning_curves(
        folder_path, results_path,
        kernels, 
        X, y, y0, test_data_portion, Noise_level,
        exp_file_path,exp_prediction_path,
        exp_x, exp_y_truth, exp_y_label,
        denormalize, scaler, norm_method_name,
        input_features, output_features,
        no_training_increments, training_iter,lr,early_stopping_tolerance,
        Progress_Plot):


    exp_y_truth_all = exp_y_truth
    
    global output_unit
    check_create_dir(folder_path)
    
    print(f'Progress Plot: {Progress_Plot}')
    print('----- PREPARE TEST AND TRAIN DATA -----')
    train_x, train_y, test_x, test_y = prepare_training_data(X, y,test_data_portion)

    print('\n---- PLOT RAW TRAINING DATA ----\n')
    
    exp_x = torch.tensor(exp_x, dtype=torch.float32)
    MSE_all = []
    MSE_all_training = []
    test_confidence_all = []
    exp_mse_percentage = []  # default in case no truth provided

    output_features_dim = train_y.shape[1]  
    print(f'output_features_dim: {output_features_dim}')
    input_features_dim = train_x.shape[1]  

    if Progress_Plot == True:
        print(f'Progress Plot: {Progress_Plot}')
        print(f'Number of Tasks = {output_features_dim}')

    for kernel in kernels:
        train_sizes = []
        train_mse = []
        test_mse = []
        test_confidence = []
        Loss = []
        
        for i in range(int(len(train_x)/no_training_increments), len(train_x) + 1, no_training_increments): 
            X_train_subset = train_x[:i]
            y_train_subset = train_y[:i]

            if Progress_Plot == True:
                print(f'Train x size - {i}: {train_x.shape}')
                print(f'Train y size - {i}: {train_y.shape}')
                print(f'Train x subset size - {i}: {X_train_subset.shape}')
                print(f'Train y subset size - {i}: {y_train_subset.shape}')

            textstr = (f'Input features: {", ".join(input_features)}\n'
                    f'Output features: {", ".join(output_features)}\n'
                    f'Kernel: {kernel.__class__.__name__}\n'
                    f'Training iterations= {training_iter} | $L_r$= {lr}\n'
                    f'Normalisation Method: {norm_method_name}')
            likelihood, model, final_loss , optimisation_results = train_GPR_model(
                X_train_subset,y_train_subset, 
                lr, training_iter, 
                kernel,
                Noise_level,early_stopping_tolerance,textstr,results_path)
            
            train_preds, lower, upper = make_GPR_predictions(model, X_train_subset)
            test_preds, lower, upper = make_GPR_predictions(model,test_x)
            
            test_confidence_i = np.average(upper - lower)

            y_range = train_y.max(0)[0] - train_y.min(0)[0] if output_features_dim > 1 else train_y.max() - train_y.min()

            if output_features_dim > 1:
                train_mse_percentage = []
                test_mse_percentage = []
                test_confidence = []

                for j in range(output_features_dim):
                    train_pred_j = train_preds[:, j] if train_preds.ndim > 1 else train_preds
                    test_confidence_j = test_confidence_i[:,j] if isinstance(test_confidence_i, np.ndarray) and test_confidence_i.ndim > 1 else test_confidence_i
                    y_train_subset_j = y_train_subset[:, j] if y_train_subset.ndim > 1 else y_train_subset

                    train_mse_percentage.append(Error(y_train_subset_j.numpy(), train_pred_j))
                    test_mse_percentage.append(Error(test_y[:, j].numpy(), test_preds[:, j]))
                    if isinstance(test_confidence_j, np.ndarray):
                        test_confidence.append(np.average(test_confidence_j))
                    else:
                        test_confidence.append(test_confidence_j)
                    Loss.append(final_loss)

            else:
                train_mse_percentage = Error(y_train_subset.numpy(), train_preds)
                test_mse_percentage = Error(test_y.numpy(), test_preds)
                test_confidence.append(np.average(test_confidence_i))
                Loss.append(final_loss)
                
            train_mse.append(train_mse_percentage)
            test_mse.append(test_mse_percentage)
            train_sizes.append(i)
            
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()

        if output_features_dim == 1:
            test_final_error_denormalised = denormalize(np.array(test_mse[-1]).reshape(-1, 1), scaler[0])[0][0]
            try:
                output_unit = (
                    'MPa' if any(x in output_features[0] for x in ['R', '\sigma', 'Sy'])
                    else 'GPa' if 'E' in output_features[0]
                    else ''
                )
            except IndexError:
                output_unit = ''
            
            ax1.plot(train_sizes, train_mse, label='Training Error', color='blue', marker='o')
            ax1.plot(train_sizes, test_mse, label=f'Test Error\nFinal Error = {test_mse[-1]:.2f} [{Error_unit}]',
                    color='red', marker='o')
            
            ax2 = ax1.twinx()
            ax2.plot(train_sizes, test_confidence, label=f'Test Average Confidence Margin\nFinal Average Confidence Margin = {test_confidence[-1]:.2f}[Normalised]', 
                    color='green', marker='o', linestyle='--')
            ax2.set_ylabel('Confidence Margin [Normalised]')
        else:
            for task in range(output_features_dim):
                try:
                    output_unit = (
                        'MPa' if any(x in output_features[task] for x in ['R', '\sigma', 'Sy'])
                        else 'GPa' if 'E' in output_features[task]
                        else ''
                    )
                except IndexError:
                    output_unit = ''
                
                ax1.plot(train_sizes, [train_mse[i][task] for i in range(len(train_mse))], 
                        label=f'Training Error | Task: {output_features[task]}', marker='o')
                ax1.plot(train_sizes, [test_mse[i][task] for i in range(len(test_mse))], 
                        label=f'Test Error | Task: {output_features[task]}\nFinal Error = {test_mse[-1][task]:.2f} [{Error_unit}] ', 
                        marker='*')
                
                ax2 = ax1.twinx()
                ax2.plot(train_sizes, [test_confidence[i][task] for i in range(len(test_confidence))], 
                        label=f'Test Average Confidence Margin | Task: {output_features[task]}\nFinal Average Confidence Margin = {test_confidence[-1][task]:.2f}', 
                        color='green', marker='*', linestyle='--')

        ax1.set_xlabel('Training Set Size')
        ax1.set_ylabel(f'{Error_Caption} [{Error_unit}]')
        ax1.set_title('Learning Curve: Prediction Error and Confidence Margin vs Training Set Size')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        annotation_y = 0.7 if output_features_dim == 1 else 0.5
        textstr = (f'Input features: {", ".join(input_features)}\n'
                f'Output features: {", ".join(output_features)}\n'
                f'Kernel: {kernel.__class__.__name__}\n'
                f'Training iterations= {training_iter} | $L_r$= {lr}\n'
                f'Normalisation Method: {norm_method_name}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
        ax1.text(1, annotation_y, textstr, fontsize=10, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        timestamp = now.strftime("%d-%m-%y-%H-%M-%S")
        save_dir_temp = f'{results_path}/{clean_textstr(kernel.__class__.__name__)}/'
        check_create_dir(save_dir_temp)
        filename = save_dir_temp + f'Learning Curve with Confidence {clean_textstr(textstr)}.png'
        plt.savefig(filename)
        plt.grid(True)
        plt.show()

        #### Error-only learning curve
        plt.figure(figsize=(10, 6))
        if output_features_dim == 1:
            test_final_error_denormalised = denormalize(np.array(test_mse[-1]).reshape(-1,1),scaler[0])[0][0]
            try:
                output_unit = (
                    'MPa' if any(x in output_features[0] for x in ['R', '\sigma', 'Sy'])
                    else 'GPa' if 'E' in output_features[0]
                    else ''
                )
            except IndexError: 
                output_unit = ''

            plt.plot(train_sizes, train_mse, label=f'Training Error', color='blue', marker='o')
            plt.plot(train_sizes, test_mse, label=f'Test Error \nFinal Error = {test_mse[-1]:.2f} [{Error_unit}]', color='red', marker='o')
            annotation_y = 0.85
        else:
            try:
                output_unit = (
                    'MPa' if any(x in output_features[i] for x in ['R', '\sigma', 'Sy'])
                    else 'GPa' if 'E' in output_features[i]
                    else ''
                )
            except IndexError: 
                output_unit = ''

            for task in range(output_features_dim):
                test_final_error_denormalised = denormalize(np.array(test_mse[-1][task]).reshape(-1,1),scaler[task])[0][0]
                plt.plot(train_sizes, [train_mse[i][task] for i in range(len(train_mse))], 
                         label=f'Training Error | Task: {output_features[task]}', marker='o')
                plt.plot(train_sizes, [test_mse[i][task] for i in range(len(test_mse))], 
                         label=f'Test Error | Task: {output_features[task]} \nFinal Error = {test_mse[-1][task]:.2f} [{Error_unit}]', 
                         marker='*')
            annotation_y = 0.5

        textstr = (f'Input features: {", ".join(input_features)}\n'
                   f'Output features: {", ".join(output_features)}\n'
                   f'Kernel: {kernel.__class__.__name__}\n'
                   f'Training iterations= {training_iter} | $L_r$= {lr}\n'
                   f'Normalisation Method: {norm_method_name}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
        plt.text(1, annotation_y, textstr, fontsize=10, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.xlabel('Training Set Size')
        plt.ylabel(f'{Error_Caption} [{Error_unit}]')
        plt.title(f'Learning Curve: Prediction Error vs Training Set Size')
        plt.legend()
        plt.grid(True)
        timestamp = now.strftime("%d-%m-%y-%H-%M-%S")
        save_dir_temp = f'{results_path}/{clean_textstr(kernel.__class__.__name__)}/'
        check_create_dir(save_dir_temp)
        filename = save_dir_temp +f'Learning Curve {clean_textstr(textstr)}.png'
        plt.savefig(filename)

        #### Confidence-only learning curve
        plt.figure(figsize=(10, 6))
        if output_features_dim == 1:
            plt.plot(train_sizes, test_confidence, 
                     label=f'Test Average Confidence Margin\nFinal Average Confidence Margin = {test_confidence[-1]:.2f}', 
                     color='red', marker='o')
            annotation_y = 0.85
        else:
            for task in range(output_features_dim):
                plt.plot(train_sizes, [test_confidence[i][task] for i in range(len(test_mse))], 
                         label=f'Test Average Confidence Margin | Task: {output_features[task]} \nFinal Average Confidence Margin= {test_confidence[-1][task]:.2f}', 
                         marker='*')
            annotation_y = 0.5

        textstr = (f'Input features: {", ".join(input_features)}\n'
                   f'Output features: {", ".join(output_features)}\n'
                   f'Kernel: {kernel.__class__.__name__}\n'
                   f'Training iterations= {training_iter} | $L_r$= {lr}\n'
                   f'Normalisation Method: {norm_method_name}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
        plt.text(1, annotation_y, textstr, fontsize=10, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.xlabel('Training Set Size')
        plt.ylabel(f'Confidence Margin [Normalised]')
        plt.title(f'Learning Curve: Confidence Margin vs Training Set Size')
        plt.legend()
        plt.grid(True)
        timestamp = now.strftime("%d-%m-%y-%H-%M-%S")
        save_dir_temp = f'{results_path}/{clean_textstr(kernel.__class__.__name__)}/'
        check_create_dir(save_dir_temp)
        filename = save_dir_temp +f'Test Confidence Curve {clean_textstr(textstr)}.png'
        plt.savefig(filename)
        plt.show()

        #### Loss curve
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.plot(train_sizes, Loss, color="blue", marker="o")
        ax.set_xlabel("Training Iterations")
        ax.set_ylabel("Negative Marginal Log-Likelihood (Loss)")
        ax.set_title("Hyperparameters Optimisation: Loss Convergence")
        textstr = (f'Input features: {", ".join(input_features)}\n'
                f'Output features: {", ".join(output_features)}\n'
                f'Kernel: {kernel.__class__.__name__}\n'
                f'Training iterations = {training_iter} | $L_r$ = {lr}\n'
                f'Normalisation Method: {norm_method_name}')
        props = dict(boxstyle='round', facecolor='white', alpha=0.25)
        ax.text(0.99, 0.95,f"{textstr}\n{optimisation_results}", fontsize=10, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        save_dir_temp = f'{results_path}/{clean_textstr(kernel.__class__.__name__)}/'
        check_create_dir(save_dir_temp)
        filename = save_dir_temp + f'Loss Curve-{clean_textstr(textstr)}.png'
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

        # Final model on full training set
        print(f'Train x size - Final: {train_x.shape}')
        print(f'Train y size - Final: {train_y.shape}')
        likelihood, model,final_loss , optimisation_results = train_GPR_model(
            train_x,train_y, lr, training_iter, kernel, Noise_level, early_stopping_tolerance,textstr,results_path)
        
        if input_features_dim == 2:
            print(f'input_features_dim = {input_features_dim}')
            if norm_method_name=='Min-max':
                x1_grid = np.linspace(0, 1, 1000)
                x2_grid = np.linspace(0, 1, 1000)
            else:
                x1_grid = np.linspace(min(train_x[:,0]), max(train_x[:,0]), 1000)
                x2_grid = np.linspace(min(train_x[:,0]), max(train_x[:,0]), 1000)
            x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
            grid_x_2D = torch.tensor(np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T, dtype=torch.float32)
            grid_mean_2D, grid_lower, grid_upper = make_GPR_predictions(model, grid_x_2D)

        if input_features_dim == 1:
            print(f'Number of input features = {input_features_dim}')
            grid_x = torch.linspace(train_x.min(), train_x.max(), 100)
            grid_x = torch.tensor(grid_x, dtype=torch.float32)
            grid_mean, grid_lower, grid_upper = make_GPR_predictions(model, grid_x)
        
        if len(exp_x) > 0:
            print(f"Experimental input = {exp_x}")
            exp_y, lower_exp, upper_exp = make_GPR_predictions(model, exp_x)
            print(f'\n--------------EXPERIMENTAL VALIDATION------------------------\n')                
            print(f'\n--------------NORMALISED------------------------\n')                
            print(f'Experimental predictions (Normalised): {exp_y}') 
            print(f'Experimental predictions Confidence (Normalised): {upper_exp - lower_exp}')

        ## Plots of predictions
        for i in range(output_features_dim):
            z_label = output_features[i]
            plot_title = f'GPR Predictions'
            if input_features_dim == 2:
                x_label, y_label = input_features[0], input_features[1]
                grid_mean_2D_i = grid_mean_2D[:,i].reshape(x1_grid.shape)
                test_mse = np.array(test_mse)
                print(f'train_x: {train_x.shape}')
                print(f'train_y: {train_y.shape}')
                print(f'test_x: {test_x.shape}')
                print(f'exp_x: {test_y.shape}')
                print(f'test_mse: {test_mse.shape}')
                print(f'grid_mean_2D_i: {grid_mean_2D_i.shape}')
                
                plot_gpr_predictions_3d_and_contour(
                    train_x, train_y[:,i], test_x, test_y[:,i], 
                    exp_x, exp_y[:,i], 
                    x1_grid, x2_grid, grid_mean_2D_i,
                    kernel, test_mse[-1], 
                    x_label, y_label, z_label, 
                    results_path, plot_title, textstr)
                
            if input_features_dim == 1:
                x_label, y_label = input_features[0], output_features[i]
                plot_gpr_predictions_2d_xy(
                        train_x, train_y[:,i], 
                        test_x, test_y[:,i], 
                        grid_x, grid_mean[:,i],
                        grid_lower[:,i], grid_upper[:,i],
                        exp_x, exp_y[:,i], lower_exp[i], upper_exp[i],
                        kernel, test_mse, 
                        x_label, y_label, 
                        results_path, plot_title,textstr)
                
            if input_features_dim > 2:
                print(f'Input features = {input_features_dim}. No plot')

            plt.figure(figsize=(10, 6))

            if exp_prediction_path:
                exp_plot_title = f'GPR Predictions for {", ".join(output_features)} from {", ".join(input_features)}'
                print(f'Scaler:{scaler}')
                print(f'LENGHT OF OUTPUR FEATURES = {len(output_features)}')
                for i in range(len(output_features)):
                    if len(exp_y_truth)>0:
                        exp_y_i = exp_y[:,i].reshape(-1,1)
                        lower_exp_i = lower_exp[:,i].reshape(-1,1)
                        upper_exp_i = upper_exp[:,i].reshape(-1,1)
                        exp_y_truth_i = [exp_y_truth[key][i] for key in exp_y_truth.keys()]
                        exp_y_truth_o_i = np.array([denormalize(exp_y_truth[key][i], scaler[i]) for key in exp_y_truth.keys()])
                    else:
                        exp_y_i = exp_y[:,i].reshape(-1,1)
                        lower_exp_i = lower_exp[:,i].reshape(-1,1)
                        upper_exp_i = upper_exp[:,i].reshape(-1,1)
                        exp_y_truth = []

                    print(f'\n----------------DENOMALISE PREDICTIONS----------------------\n')        
                    print(f'Output Feature: {output_features[i]}\nScaler: {scaler[i]}')

                    exp_y_o = denormalize(exp_y_i, scaler[i])
                    lower_exp_o = denormalize(lower_exp_i, scaler[i])
                    upper_exp_o = denormalize(upper_exp_i, scaler[i])


                    print(f'Experimental Predictions (denormalized): {exp_y_o}')
                    print(f'Experimental Confidence (denormalized): {upper_exp_o-lower_exp_o}')
                    print(f'Experimental Lower bound (denormalized): {lower_exp_o}')
                    print(f'Experimental Upper bound (denormalized): {upper_exp_o}')

                    if len(exp_y_truth)>0:
                        exp_y_truth_i = np.array([val.flatten()[0] for val in exp_y_truth_i])
                        try: 
                            print(f'Truth: {exp_y_truth_i.flatten()} \n Predictions: {exp_y_i.flatten()}')
                        except:
                            print(f'Truth: {exp_y_truth_i} \n Predictions: {exp_y_i.flatten()}')

                        try:
                            exp_mse_percentage = Error(exp_y_truth_i, exp_y_i.flatten())   
                        except:
                            print(f'exp_y_truth_i:{exp_y_truth_i}')
                            exp_mse_percentage = Error(exp_y_truth_i, exp_y_i)   

                        print(f'Experimental Error [Normalised] = {exp_mse_percentage}')
                        print(f'Truth (original): {exp_y_truth_o_i} \n Predictions (original): {exp_y_o}') 
                        exp_y_truth_o_i = exp_y_truth_o_i.flatten()
                        exp_y_0_flat = exp_y_o.flatten() 
                        exp_error_og = Error(exp_y_truth_o_i, exp_y_0_flat )                            
                        print(f'Experimental Error Original = {exp_error_og}')

                    prediction_std = np.std(exp_y_i)
                    ymin = min(lower_exp_o)*0.75
                    ymax = max(upper_exp_o)*1.25

                    if len(exp_y_truth)>0:
                        plot_exp_predictions(exp_file_path, results_path, 
                                            exp_y_o,prediction_std,
                                            upper_exp_o, 
                                            lower_exp_o, 
                                            exp_mse_percentage,exp_error_og,
                                            ymin, ymax, kernel, 
                                            exp_plot_title, output_unit,
                                            output_features[i], 
                                            exp_y_truth_o_i, 
                                            exp_y_label, textstr)
                    else:
                        plot_exp_predictions(exp_file_path, results_path, 
                                            exp_y_o,prediction_std,
                                            upper_exp_o, 
                                            lower_exp_o, 
                                            [],[],
                                            ymin, ymax, kernel, 
                                            exp_plot_title, output_unit,
                                            output_features[i], 
                                            [], 
                                            exp_y_label, textstr)
        else:
            exp_y, lower_exp, upper_exp = [],[],[]   

        MSE_all.append(test_mse_percentage)
        MSE_all_training.append(train_mse_percentage)
        print(f'Test confidence updated {kernel}')
        test_confidence_all.append(test_confidence_i)

        print(test_confidence_all)
        print(f'TEST CONFIDENCE SIZE {len(test_confidence_all)}')

    least_MSE = min(MSE_all)
    least_MSE_training = min(MSE_all_training)
    least_MSE_index = MSE_all.index(least_MSE)
    least_MSE_kernel = kernels[least_MSE_index]

    least_test_confidence = min(test_confidence_all)
    least_confidence_index = test_confidence_all.index(least_test_confidence)
    least_confidence_kernel = kernels[least_confidence_index]

    print(f'Least MSE = {least_MSE}')
    print(f'Least MSE Kernel= {least_MSE_kernel}')

    return model, least_MSE, least_MSE_kernel, least_MSE_training, exp_y_o, lower_exp_o, upper_exp_o, exp_y, lower_exp, upper_exp, exp_mse_percentage, least_test_confidence, least_confidence_kernel


# ===== Error Metrics =====

def Error(pred,truth):
    """Wrapper: compute RMSE (labels set globally for plotting)."""
    mse, rmse, mae, mape, nrmse = All_Error(pred, truth)
    global Error_Caption, Error_unit
    Error_unit = 'Normalised'
    Error_Caption = f'RMSE'
    return rmse


def All_Error(pred, truth):
    """Compute MSE / RMSE / MAE / MAPE / NRMSE on numpy arrays."""
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    mse = mean_squared_error(truth, pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs((truth - pred) / 1)) 
    mape = np.mean(np.abs((truth - pred) / truth))  * 100
    nrmse = (rmse / (truth.max() - truth.min())) * 100
    return mse, rmse, mae, mape, nrmse


# ===== Data Preparation =====

def prepare_training_data(X,y, test_data_portion):
    """Train/test split then cast to torch tensors."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_portion, random_state=42)
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    test_x = torch.tensor(X_test, dtype=torch.float32)
    test_y = torch.tensor(y_test, dtype=torch.float32)
    return train_x, train_y, test_x, test_y


def check_create_dir(directory):
    """Create directory (parents=True) if missing; print status."""
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory created. \n {directory} ")
    else:
        print(f"Directory already exists.\n {directory} ")


# ===== Plotting Functions =====

def plot_gpr_predictions_3d_and_contour(
        train_x, train_y, test_x, test_y, 
        exp_x, exp_y, x1_grid, x2_grid, grid_mean,
        kernel, mse_test_percentage, 
        x_label, y_label, z_label, folder_path, plot_title, textstr):
    """3D surface + 2D contour visualisation of 2D-input GPR predictions."""
    print(f'exp_y:{exp_y.shape}')
    print(f'exp_x:{exp_x.shape}')
    fig = plt.figure(figsize=(16, 8))

    # 3D scatter + surface
    ax1 = fig.add_subplot(121, projection='3d')
    train_z = train_y.numpy()

    # guard against shape mismatch
    print(f"Shape of train_x: {train_x.shape}")
    print(f"Shape of train_y: {train_y.shape}")
    if train_x.shape[0] != train_y.shape[0]:
        print("Mismatch in number of samples between train_x and train_y.")
        return

    ax1.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), train_z, 
                c=train_z, cmap='viridis', edgecolor='k', s=50, label='Training Data')

    # experimental points (optional)
    if exp_x is not None and len(exp_x) > 0:
        try:
            ax1.scatter(exp_x[:, 0].numpy(), exp_x[:, 1].numpy(), exp_y, 
                        cmap='viridis', edgecolor='k', s=100, label='Experimental Validation')
        except AttributeError:
            ax1.scatter(exp_x[:, 0], exp_x[:, 1], exp_y, 
                        cmap='viridis', edgecolor='k', s=100, label='Experimental Validation')

    # test scatter + predicted surface
    test_z = test_y.numpy()
    ax1.scatter(test_x[:, 0].numpy(), test_x[:, 1].numpy(), test_z, 
                c=test_z, cmap='viridis', edgecolor='k', s=50, marker='^', 
                label='Test Data')
    ax1.plot_surface(x1_grid, x2_grid, grid_mean, cmap='viridis', alpha=0.5)
    ax1.set_xlabel(x_label); ax1.set_ylabel(y_label); ax1.set_zlabel(z_label)
    ax1.legend()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text2D(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

    # 2D contour + scatter
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(x1_grid, x2_grid, grid_mean, cmap='viridis')
    plt.colorbar(contour, ax=ax2)
    ax2.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), c=train_y, cmap='viridis', 
                edgecolor='k', s=50, label='Training Data')
    ax2.scatter(test_x[:, 0].numpy(), test_x[:, 1].numpy(), c=test_z, cmap='viridis', 
                edgecolor='k', s=50, label='Test Data')
    if exp_x is not None and len(exp_x) > 0:
        try:
            ax2.scatter(exp_x[:, 0].numpy(), exp_x[:, 1].numpy(), c=exp_y, cmap='viridis', 
                        edgecolor='k', s=50, marker='x', label='Experimental Validation')
        except AttributeError:
            ax2.scatter(exp_x[:, 0], exp_x[:, 1], c=exp_y, cmap='viridis', 
                        edgecolor='k', s=50, marker='x', label='Experimental Validation')

    x1g = np.asarray(x1_grid); x2g = np.asarray(x2_grid)
    ax2.set_xlim(x1g.min(), x1g.max()); ax2.set_ylim(x2g.min(), x2g.max())
    ax2.set_xlabel(x_label); ax2.set_ylabel(y_label); ax2.legend()

    plt.suptitle(plot_title)
    now = datetime.now()
    timestamp = now.strftime('%d-%m-%y-%H-%M-%S')
    folder_path = f'{folder_path}/{kernel.__class__.__name__}/'
    filename = f'{folder_path}GPR Prediction 2D-{clean_textstr(textstr)}.png'
    plt.savefig(filename); plt.show()
    print(f'Plot saved to: {filename}')


def plot_gpr_predictions_2d_xy(
        train_x, train_y, test_x, test_y, 
        x_grid,grid_mean,grid_lower, grid_upper,
        exp_x, exp_y, lower_exp, upper_exp,
        kernel, mse_test_percentage, 
        x_label, y_label, folder_path, plot_title,textstr):
    """1D-input GPR visualisation: mean + confidence band + scatter for train/test/exp."""
    print(f'exy_y shape: {exp_y.shape}')
    print(f'exp_x shape: {exp_x.shape}')
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(x_grid, grid_mean, label='Predicted Mean', color='blue')
    ax.fill_between(x_grid, grid_lower, grid_upper, alpha=0.2, color='blue', label='Confidence Region')
    plt.scatter(train_x.numpy(), train_y.numpy(), label='Training Data', color='green', s=5)
    plt.scatter(test_x.numpy(), test_y.numpy(), color='red',s=5)

    # experimental predictions with error bars
    if exp_x is not None and len(exp_x) > 0:        
        print('Error bars only ')
        plt.errorbar(exp_x, exp_y, 
                     yerr=[abs(exp_y - lower_exp), abs(upper_exp - exp_y)], 
                     fmt='o', markersize = 5,color='black')

    ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.legend()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.85, 0.1, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.title(plot_title)
    now = datetime.now()
    timestamp = now.strftime('%d-%m-%y-%H-%M-%S')
    filename = f'{folder_path}GPR_Prediction_2D_XY-{clean_textstr(textstr)}.png'
    plt.savefig(filename); plt.show()
    print(f'Plot saved to: {filename}')


# ===== GPR Prediction and Utilities =====

def make_GPR_predictions(model, input):
    """Run model(input) and return mean + (lower, upper) confidence region (handles single/multi-task)."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model(input)
        mean = predictions.mean  
        lower, upper = predictions.confidence_region()

        if mean.ndim > 1:  # multi-task
            means = mean.numpy()
            lowers = lower.numpy()
            uppers = upper.numpy()
        else:  # single-task
            means = mean.numpy().reshape(-1, 1)
            lowers = lower.numpy().reshape(-1, 1)
            uppers = upper.numpy().reshape(-1, 1)

    return means, lowers, uppers


def _flatten_np(t, max_items=6):
    """Pretty-print a small slice of a tensor for annotations."""
    try:
        arr = t.detach().cpu().view(-1).numpy()
    except Exception:
        return "N/A"
    if arr.size == 0:
        return "[]"
    if arr.size <= max_items:
        return "[" + ", ".join(f"{v:.3g}" for v in arr) + "]"
    head = ", ".join(f"{v:.3g}" for v in arr[: max_items // 2])
    tail = ", ".join(f"{v:.3g}" for v in arr[-(max_items // 2):])
    return f"[{head}, …, {tail}]"


def _collect_hyperparams(model, likelihood):
    """Collect constrained hyperparameters (likelihood + (composite) kernels) for annotation."""
    hp = {}
    try:
        if hasattr(likelihood, "noise"):
            hp["lik.noise"] = likelihood.noise
    except Exception:
        pass
    try:
        if hasattr(likelihood, "task_noises"):
            hp["lik.task_noises"] = likelihood.task_noises
    except Exception:
        pass
    
    def _collect_from_kernel(kern, prefix="kern"):
        d = {}
        if kern is None:
            return d
        cname = kern.__class__.__name__

        if isinstance(kern, gpytorch.kernels.ScaleKernel):
            if hasattr(kern, "outputscale"):
                d["Kernel Outputscale"] = kern.outputscale
            base = getattr(kern, "base_kernel", None)
            d.update(_collect_from_kernel(base))
            return d

        if hasattr(kern, "kernels"):
            for ck in kern.kernels:
                d.update(_collect_from_kernel(ck))
            return d

        if hasattr(kern, "lengthscale"):
            d[f"{cname} Lengthscale"] = kern.lengthscale
        if hasattr(kern, "alpha"):
            d[f"{cname} Alpha"] = kern.alpha
        if hasattr(kern, "period_length"):
            d[f"{cname} Period Length"] = kern.period_length
        if hasattr(kern, "variance"):
            d[f"{cname} Variance"] = kern.variance
        if hasattr(kern, "bias"):
            d[f"{cname} Bias"] = kern.bias
        return d

    try:
        cov = getattr(model, "covar_module", None)
        if cov is not None:
            hp.update(_collect_from_kernel(cov, prefix="kern"))
    except Exception:
        pass

    # Back-compat convenience aliases
    if "kern.lengthscale" not in hp:
        for k, v in hp.items():
            if k.endswith(".lengthscale"):
                hp["kern.lengthscale"] = v
                break
    if "kern.outputscale" not in hp:
        for k, v in hp.items():
            if k.endswith(".outputscale"):
                hp["kern.outputscale"] = v
                break

    return hp


def clean_textstr(textstr):
    """Sanitise text for filenames (remove newlines/special chars and compress underscores)."""
    replacements = [
        '\n','Input features:','Output features:','Kernel:','Training iterations=','Training iterations',
        'Normalisation Method:',',','=','\'','$','\\','{','}',
    ]
    for r in replacements:
        textstr = textstr.replace(r, '_')
    textstr = textstr.replace(' ', '')
    textstr = textstr.replace('|', '_')
    while '__' in textstr:
        textstr = textstr.replace('__', '_')
    return textstr.strip('_')


# ===== Model Training Functions =====

def train_GPR_model(
    train_x,
    train_y,
    lr,
    training_iter,
    kernel,
    Noise_level,
    early_stopping_tolerance,textstr,save_dir):
    """Fit single-/multi-task Exact GP with Adam; log loss trace and annotate hyperparameters."""
    # visualisation controls
    record_history: bool = True
    plot: bool = True
    plot_every=0
    fig_ax=None
    title: str = "GPR Training: Negative MLL vs Iteration"

    # sanitise plotting args
    try:
        plot = bool(plot)
    except Exception:
        plot = True if str(plot).lower() in ("1", "true", "yes") else False
    try:
        if isinstance(plot_every, (list, tuple)):
            plot_every = plot_every[0] if len(plot_every) > 0 else 0
        plot_every = int(plot_every)
    except Exception:
        plot_every = 0

    # task count
    output_features_dim = train_y.shape[1] if train_y.ndim == 2 else 1

    # build likelihood + model
    if output_features_dim == 1:
        y_vec = train_y[:, 0] if train_y.ndim == 2 else train_y
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(Noise_level)
        )
        model = GPRegressionModel(train_x, y_vec, likelihood, kernel)
    else:
        noise_constraint = gpytorch.constraints.GreaterThan(Noise_level)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=output_features_dim,
            noise_constraint=noise_constraint
        )
        model = MultitaskGPRegressionModel(
            train_x, train_y, likelihood,
            num_tasks=output_features_dim,
            base_kernel=kernel
        )

    init_hp = _collect_hyperparams(model, likelihood)

    # training loop
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    prev_loss = float('inf')
    history = {"iter": [], "loss": []} if record_history else None
    do_live = plot and (plot_every > 0)

    # live plot setup (optional)
    if do_live:
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(7.8, 5.4))
        else:
            if isinstance(fig_ax, plt.Axes):
                ax = fig_ax; fig = ax.figure
            else:
                fig, ax = fig_ax
        ax.set_title(str(title)); ax.set_xlabel("Iteration"); ax.set_ylabel("Negative MLL (loss)")
        line, = ax.plot([], [], lw=1.6, marker="o", markersize=2.5)
        ax.grid(True, alpha=0.3); plt.tight_layout(); plt.pause(0.001)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, y_vec) if output_features_dim == 1 else -mll(output, train_y)
        loss.backward(); optimizer.step()
        cur = float(loss.item())
        if record_history:
            history["iter"].append(i); history["loss"].append(cur)

        # live refresh
        if do_live and (i % plot_every == 0 or i == training_iter - 1):
            xs = history["iter"] if record_history else list(range(i + 1))
            ys = history["loss"] if record_history else [cur]
            line.set_xdata(xs); line.set_ydata(ys)
            if xs: ax.set_xlim(0, max(xs) if xs else 1)
            if ys:
                ymin, ymax = min(ys), max(ys)
                if ymin == ymax: ymin -= 1e-3; ymax += 1e-3
                margin = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - margin, ymax + margin)
            plt.draw(); plt.pause(0.001)

        # early stopping tolerance on loss change
        if abs(prev_loss - cur) < early_stopping_tolerance:
            print(f'Early stopping at iteration {i}, loss stabilized (Δ={abs(prev_loss - cur):.3e}).')
            break
        prev_loss = cur

    # evaluation mode
    model.eval(); likelihood.eval()
    final_hp = _collect_hyperparams(model, likelihood)

    # static plot (if no live)
    if plot and (not do_live):
        if record_history and len(history["iter"]) > 0:
            if fig_ax is None:
                fig, ax = plt.subplots(figsize=(7.8, 5.4))
            else:
                if isinstance(fig_ax, plt.Axes):
                    ax = fig_ax; fig = ax.figure
                else:
                    fig, ax = fig_ax
            ax.plot(history["iter"], history["loss"], lw=1.8, marker="o", markersize=2.5)
            ax.set_title(str(title)); ax.set_xlabel("Iteration"); ax.set_ylabel("Negative MLL"); ax.grid(True, alpha=0.3)

    # annotate init→final hyperparameters
    if plot:
        ann_lines = []
        keys = sorted(set(list(init_hp.keys()) + list(final_hp.keys())))
        for k in keys:
            v0 = init_hp.get(k, None); v1 = final_hp.get(k, None)
            s0 = _flatten_np(v0) if v0 is not None else "N/A"
            s1 = _flatten_np(v1) if v1 is not None else "N/A"
            ann_lines.append(f"{k}: {s0} → {s1}")
        if len(ann_lines) > 0:
            txt1 = "\n".join(ann_lines)
            txt = f"{textstr}\n{txt1}"
            ax.text(0.99, 0.98, txt, transform=ax.transAxes, va="top", ha="right",
                    fontsize=9, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.35"))
            plt.tight_layout()

        # save annotated loss plot
        save_dir = f'{save_dir}/{kernel.__class__.__name__}/'
        if isinstance(save_dir, str) and len(save_dir.strip()) > 0:
            try:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                fname = f"MLL_{clean_textstr(textstr)}.png"
                out_path = os.path.join(save_dir, fname)
                fig = ax.figure
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"[GPR] Training plot saved to: {out_path}")
            except Exception as e:
                print(f"[GPR] Could not save training plot to optimisation_save_dir: {e}")

        if fig_ax is None:
            try:
                plt.show()
            except Exception:
                pass

    return likelihood, model, history["loss"][-1], txt1


def plot_exp_predictions(
        dir_path, save_dir_path, 
        mean, mean_std,upper, lower, 
        exp_mse_percentage,exp_error_og,
        ymin, ymax, kernel,
        plot_title, output_unit,
        param, 
        truth, truth_label, 
        textstr):
    """Per-experiment bar/error plot of de-normalised predictions vs (optional) truth."""
    RPV = False
    print(f'Output feature range: {ymin}-{ymax}')

    files = [os.path.splitext(f)[0] for f in os.listdir(dir_path) if f.endswith('.csv')]
    if 'Eu' in files:
        file_name = [files[i][:-8] for i in range(len(files))]
    else:
        file_name = [files[i][:-6] for i in range(len(files))]
    
    file_name = sorted(file_name)
    if truth_label!='': 
        file_name = sorted(file_name)
    else:
        print('nothing')

    plt.figure(figsize=(16, 8))
    mean = np.array(mean); upper = np.array(upper); lower = np.array(lower)

    # scatter + error bar per experiment
    for i in range(len(mean)):
        if i ==0:
            print(f'Length of truth vector ={len(truth)}')
            print(f'Length of mean vector ={len(mean)}')
        if len(truth) == 0:
            print('No refernce values')
        elif len(truth) > 0:
            truth_i = truth[i]
            try:
                print(f"Processing index {i}, mean size: {len(mean)}, mean_i: {mean[i]}, truth_i: {truth[i]}")
                if RPV == True:
                    plt.scatter([i+1], mean[i], marker='o')
                    plt.scatter([i+1], truth_i, marker='x', s=60, color='red')
                else:
                    plt.scatter([i+1], mean[i], marker='o')
                    plt.scatter([i+1], truth_i, marker='x', s=60, color='red', label=truth_label[i])
            except IndexError:
                print('Fine')
                plt.scatter([i+1], truth_i, marker='x', s=60, color='red')

        print(f'Err length [De-Normalised] {-lower[i]+upper[i]}')
        plt.errorbar([i + 1], mean[i],
                     yerr=[abs(mean[i] - lower[i]), abs(upper[i] - mean[i])],
                     fmt='o', alpha=1, label='GPR Prediction' if i == 0 else "")

    legend = plt.legend(loc='best')
    legend_bbox = legend.get_bbox_to_anchor().transformed(plt.gca().transAxes)
    annotation_x = 0.95 if legend_bbox.x0 < 0.5 else 0.05
    annotation_y = 0.95 if legend_bbox.y0 < 0.5 else 0.05

    # optional annotation with error summary
    if exp_mse_percentage>0:
        if truth_label[0][:3] == 'Ten':
            plt.annotate(f'GPR Prediction vs Tensile Test {Error_Caption} = {exp_mse_percentage:.2f} [{Error_unit}] \n{txtstr}',
                         xy=(annotation_x, annotation_y), xycoords='axes fraction', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray", alpha = 0.24),
                         ha='right' if annotation_x > 0.5 else 'left',
                         va='top' if annotation_y > 0.5 else 'bottom')
        if truth_label[0][:3] == 'Inv':
            plt.annotate(f'GPR Prediction vs Inverse Analysis {Error_Caption} = {exp_mse_percentage:.2f} [{Error_unit}] \n{txtstr}',
                         xy=(annotation_x, annotation_y), xycoords='axes fraction', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray", alpha = 0.24),
                         ha='right' if annotation_x > 0.5 else 'left',
                         va='top' if annotation_y > 0.5 else 'bottom')
    else:
        plt.annotate(f'{textstr}', xy=(annotation_x, annotation_y), xycoords='axes fraction', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray", alpha = 0.24),
                     ha='right' if annotation_x > 0.5 else 'left',
                     va='top' if annotation_y > 0.5 else 'bottom')

    # cosmetics + save
    plt.xticks(ticks=range(1, len(files) + 1),labels=[file_name[i][:-2] for i in range(len(file_name))],rotation=15,ha="right")
    plt.ylim(ymin, ymax)
    plt.xlabel('Experiment Number'); plt.ylabel(f'{param}')
    plt.title(plot_title); plt.legend()
    now = datetime.now(); timestamp = now.strftime("%d-%m-%y-%H-%M-%S")
    save_dir_path = f"{save_dir_path}{kernel.__class__.__name__}/"
    try: 
        filename = f'{save_dir_path}Experimental prediction_{clean_textstr(textstr)}.png'
        plt.savefig(filename)
    except:
        filename = f'{save_dir_path}Experimental prediction_{clean_textstr(textstr)}.png'
        print(filename); plt.savefig(filename)
    print(f'Plot saved to:{filename}')
    plt.show()

