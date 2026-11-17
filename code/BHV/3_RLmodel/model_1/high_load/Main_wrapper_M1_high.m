%% Fit the 7-parameter model to the participant data (HIGH LOAD)
% this model has two w-parameter:
% (1) w_small for small reward (1×)
% (2) w_large for large reward (5×)

%% 1. Setup paths
analysis_dir = fileparts(mfilename('fullpath'));
cd(analysis_dir);
addpath(fullfile(analysis_dir, '..', '..', 'mfit-master'))
addpath('helper_scripts')

fprintf('Step 1. Fitting model to participant data (HIGH LOAD condition).\n\n\n')

%% 2. Fit participant data to model
% This function is the modified parfor wrapper
[results, RPEs, SPEs] = Fit_model_wrapper_parfor_M1_high;

fprintf('Step 2. Model fitting complete.\n');

%% 3. write .mat to csv
% This section now calculates Pseudo-R2, Log-Evidence, 
% and expands the parameters (x) into separate columns.

fprintf('Step 3. Exporting extended model results to CSV...\n\n');

% --- 3.1. Convert results struct to table ---
T = struct2table(results);

% --- 3.2. Expand the parameter vector 'x' into named columns ---
T.beta = T.x(:, 1);
T.alpha = T.x(:, 2);
T.lambda = T.x(:, 3);
T.w_small = T.x(:, 4);
T.w_large = T.x(:, 5);
T.st = T.x(:, 6);
T.respst = T.x(:, 7);

% --- 3.3. Calculate new metrics ---
% Calculate LL_random (Log-Likelihood of a chance model)
T.LL_random = T.trials .* log(0.5);
% Calculate Pseudo R-squared: Pseudo_R2 = 1 - (LL_model / LL_random)
T.Pseudo_R2 = 1 - (T.loglik ./ T.LL_random);
% Calculate Log-Evidence (approximate): LME = -0.5 * BIC
T.Log_Evidence = -0.5 .* T.bic;


% --- 3.4. Select all useful columns for export ---
all_columns = T.Properties.VariableNames;

% Define the columns we want in the final CSV, in a logical order
columns_to_save = {
    'id', 'trials', ...
    'beta', 'alpha', 'lambda', 'w_small', 'w_large', 'st', 'respst', ... % Expanded params
    'aic', 'bic', 'loglik', 'logpost', ...                 % Fit metrics
    'LL_random', 'Pseudo_R2', 'Log_Evidence'              % New calculated metrics
};

columns_to_save = intersect(columns_to_save, all_columns, 'stable');
selectedColumns = T(:, columns_to_save);

% --- 3.5. Write to CSV ---
output_filename = 'results_M1_high.csv';
writetable(selectedColumns, output_filename);

fprintf('Successfully saved results to: %s\n', output_filename);
fprintf('--- All steps complete ---\n');

% --- 4. Save RPEs/SPEs (trial-by-trial) ---
% (This part is unchanged from your original script, just paths updated)

% save trial-based RPEs
output_dir_RPEs = fullfile(analysis_dir, 'RPE');
if ~isfolder(output_dir_RPEs)
    mkdir(output_dir_RPEs);
end

for i = 1:length(RPEs)
    current_subject = RPEs(i);
    tbl = table();
    tbl.id = repmat(current_subject.id,current_subject.trials,1);
    tbl.Trial = (1:current_subject.trials)';
    tbl.RPE = current_subject.RPE_values(:);
    filename = sprintf('%s/subject_%d_RPE.csv', output_dir_RPEs, current_subject.id);
    writetable(tbl, filename);
end

% save trial-based SPEs
output_dir_SPEs = fullfile(analysis_dir, 'SPE');
if ~isfolder(output_dir_SPEs)
    mkdir(output_dir_SPEs);
end

for i = 1:length(SPEs)
    current_subject = SPEs(i);
    tbl = table();
    tbl.id = repmat(current_subject.id,current_subject.trials,1);
    tbl.Trial = (1:current_subject.trials)';
    tbl.SPE = current_subject.SPE_values(:);
    filename = sprintf('%s/subject_%d_SPE.csv', output_dir_SPEs, current_subject.id);
    writetable(tbl, filename);
end

% --- 5. Save RPEs/SPEs (block-based) ---
T_RPEs = struct2table(RPEs);
selectedColumns_RPEs = T_RPEs(:, {'id', 'trials','abs_mean_RPE','mean_RPE','std_RPE'});
writetable(selectedColumns_RPEs, fullfile(output_dir_RPEs, 'RPEs_M1_high.csv'));

T_SPEs = struct2table(SPEs);
selectedColumns_SPEs = T_SPEs(:, {'id', 'trials','abs_mean_SPE','mean_SPE','std_SPE'});
writetable(selectedColumns_SPEs, fullfile(output_dir_SPEs, 'SPEs_M1_high.csv'));

clear T T_RPEs T_SPEs params_matrix selectedColumns* all_columns columns_to_save output_filename;

%% play a sound when finished
load gong
sound(y,Fs)