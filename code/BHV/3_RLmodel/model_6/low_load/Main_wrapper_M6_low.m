%% Fit the 8-parameter model to the participant data
% this model has two w-parameter and two α-parameter:
% (1) w_small & α_small for small reward (1×)
% (2) w_large & α_large for large reward (5×)

%% 1. fit participant data to model
analysis_dir = fileparts(mfilename('fullpath'));
cd(analysis_dir);
addpath(fullfile(analysis_dir, '..', '..', 'mfit-master'))
addpath('helper_scripts')

fprintf('Step 2. Fitting model to participant data.\n\n\n')

disp(pwd)
[results, RPEs, SPEs] = Fit_model_wrapper_parfor_M6_low;

%% write .mat to csv
% save RL model results and paramters
T = struct2table(results);

T.beta = T.x(:, 1);
T.alpha_small = T.x(:, 2);
T.alpha_large = T.x(:, 3);
T.lambda = T.x(:, 4);
T.w_small = T.x(:, 5);
T.w_large = T.x(:, 6);
T.st = T.x(:, 7);
T.respst = T.x(:, 8);

% Calculate Pseudo R-squared: 1 - (LL_model / LL_random)
T.LL_random = T.trials .* log(0.5);
T.Pseudo_R2 = 1 - (T.loglik ./ T.LL_random);

% 3c. Calculate Log-Evidence (approximate)
% LME = -0.5 * BIC
T.Log_Evidence = -0.5 .* T.bic;

selectedColumns = T(:, { 'id', 'trials', ...
    'beta', 'alpha_small', 'alpha_large','lambda', 'w_small', 'w_large', 'st', 'respst', ... % Expanded params
    'aic', 'bic', 'loglik', 'logpost', ...                 % Fit metrics
    'LL_random', 'Pseudo_R2', 'Log_Evidence'});              % New calculated metrics
writetable(selectedColumns, 'results_M6_low.csv');

% save trial-based RPEs of each participants
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

% save trial-based SPEs of each participants
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

% save block-based RPEs of each participants
T_RPEs = struct2table(RPEs);
selectedColumns_RPEs = T_RPEs(:, {'id', 'trials','abs_mean_RPE','mean_RPE','std_RPE'});
writetable(selectedColumns_RPEs, fullfile(output_dir_RPEs, 'RPEs_M6_low.csv'));

T_SPEs = struct2table(SPEs);
selectedColumns_SPEs = T_SPEs(:, {'id', 'trials','abs_mean_SPE','mean_SPE','std_SPE'});
writetable(selectedColumns_SPEs, fullfile(output_dir_SPEs, 'SPEs_M6_low.csv'));


%% play a sound when finished
load gong
sound(y,Fs)