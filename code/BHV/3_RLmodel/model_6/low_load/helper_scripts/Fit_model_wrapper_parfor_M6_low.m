%% Notes for functionmodel_4
% Chengyan Yang, November 2025

function [results,RPEs,SPEs] = Fit_model_wrapper_parfor_M6_low

%% Step 1. run for sample    
helper_dir = fileparts(mfilename('fullpath'));
load(fullfile(helper_dir, '..', '..', '..', '..', '..', '..', 'data', 'BHV', '2_stage_task', 'preprocessed', 'learning', 'groupdata_lload.mat'), 'groupdata');
participant_data = groupdata;

%% Step 2. Start analysing all subjects with a parfor loop
subs = length(participant_data.subdata(:));

RPEs(subs) = struct('id', [], 'RPE_values', [], 'trials', []);

SPEs(subs) = struct('id', [], 'SPE_values', [], 'trials', []);

    % random starting locations for optimizer
    nstarts = 25;

    parfor i = 1:subs

        data = participant_data.subdata(i);
        % Reproducible starts for this model, load and participant.
        fit_seed = 600000 + double(data.id);
        rng(fit_seed, 'twister');
        fprintf('Fitting participant no. %d\n\n',i)

        % run optimization
        params = set_params_M6;
        f = @(x,data) MB_MF_llik_M6_low(x,data);
        fit_result = mfit_optimize(f,params,data,nstarts);
        fit_result.fit_seed = fit_seed;
        results(i) = fit_result;

        [~, sub_RPEs, sub_SPEs] = MB_MF_llik_M6_low(results(i).x, data);
        
        RPEs(i).id = data.id;
        RPEs(i).RPE_values = sub_RPEs;
        RPEs(i).trials = length(sub_RPEs);
        RPEs(i).abs_mean_RPE = mean(abs(sub_RPEs));
        RPEs(i).mean_RPE = mean(sub_RPEs);
        RPEs(i).std_RPE = std(sub_RPEs);

        SPEs(i).id = data.id;
        SPEs(i).SPE_values = sub_SPEs;
        SPEs(i).trials = length(sub_SPEs);
        SPEs(i).abs_mean_SPE = mean(abs(sub_SPEs));
        SPEs(i).mean_SPE = mean(sub_SPEs);
        SPEs(i).std_SPE = std(sub_SPEs);

    end

%% Step 3. Save results per sample
results = results;
save results_M6_low.mat results
RPEs = RPEs;
save RPE_M6_low.mat RPEs
SPEs = SPEs;
save SPE_M6_low.mat SPEs

end

   