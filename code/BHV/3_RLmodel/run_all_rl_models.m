function run_all_rl_models()
% Fit models 1-7 under low and high load using their existing entry points.
root = fileparts(mfilename('fullpath'));
input_dir = fullfile(root, '..', '..', '..', 'data', 'BHV', '2_stage_task', 'preprocessed', 'learning');
for file = ["groupdata_lload.mat", "groupdata_hload.mat"]
    if ~isfile(fullfile(input_dir, file))
        error('Missing %s in %s. Run both CSV-to-MAT scripts first.', file, input_dir);
    end
end

for model = 1:7
    for level = ["low", "high"]
        wrapper = fullfile(root, sprintf('model_%d', model), level + "_load", ...
            sprintf('Main_wrapper_M%d_%s.m', model, level));
        fprintf('\nFitting model %d of 7 (%s load)...\n', model, level);
        run_wrapper(wrapper);
    end
end
fprintf('\nCompleted all 14 RL fits.\n');
end

function run_wrapper(wrapper)
% Each script has its own workspace and adds its own model-specific helpers.
original_dir = pwd;
original_path = path;
cleanup = onCleanup(@() restore_session(original_dir, original_path));
run(wrapper);
end

function restore_session(original_dir, original_path)
cd(original_dir);
path(original_path);
end
