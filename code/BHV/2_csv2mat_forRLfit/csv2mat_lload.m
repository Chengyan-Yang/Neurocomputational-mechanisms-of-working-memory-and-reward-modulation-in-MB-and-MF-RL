%%% first script to transform the data format(from .csv to .mat).
% Edited by Chengyan Yang on 7 November 2025
% based on scripts written by Wouter Kool and CR Smid
% for questions: yangcy@psych.ac.cn


%% combine data files into one matrix
disp('Step 1. Combining data files.\n\n\n')
% need to create a log of removed participants here.
[groupdata] = groupanalysis;


function [groupdata] = groupanalysis

groupdata = struct('subdata', struct([]), 'N', [], 'id', []);
com_log = [];

method_dir = fileparts(mfilename('fullpath'));
learning_dir = fullfile(method_dir, '..', '..', '..', 'data', 'BHV', '2_stage_task', 'preprocessed', 'learning');
subject_dir = fullfile(learning_dir, 'low_load');
if ~isfolder(subject_dir)
    mkdir(subject_dir);
end

csv_files = dir(fullfile(subject_dir, '*.csv'));
disp(csv_files)


nrfiles = length(csv_files);

subject_ids = zeros(nrfiles, 1);
s = 0;
for a = 1:nrfiles
    data = readtable([csv_files(a).folder,'/',csv_files(a).name]);
    ids = unique(rmmissing(data.participant));
    if numel(ids) ~= 1 || ~ismember(ids, 1:52)
        error('Expected one participant ID from 1 to 52 in %s.', csv_files(a).name);
    end
    subject_ids(a) = ids;
    data = rmmissing(data);
    data(data.probe == 1, :) = [];
    [~, name, ~] = fileparts(csv_files(a).name);
    savePath = fullfile(subject_dir, [name, '.mat']);
    save(savePath, 'data');
end 


for i = 1:nrfiles
    
    fprintf('Analysing PP %d\n',i)
    
    [~, name] = fileparts(csv_files(i).name);
    mat_name = [name, '.mat'];
    load(fullfile(subject_dir, mat_name), 'data');
    
    subdata = [];
    log = [];
    log.id = subject_ids(i);
    log.trials_attempted = sum(data.choice ~= 0);
    
    if sum(~isnan(data.trials)) >= 130
        
        log.inclusion = 1;
        
        trials = 1:sum(~isnan(data.trials));
        
        % general info
        subdata.id = data.participant(1,1);
        subdata.dataFileName = mat_name;
        
        % task info
        subdata.rews(:,1) = data.rews_1;
        subdata.rews(:,2) = data.rews_2;
        subdata.s(:,1) = data.pairs;
        subdata.s(:,2) = data.animal;
        subdata.stimuli(:,1) = data.stimuli_left;
        subdata.stimuli(:,2) = data.stimuli_right;
        subdata.magnitude = data.magnitude;
        subdata.load = data.load;
        subdata.choice = data.choice;
        subdata.rt(:,1) = data.rt_1;
        subdata.rt(:,2) = data.rt_2;
        subdata.score = data.score;
        subdata.points = data.points;
        
        subdata.N = length(subdata.points);

        s = s + 1;
        if s == 1
            groupdata.subdata = subdata;
        else
            groupdata.subdata(s) = subdata;
        end
        groupdata.N(s,1) = subdata.N;
        groupdata.id(s,1) = subdata.id;

        
    else
        log.inclusion = 0;
        
    end
    com_log = [com_log, log];

    
end

% save and rename here
all_included = sum([com_log.inclusion]);
    
disp('How many pps included?: ');
disp(all_included);

groupdata = groupdata;
save(fullfile(learning_dir, 'groupdata_lload.mat'), 'groupdata');
log = com_log;
save(fullfile(learning_dir, 'log_lload.mat'), 'log');

end

