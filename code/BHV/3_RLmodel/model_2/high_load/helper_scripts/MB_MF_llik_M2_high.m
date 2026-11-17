function [LL,RPE_2,SPE_1] = MB_MF_llik_M2_high(x,data)

% parameters (Global for high-load)
b = x(1);        % softmax inverse temperature
lr_small = x(2);   % learning rate small reward
lr_large = x(3);  % learning rate large reward
lambda = x(4);   % eligibility trace decay
w = x(5);        % mixing weight
st = x(6);       % choice stickiness
respst = x(7);   % response stickiness

% --- Initialize TWO parts of Q-values ---
% One for each task context: part1 & part2

% part 1
Qmf_part1 = ones(2,2)*4.5;
Q2_part1 = ones(2,1)*4.5;
Tm_part1{1} = [1 0; 0 1];
Tm_part1{2} = [1 0; 0 1];
M_part1 = [0 0; 0 0];
R_part1 = [0; 0];

% Context 2
Qmf_part2 = ones(2,2)*4.5;
Q2_part2 = ones(2,1)*4.5;
Tm_part2{1} = [0 1; 1 0];
Tm_part2{2} = [0 1; 1 0];
M_part2 = [0 0; 0 0];
R_part2 = [0; 0];

LL = 0;
RPE_2 = zeros(data.N,1); % initialize vector of RPE
SPE_1 = zeros(data.N,1); % initialize vector of SPE

% loop through trials
for t = 1:data.N
    
    % --- KEY MODIFICATION RELATIVE TO THR LOW LOAD SCRIPT ---
    % Determine which context (task part) this trial belongs to.
    part = data.part(t); 
    
    % Select the correct part of Q-values for this trial
    if part == 1
        Qmf = Qmf_part1;
        Q2 = Q2_part1;
        Tm = Tm_part1;
        M = M_part1;
        R = R_part1;
    else % part == 2
        Qmf = Qmf_part2;
        Q2 = Q2_part2;
        Tm = Tm_part2;
        M = M_part2;
        R = R_part2;
    end
    
    % part choice structure
    if (data.stimuli(t,1) == 2) || (data.stimuli(t,1) == 4) || (data.stimuli(t,1) == 6) || (data.stimuli(t,1) == 8)
        R = flipud(R);
    end
    
    %  % Remap global states/actions (1-4 or 1-8) to local states/actions (1-2 or 1-4)
    s1 = data.s(t,1); % pairs:1,2,3, or 4
    s2 = data.s(t,2); % animals:1,2,3, or 4
    choice = data.choice(t); % 1,2,3,4,5,6,7, or 8
    % --- A local version of stim_left for the Response-stickiness check ---
    stim_left_local = data.stimuli(t,1); 

    if part == 1
        % Task 1 uses pairs 1,2 and animals 1,2
        s1 = s1;       % pairs 1,2 -> local s1=1,2
        s2 = s2;       % animals 1,2 -> local s2=1,2
        a = choice;    % choice 1,2,3,4
        action = a;
        a = a - (s1 == 2)*(2); % remaps choice [1,2]->a[1,2], [3,4]->a[1,2]
        % stim_left_local is already (1-4) in part 1
    
    else % part == 2
        % Task 2 uses pairs 3,4 and animals 3,4
        s1 = s1 - 2;   % pairs 3,4 -> local s1=1,2
        s2 = s2 - 2;   % animals 3,4 -> local s2=1,2
        a = choice - 4;% choice 5,6,7,8 -> local a=1,2,3,4
        action = a;
        a = a - (s1 == 2)*(2); % remaps choice [1,2]->a[1,2], [3,4]->a[1,2]
        % remap stim_left_local for the R-check in part 2
        stim_left_local = stim_left_local - 4; % (5-8) -> (1-4)
    end
    
    % compute model based Q value
    Qmb = Tm{s1}'*Q2;

     % depending on magnitude, either use w_small or w_large
    if strcmp(data.magnitude{t}, 'small')
        lr = lr_small;
    else
        lr = lr_large;
    end

    % mix TD and model value
    Q = w*Qmb + (1-w)*Qmf(s1,:)' + st.*M(s1,:)' + respst.*R;
    
    % calculate log likelihood
    LL = LL + b*Q(a) - logsumexp(b*Q);
    
    % make the last rocket choice sticky
    M = zeros(2,2);
    M(s1,a) = 1;
    
    % make the last choice sticky
    R = zeros(2,1);
    if action == stim_left_local
        R(1) = 1;
    else
        R(2) = 1;
    end
    
    % backup with actual choice (sarsa)
    dtQ(1) = Q2(s2) - Qmf(s1,a);
    SPE_1(t) = dtQ(1);
    
    % update model-free Q values
    Qmf(s1,a) = Qmf(s1,a) + lr*dtQ(1);
    
    % calculate prediction error (bear reward: 2nd choice)
    dtQ(2) = data.points(t) - Q2(s2);
    RPE_2(t) = dtQ(2);

    % update Q values
    Q2(s2) = Q2(s2) + lr*dtQ(2);
    
    % update model-free Q values with eligibility trace
    Qmf(s1,a) = Qmf(s1,a) + lambda*lr*dtQ(2);
    
    % Write the updated values back to their respective parts for next
    % within-part trial caclulation
    if part == 1
        Qmf_part1 = Qmf;
        Q2_part1 = Q2;
        M_part1 = M;
        R_part1 = R;
    else % ctx == 2
        Qmf_part2 = Qmf;
        Q2_part2 = Q2;
        M_part2 = M;
        R_part2 = R;
    end
    
end
end