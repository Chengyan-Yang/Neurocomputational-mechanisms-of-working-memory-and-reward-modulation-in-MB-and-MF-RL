function  [LL,RPE_2,SPE_1] = MB_MF_llik_M3_low(x,data)

% parameters
b_small = x(1);           % softmax inverse temperature small reward
b_large = x(2);          % softmax inverse temperature rate large reward
lr = x(3);              % learning rate
lambda = x(4);          % eligibility trace decay
w = x(5);               % mixing weight small reward
st = x(6);              % stickiness
respst = x(7);          % stickiness

Qmf = ones(2,2)*4.5;
Q2 = ones(2,1)*4.5;     % Q(s,a): state-action value function for Q-learning

Tm = cell(2,1);
Tm{1} = [1 0; 0 1];     % transition matrix door1 - bear1(rews_1), door2 - bear2(rews_2)
Tm{2} = [1 0; 0 1];     % transition matrix door3 - bear1(rews_1), door4 - bear2(rews_2)
M = [0 0; 0 0];         % last choice structure
R = [0; 0];             % last choice structure
LL = 0;

RPE_2 = zeros(data.N,1); % initialize vector of RPE
SPE_1 = zeros(data.N,1); % initialize vector of SPE

% loop through trials
for t = 1:data.N
    
    % set choice structure
    if (data.stimuli(t,1) == 2) || (data.stimuli(t,1) == 4)
        R = flipud(R);
    end
    
    % get states and action
    s1 = data.s(t,1);
    s2 = data.s(t,2);
    a = data.choice(t);
    action = a;
    a = a - (s1 == 2)*(2);
    
    % compute model based Q value which is based on Q-values and transition structure
    Qmb = Tm{s1}'*Q2;                                           
    
    % depending on magnitude, either use b_small or b_large
    if strcmp(data.magnitude{t}, 'small')
        b = b_small;
    else
        b = b_large;
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
    if action == data.stimuli(t,1)
        R(1) = 1;                                               
    else
        R(2) = 1;
    end
    
    % backup with actual choice (i.e. sarsa)
    dtQ(1) = Q2(s2) - Qmf(s1,a); 
    SPE_1(t) = dtQ(1);
    
    % update model-free Q values with learning rate 
    Qmf(s1,a) = Qmf(s1,a) + lr*dtQ(1);                          
    
    % calculate prediction error (bear reward: 2nd choice)
    dtQ(2) = data.points(t) - Q2(s2);                           
    RPE_2(t) = dtQ(2);

    % update Q values with learning rate
    Q2(s2) = Q2(s2) + lr*dtQ(2); 
    
    % update model-free Q values with eligibility trace and learning rate
    Qmf(s1,a) = Qmf(s1,a) + lambda*lr*dtQ(2);                   
    
end