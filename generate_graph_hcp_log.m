addpath('/Users/siyuangao/Working_Space/fmri/gspBox/gspbox')
addpath('/Users/siyuangao/Working_Space/fmri/unlocbox')
addpath('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/GSR')

gsp_start;
init_unlocbox;

% HCP
% task_name = {'GAMBLING', 'REST', 'REST2', 'LANGUAGE', 'MOTOR', ...
%     'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION'};
task_name = {'GAM', 'REST', 'REST2', 'LAN', 'MOT', 'REL', 'SOC', 'WM', 'EMO'};
missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];

%% smooth graph
n_sub = 515;
all_mats_LR_smooth = zeros(33411, n_sub, 9);
all_mats_RL_smooth = zeros(33411, n_sub, 9);
a = 1;
b = 0.1;
s = 1;
for i_task = 1 : 9
    disp(i_task)
    % LR
    load(['signal_LR_', task_name{i_task}, '_GSR.mat'])
    signal_LR(missing_nodes, :, :) = [];
    for j_sub = 1 : n_sub
        X = signal_LR(:, :, j_sub);
        n = size(X,1);
%         s = sqrt(2*(n-1))/2 / 3;
        params.maxit = 50000;
        params.step_size = 0.1;
        params.verbosity = 1;
        params.tol = 1e-5;
        
        Z = gsp_distanz(X').^2;  % calculates the pairwise Euclidean distance between points
        Z = Z./max(Z(:));
        W = gsp_learn_graph_log_degrees(s*Z, s*a, s*b, params); % this gives the learned graph weight
        W(W<1e-5)=0; % clean the weight
        W(1:260:end) = 0;
        all_mats_LR_smooth(:, j_sub, i_task) = squareform(W);
    end
    
    % RL
    load(['signal_RL_', task_name{i_task}, '_GSR.mat'])
    signal_RL(missing_nodes, :, :) = [];
    for j_sub = 1 : n_sub
        X = signal_RL(:, :, j_sub);
        n = size(X,1);
%         s = sqrt(2*(n-1))/2 / 3;
        params.maxit = 50000;
        params.step_size = 0.1;
        params.verbosity = 1;
        params.tol = 1e-5;
        
        Z = gsp_distanz(X').^2;  % calculates the pairwise Euclidean distance between points
        Z = Z./max(Z(:));
        W = gsp_learn_graph_log_degrees(s*Z, s*a, s*b, params); % this gives the learned graph weight
        W(W<1e-5)=0; % clean the weight
        W(1:260:end) = 0;
        all_mats_RL_smooth(:, j_sub, i_task) = squareform(W);
    end
end
save('all_mats_b0.1.mat', 'all_mats_LR_smooth', 'all_mats_RL_smooth')

%% correlation
all_mats_LR_cc = zeros(33411, n_sub, 9);
all_mats_RL_cc = zeros(33411, n_sub, 9);

for i_task = 1 : 9
    disp(i_task)
    % LR
    load(['signal_LR_', task_name{i_task}, '_GSR.mat'])
    signal_LR(missing_nodes, :, :) = [];
    for j_sub = 1 : n_sub
        X = signal_LR(:, :, j_sub);
        W = corr(X');
        W(1:260:end) = 0;
        all_mats_LR_cc(:, j_sub, i_task) = squareform(W);
    end
    
    % RL
    load(['signal_RL_', task_name{i_task}, '_GSR.mat'])
    signal_RL(missing_nodes, :, :) = [];
    for j_sub = 1 : n_sub
        X = signal_RL(:, :, j_sub);
        W = corr(X');
        W(1:260:end) = 0;
        all_mats_RL_cc(:, j_sub, i_task) = squareform(W);
    end
end