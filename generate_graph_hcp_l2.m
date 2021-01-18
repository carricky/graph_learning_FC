addpath('/home/siyuan/code/gspbox')
addpath('/home/siyuan/code/unlocbox')
addpath('/mnt/store4/mri_group/siyuan_data/HCP515/signals/GSR')

gsp_start;
init_unlocbox;

% HCP
% task_name = {'GAMBLING', 'REST', 'REST2', 'LANGUAGE', 'MOTOR', ...
%     'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION'};
task_name = {'GAM', 'REST', 'REST2', 'LAN', 'MOT', 'REL', 'SOC', 'WM', 'EMO'};
missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];

%% smooth graph
% a_list = [10, 3, 1.5, 0.5, 0.3, 0.1, 0.07, 0.05, 0.03, 0.01];
a_list = [0.5];
n_sub = 515;
all_mats_LR_dong = zeros(33411, n_sub, 9, numel(a_list));
all_mats_RL_dong = zeros(33411, n_sub, 9, numel(a_list));

for idx_a = 1 : numel(a_list)
    s = 1;
    for i_task = 1:9
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
            W = gsp_learn_graph_l2_degrees(s*Z, s*a_list(idx_a), params); % this gives the learned graph weight
            W(W<1e-5)=0; % clean the weight
            W(1:260:end) = 0;
            all_mats_LR_dong(:, j_sub, i_task, idx_a) = squareform(W);
        end
        
        % RL
        load(['signal_RL_', task_name{i_task}, '_GSR.mat'])
        signal_RL(missing_nodes, :, :) = [];
        for j_sub = 1 : n_sub
            X = signal_RL(:, :, j_sub);
            n = size(X,1);
            %        s = sqrt(2*(n-1))/2 / 3;
            params.maxit = 50000;
            params.step_size = 0.1;
            params.verbosity = 1;
            params.tol = 1e-5;
            
            Z = gsp_distanz(X').^2;  % calculates the pairwise Euclidean distance between points
            Z = Z./max(Z(:));
            W = gsp_learn_graph_l2_degrees(s*Z, s*a_list(idx_a), params); % this gives the learned graph weight
            W(W<1e-5)=0; % clean the weight
            W(1:260:end) = 0;
            all_mats_RL_dong(:, j_sub, i_task, idx_a) = squareform(W);
        end
    end
end
save('all_mats_l2_a0_5.mat', 'all_mats_LR_dong', 'all_mats_RL_dong', 'a_list')