%% load data
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018fall/gsp/smooth_graph')
% load('all_mats_cc_GSR.mat')
load('all_mats_cc_short.mat')

% fisher transform
% all_mats_LR_cc = atanh(all_mats_LR_cc);
% all_mats_RL_cc = atanh(all_mats_RL_cc);

%% parse out dimension
n_task = size(all_mats_LR_cc, 3);
n_sub_total = size(all_mats_LR_cc, 2);
rank_deficient_task = [1, 5, 6, 7, 9];

%% set variables
dis_gd = zeros(n_sub_total, n_sub_total, n_task);
lambda = 2;

%% generate geodesic distances
for i_task = 1 : 9
    disp(i_task)
    if any(rank_deficient_task==i_task)
        reg = 1;
    else
        reg = 0;
    end
    % geodesic distance
    for j_sub = 1 : n_sub_total
        disp(j_sub)
        X = all_mats_LR_cc(:, j_sub, i_task);
        X = squareform(X);
        if reg == 1
            X(1:260:end) = lambda;
        end
        
        [u, s, ~] = svd(X);
        % lift very small eigen values
        for k_node = 1 : 259
            if s(k_node, k_node) < 1e-3
                s(k_node, k_node) = 1e-3;
            end
        end
        X_mod = u * s^-0.5 * u';
        
        for k_sub = 1 : n_sub_total
            Y = all_mats_RL_cc(:, k_sub, i_task);
            Y = squareform(Y);
            if reg == 1
                Y(1:260:end) = lambda;
            end
            M = X_mod * Y * X_mod;
            [~, s, ~] = svd(M);
            dis_gd(j_sub, k_sub, i_task) = sqrt(sum(log(diag(s)).^2));
        end
    end
end

save('dis_gd_short.mat', 'dis_gd')

