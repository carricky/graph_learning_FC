%% config
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/CPM_siyuan/utils')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/all_behav.mat')

%% find the sparse edges
n_sub = size(all_mats_LR_smooth, 2);
n_task = size(all_mats_LR_smooth, 3);
n_edge = size(all_mats_LR_smooth, 1);

all_mats_smooth_temp = all_mats_LR_smooth+all_mats_RL_smooth;
edge_density = squeeze(sum(all_mats_smooth_temp~=0, 2)/n_sub);

edge_idx = zeros(n_edge, n_task);
for i_task = 1 : n_task
    edge_idx(edge_density(:, i_task)>0.5, i_task) = 1;
end
%% prediction
for i_task = 1 : n_task
    [q_s, r_pearson, r_rank, y, mask_pos, mask_neg] = ridgeCPM_edge(...
        all_mats_smooth_temp(logical(edge_idx(:, i_task)), :, i_task), all_behav, 1);
    disp(sqrt(q_s))
end