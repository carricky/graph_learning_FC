%% load data
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018fall/gsp/smooth_graph')
load('all_mats_b0.1.mat')
load('all_mats_dong.mat')
load('all_mats_cc_GSR.mat')

task_name = {'GAM', 'REST', 'REST2', 'LAN', 'MOT', 'REL', 'SOC', 'WM', 'EMO'};

%% parse out dimension
n_task = size(all_mats_LR_smooth, 3);
n_sub_total = size(all_mats_LR_smooth, 2);

%% set variables
accuracy_smooth = zeros(n_task, n_task);
accuracy_dong = zeros(n_task, n_task);
accuracy_cc = zeros(n_task, n_task);
accuracy_gd = zeros(n_task, n_task);

%% fingerprinting
rng(665);

for i_task = 1 : 9
    disp(i_task)
    for j_task = 1:9
        if i_task ~= j_task
            % smooth graph
            [~, I] = pdist2(all_mats_LR_smooth(:, :, i_task)',...
                all_mats_LR_smooth(:, :, j_task)', 'correlation', 'Smallest', 1);
            accuracy_smooth(i_task, j_task) = sum(I == (1:n_sub_total))/n_sub_total;
            
            % pearson correlation
            [~, I] = pdist2(all_mats_LR_cc(:, :, i_task)',...
                all_mats_LR_cc(:, :, j_task)', 'correlation', 'Smallest', 1);
            accuracy_cc(i_task, j_task) = sum(I == (1:n_sub_total))/n_sub_total;
            
            % dong method
            [~, I] = pdist2(all_mats_LR_dong(:, :, i_task)',...
                all_mats_LR_dong(:, :, j_task)', 'correlation', 'Smallest', 1);
            accuracy_dong(i_task, j_task) = sum(I == (1:n_sub_total))/n_sub_total;
        end
    end
end

%% plot
figure;
subplot(1, 3, 1)
heatmap(task_name, task_name, accuracy_smooth)
title('log penalized graph accuracy')
% caxis([0, 1])

subplot(1, 3, 2)
heatmap(task_name, task_name, accuracy_smooth-accuracy_dong)
title('log graph accuracy-l2 graph accuracy')
% caxis([0, 1])

subplot(1, 3, 3)
heatmap(task_name, task_name, accuracy_smooth-accuracy_cc)
title('log graph accuracy-correlation graph accuracy')
% caxis([0, 1])
