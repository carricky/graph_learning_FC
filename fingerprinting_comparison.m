%% load data
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018fall/gsp/smooth_graph')
load('all_mats_b0.1.mat')
load('all_mats_dong.mat')
load('all_mats_cc_GSR.mat')
load('dis_gd_short.mat')

task_name = {'GAM', 'REST', 'REST2', 'LAN', 'MOT', 'REL', 'SOC', 'WM', 'EMO'};
%% parse out dimension
n_task = size(all_mats_LR_smooth, 3);
n_sub_total = size(all_mats_LR_smooth, 2);
n_rep = 20;

%% set variables
accuracy_smooth = zeros(n_task, n_rep);
accuracy_dong = zeros(n_task, n_rep);
accuracy_cc = zeros(n_task, n_rep);
accuracy_gd = zeros(n_task, n_rep);

%% fingerprinting
rng(665); 
n_sub_select = 100;
for rep = 1 : n_rep
    disp(rep)
%      sub_idx = datasample(1:n_sub_total, n_sub_select); % w replacement
    sub_idx = randperm(n_sub_total, n_sub_select); % w/o replacement
       
    for i_task = 1 : 9
        % smooth graph
        [~, I] = pdist2(all_mats_LR_smooth(:, sub_idx, i_task)',...
            all_mats_RL_smooth(:, sub_idx, i_task)', 'correlation', 'Smallest', 1);
        accuracy_smooth(i_task, rep) = sum(I == (1:n_sub_select))/n_sub_select;
        
        % pearson correlation
        [~, I] = pdist2(all_mats_LR_cc(:, sub_idx, i_task)',...
            all_mats_RL_cc(:, sub_idx, i_task)', 'correlation', 'Smallest', 1);
        accuracy_cc(i_task, rep) = sum(I == (1:n_sub_select))/n_sub_select;
            
        % dong method
        [~, I] = pdist2(all_mats_LR_dong(:, sub_idx, i_task)',...
            all_mats_RL_dong(:, sub_idx, i_task)', 'correlation', 'Smallest', 1);
        accuracy_dong(i_task, rep) = sum(I == (1:n_sub_select))/n_sub_select;
        
        % geodesic distance
        [~, I] = sort(dis_gd(sub_idx, sub_idx, i_task), 2, 'ascend');
        accuracy_gd(i_task, rep) = sum(I(:, 1) == (1:n_sub_select)')/n_sub_select;
    end
end

%% plot
figure;
accuracy = [mean(accuracy_smooth(:, 1:20), 2), mean(accuracy_dong(:, 1:20), 2),...
    mean(accuracy_cc(:, 1:20), 2), mean(accuracy_gd(:, 1:20), 2)];

se = [std(accuracy_smooth(:, 1:20), 0, 2), std(accuracy_dong(:, 1:20), 0, 2),...
    std(accuracy_cc(:, 1:20), 0, 2), std(accuracy_gd(:, 1:20), 0, 2)];
se = se./sqrt(n_rep);

b = bar(accuracy, 'grouped');
legend('log penalty graph', 'l2 penalty graph', 'correlation graph', 'geodesic distance')
xticklabels(task_name)
ylabel('fingerprinting accuracy')

hold on
% Calculate the number of bars in each group
nbars = size(accuracy, 2);
% Get the x coordinate of the bars
x = [];
for i = 1:nbars
    x = [x ; b(i).XEndPoints];
end
% Plot the errorbars
errorbar(x', accuracy, se,'k','linestyle','none', 'LineWidth', 1, 'HandleVisibility','off', 'CapSize', 6);
hold off
