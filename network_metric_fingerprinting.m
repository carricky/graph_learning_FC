%% path config
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018fall/gsp/smooth_graph')
addpath('/Users/siyuangao/Working_Space/fmri/bctnet/BCT/2017_01_15_BCT')
addpath('/Users/siyuangao/Working_Space/fmri/data/HCP515')

load('all_mats_b0.1.mat')
load('all_mats_dong.mat')
load('all_mats_cc_GSR.mat')

%% parse
n_task = size(all_mats_LR_smooth, 3);
n_sub = size(all_mats_LR_smooth, 2);
n_edge = size(all_mats_LR_smooth, 1);
n_node = 259;

% convert smooth graph
all_mats_LR_smooth_2d = zeros(n_node, n_node, n_sub, n_task);
all_mats_RL_smooth_2d = zeros(n_node, n_node, n_sub, n_task);

for i_task = 1 : n_task
    for j_sub = 1 : n_sub
        all_mats_LR_smooth_2d(:, :, j_sub, i_task) = squareform(all_mats_LR_smooth(:, j_sub, i_task));
        all_mats_RL_smooth_2d(:, :, j_sub, i_task) = squareform(all_mats_RL_smooth(:, j_sub, i_task));
    end
end

% convert dong graph
all_mats_LR_dong_2d = zeros(n_node, n_node, n_sub, n_task);
all_mats_RL_dong_2d = zeros(n_node, n_node, n_sub, n_task);

for i_task = 1 : n_task
    for j_sub = 1 : n_sub
        all_mats_LR_dong_2d(:, :, j_sub, i_task) = squareform(all_mats_LR_dong(:, j_sub, i_task));
        all_mats_RL_dong_2d(:, :, j_sub, i_task) = squareform(all_mats_RL_dong(:, j_sub, i_task));
    end
end

% convert cc grpah
% sparsify the graph based on the sparsity in the smooth graph
% take the absolute value as the edge weight so node degree is not weird
for i_task = 1 : n_task
    for j_sub = 1 : n_sub
        nonzero_lr = sum(all_mats_LR_smooth(:, j_sub, i_task)~=0);
        nonzero_rl = sum(all_mats_RL_smooth(:, j_sub, i_task)~=0);
        weight_sorted = sort(abs(all_mats_LR_cc(:, j_sub, i_task)), 'descend');
        thresh_lr = weight_sorted(nonzero_lr);
        weight_sorted = sort(abs(all_mats_RL_cc(:, j_sub, i_task)), 'descend');
        thresh_rl = weight_sorted(nonzero_rl);
        
        temp = all_mats_LR_cc(:, j_sub, i_task);
        temp(abs(temp)<thresh_lr) = 0;
        all_mats_LR_cc(:, j_sub, i_task) = abs(temp);
        
        temp = all_mats_RL_cc(:, j_sub, i_task);
        temp(abs(temp)<thresh_rl) = 0;
        all_mats_RL_cc(:, j_sub, i_task) = abs(temp);
    end
end

all_mats_LR_cc_2d = zeros(n_node, n_node, n_sub, n_task);
all_mats_RL_cc_2d = zeros(n_node, n_node, n_sub, n_task);

for i_task = 1 : n_task
    for j_sub = 1 : n_sub
        all_mats_LR_cc_2d(:, :, j_sub, i_task) = squareform(all_mats_LR_cc(:, j_sub, i_task));
        all_mats_RL_cc_2d(:, :, j_sub, i_task) = squareform(all_mats_RL_cc(:, j_sub, i_task));
    end
end

%% calculate node degree in smooth graph
% vector metrics
all_degree_smooth = zeros(n_sub, n_task, n_node, 2);
all_cc_vec_smooth = zeros(n_sub, n_task, n_node, 2);

for i_task = 1 : n_task
    fprintf('%dth task\n', i_task);
    for j_sub = 1 : n_sub
        % clustering coefficient
        all_cc_vec_smooth(j_sub, i_task, :, 1) = clustering_coef_wu(...
            all_mats_LR_smooth_2d(:, :, j_sub, i_task));
        all_cc_vec_smooth(j_sub, i_task, :, 2) = clustering_coef_wu(...
            all_mats_RL_smooth_2d(:, :, j_sub, i_task));      
        
        % degree
        all_degree_smooth(j_sub, i_task, :, 1) = degrees_und(all_mats_LR_smooth_2d(:, :, j_sub, i_task));
        all_degree_smooth(j_sub, i_task, :, 2) = degrees_und(all_mats_RL_smooth_2d(:, :, j_sub, i_task));

    end
end

%% calculate node degree in dong graph
% vector metrics
all_degree_dong = zeros(n_sub, n_task, n_node, 2);
all_cc_vec_dong = zeros(n_sub, n_task, n_node, 2);

for i_task = 1 : n_task
    fprintf('%dth task\n', i_task);
    for j_sub = 1 : n_sub
        % clustering coefficient
        all_cc_vec_dong(j_sub, i_task, :, 1) = clustering_coef_wu(...
            all_mats_LR_dong_2d(:, :, j_sub, i_task));
        all_cc_vec_dong(j_sub, i_task, :, 2) = clustering_coef_wu(...
            all_mats_RL_dong_2d(:, :, j_sub, i_task));
        
        % degree
        all_degree_dong(j_sub, i_task, :, 1) = degrees_und(all_mats_LR_dong_2d(:, :, j_sub, i_task));
        all_degree_dong(j_sub, i_task, :, 2) = degrees_und(all_mats_RL_dong_2d(:, :, j_sub, i_task));

    end
end

%% calculate node degree in cc graph
% vector metrics
all_degree_cc = zeros(n_sub, n_task, n_node, 2);
all_cc_vec_cc = zeros(n_sub, n_task, n_node, 2);

for i_task = 1 : n_task
    fprintf('%dth task\n', i_task);
    for j_sub = 1 : n_sub
        % clustering coefficient
        all_cc_vec_cc(j_sub, i_task, :, 1) = clustering_coef_wu(...
            all_mats_LR_cc_2d(:, :, j_sub, i_task));
        all_cc_vec_cc(j_sub, i_task, :, 2) = clustering_coef_wu(...
            all_mats_RL_cc_2d(:, :, j_sub, i_task));
             
        % degree
        all_degree_cc(j_sub, i_task, :, 1) = degrees_und(all_mats_LR_cc_2d(:, :, j_sub, i_task));
        all_degree_cc(j_sub, i_task, :, 2) = degrees_und(all_mats_RL_cc_2d(:, :, j_sub, i_task));

    end
end


%% fingerprinting with these vectors
accuracy_degree_smooth = zeros(n_task, 1);
accuracy_cc_smooth = zeros(n_task, 1);

accuracy_degree_dong = zeros(n_task, 1);
accuracy_cc_dong= zeros(n_task, 1);


accuracy_degree_cc = zeros(n_task, 1);
accuracy_cc_cc = zeros(n_task, 1);


% smooth graph
for i_task = 1 : n_task   
    temp_corr = corr(squeeze(all_degree_smooth(:, i_task, :, 1))',...
        squeeze(all_degree_smooth(:, i_task, :, 2))');
    [~, I] = sort(temp_corr, 2, 'descend');
    accuracy_degree_smooth(i_task) = sum(I(:, 1) == (1:n_sub)')/n_sub;
    
    temp_corr = corr(squeeze(all_cc_vec_smooth(:, i_task, :, 1))',...
        squeeze(all_cc_vec_smooth(:, i_task, :, 2))');
    [~, I] = sort(temp_corr, 2, 'descend');
    accuracy_cc_smooth(i_task) = sum(I(:, 1) == (1:n_sub)')/n_sub;
end

% dong graph
for i_task = 1 : n_task
    
    temp_corr = corr(squeeze(all_degree_dong(:, i_task, :, 1))',...
        squeeze(all_degree_dong(:, i_task, :, 2))');
    [~, I] = sort(temp_corr, 2, 'descend');
    accuracy_degree_dong(i_task) = sum(I(:, 1) == (1:n_sub)')/n_sub;
    
    temp_corr = corr(squeeze(all_cc_vec_dong(:, i_task, :, 1))',...
        squeeze(all_cc_vec_dong(:, i_task, :, 2))');
    [~, I] = sort(temp_corr, 2, 'descend');
    accuracy_cc_dong(i_task) = sum(I(:, 1) == (1:n_sub)')/n_sub;
end

% cc graph
for i_task = 1 : n_task   
    temp_corr = corr(squeeze(all_degree_cc(:, i_task, :, 1))',...
        squeeze(all_degree_cc(:, i_task, :, 2))');
    [~, I] = sort(temp_corr, 2, 'descend');
    accuracy_degree_cc(i_task) = sum(I(:, 1) == (1:n_sub)')/n_sub;
    
    temp_corr = corr(squeeze(all_cc_vec_cc(:, i_task, :, 1))',...
        squeeze(all_cc_vec_cc(:, i_task, :, 2))');
    [~, I] = sort(temp_corr, 2, 'descend');
    accuracy_cc_cc(i_task) = sum(I(:, 1) == (1:n_sub)')/n_sub;
end

%% plot of fingerprinting accuracy
figure;
bar([accuracy_degree_smooth, accuracy_degree_dong, accuracy_degree_cc])
legend({'log graph', 'l2 graph', 'cc graph'})
xticklabels(task_name)

figure;
bar([accuracy_cc_smooth, accuracy_cc_dong, accuracy_cc_cc])
legend({'log graph', 'l2 graph', 'cc graph'})
xticklabels(task_name)

%% guess which task each subject is doing
n_fold = 10;
indices = crossvalind('Kfold', n_sub, n_fold);

% smooth graph
predicted = [];
label = [];
accuracy_smooth = zeros(n_fold, 1);

for i_fold = 1 : n_fold
    tic
    fprintf('%dth fold\n', i_fold);
    
    test_idx = (indices==i_fold);
    train_idx = (indices~=i_fold);
    train_mats = reshape(all_degree_smooth(train_idx, :, :, 1), [], n_node);
    train_behav = reshape(repmat([1:n_task], sum(train_idx), 1), [], 1);
    train_behav(train_behav==3)=2;
    test_mats = reshape(all_degree_smooth(test_idx, :, :, 1), [], n_node);
    test_behav = reshape(repmat([1:n_task], sum(test_idx), 1), [], 1);
    test_behav(test_behav==3)=2;
    
    Mdl = fitcecoc(train_mats, train_behav);
    y = predict(Mdl, test_mats);
    accuracy_smooth(i_fold) = sum(y==test_behav)/numel(y);
    
    predicted = [predicted;y];
    label = [label;test_behav];
    toc
end

% dong graph
predicted = [];
label = [];
accuracy_dong = zeros(n_fold, 1);

for i_fold = 1 : n_fold
    tic
    fprintf('%dth fold\n', i_fold);
    
    test_idx = (indices==i_fold);
    train_idx = (indices~=i_fold);
    train_mats = reshape(all_degree_dong(train_idx, :, :, 1), [], n_node);
    train_behav = reshape(repmat([1:n_task], sum(train_idx), 1), [], 1);
    train_behav(train_behav==3)=2;
    test_mats = reshape(all_degree_dong(test_idx, :, :, 1), [], n_node);
    test_behav = reshape(repmat([1:n_task], sum(test_idx), 1), [], 1);
    test_behav(test_behav==3)=2;
    
    Mdl = fitcecoc(train_mats, train_behav);
    y = predict(Mdl, test_mats);
    accuracy_dong(i_fold) = sum(y==test_behav)/numel(y);
    
    predicted = [predicted;y];
    label = [label;test_behav];
    toc
end

% cc graph
predicted = [];
label = [];
accuracy_cc = zeros(n_fold, 1);

for i_fold = 1 : n_fold
    tic
    fprintf('%dth fold\n', i_fold);
    
    test_idx = (indices==i_fold);
    train_idx = (indices~=i_fold);
    train_mats = reshape(all_degree_cc(train_idx, :, :, 1), [], n_node);
    train_behav = reshape(repmat([1:n_task], sum(train_idx), 1), [], 1);
    train_behav(train_behav==3)=2;
    test_mats = reshape(all_degree_cc(test_idx, :, :, 1), [], n_node);
    test_behav = reshape(repmat([1:n_task], sum(test_idx), 1), [], 1);
    test_behav(test_behav==3)=2;
    
    Mdl = fitcecoc(train_mats, train_behav);
    y = predict(Mdl, test_mats);
    accuracy_cc(i_fold) = sum(y==test_behav)/numel(y);
    
    predicted = [predicted;y];
    label = [label;test_behav];
    toc
end

%% plot for task decoding accuracy
figure;
% boxchart([accuracy_smooth, accuracy_dong, accuracy_cc])
barplot([accuracy_smooth, accuracy_dong, accuracy_cc])
xticklabels({'log graph', 'l2 graph', 'correlation graph'})
ylabel('task decoding accuracy')
ylim([0.5, 1])
% disp(sum(predicted==label)/numel(predicted))

%% plot
figure;
bar([accuracy_mat_s, accuracy_degree_smooth, accuracy_cc_smooth,...
    accuracy_mat_cc, accuracy_degree_cc, accuracy_cc_cc])
% legend({'mat s', 'd s', 'cc s', 'mat cc', 'd cc', 'cc cc'})
legend({'mat l2', 'd l2', 'cc l2', 'mat log', 'd log', 'cc log'})
xticklabels(task_name)

%% save out the node degrees for visualize
missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
idx = ones(268, 1);
idx(missing_nodes) = 0;
idx = logical(idx);
% node_degree_s = squeeze(mean(mean(all_degree_s, 4), 1));
node_degree_s = squeeze(all_degree_smooth(4, :, :, 1));
temp_mats_s = zeros(268, 268, n_task);
for i = 1 : n_task
    temp = zeros(268, 1);
    temp(idx) = node_degree_s(i, :);
    temp = diag(temp);
    temp_mats_s(:, :, i) = temp;
    dlmwrite([task_name{i},'_degree.txt'], temp)
end