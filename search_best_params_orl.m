%==========================================================================
% GDMFC Parameter Search on ORL Face Dataset
%==========================================================================
% 此脚本通过网格搜索找到GDMFC在ORL数据集上的最佳参数组合
% This script finds the best parameter combination for GDMFC on ORL dataset
% through grid search
%
% 目标性能 (Target Performance):
%   ACC: 87.80±1.08
%   NMI: 93.81±0.39
%   Purity: 89.70±0.54
%==========================================================================

clear; clc; close all;

%% Add paths 添加路径
addpath(genpath('../DMF_MVC'));

fprintf('========================================\n');
fprintf('GDMFC Parameter Search on ORL Dataset\n');
fprintf('========================================\n\n');

%% ==================== Load ORL Dataset 加载ORL数据集 ====================
fprintf('Step 1: Loading ORL face dataset...\n');

dataPath = '../../dataset/orl';
numSubjects = 40;
imagesPerSubject = 10;
imageHeight = 112;
imageWidth = 92;
numSamples = numSubjects * imagesPerSubject;

% Load all images
allImages = zeros(numSamples, imageHeight * imageWidth);
labels = zeros(numSamples, 1);

sampleIdx = 1;
for subjID = 1:numSubjects
    subjFolder = fullfile(dataPath, sprintf('s%d', subjID));
    for imgID = 1:imagesPerSubject
        imgPath = fullfile(subjFolder, sprintf('%d.pgm', imgID));
        img = imread(imgPath);
        allImages(sampleIdx, :) = double(img(:)');
        labels(sampleIdx) = subjID;
        sampleIdx = sampleIdx + 1;
    end
end

fprintf('  Loaded %d images\n\n', numSamples);

%% ==================== Construct Multi-View Features 构造多视图特征 ====================
fprintf('Step 2: Constructing multi-view features...\n');

% View 1: Downsampled pixels
downsampleFactor = 2;
newHeight = imageHeight / downsampleFactor;
newWidth = imageWidth / downsampleFactor;
X1 = zeros(numSamples, newHeight * newWidth);

for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    imgDown = imresize(img, [newHeight, newWidth], 'bilinear');
    X1(i, :) = imgDown(:)';
end

% View 2: Block statistics
blockSize = 8;
numBlocksH = floor(imageHeight / blockSize);
numBlocksW = floor(imageWidth / blockSize);
lbpFeatDim = numBlocksH * numBlocksW * 4;
X2 = zeros(numSamples, lbpFeatDim);

for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    featIdx = 1;
    for bh = 1:numBlocksH
        for bw = 1:numBlocksW
            rowStart = (bh-1) * blockSize + 1;
            rowEnd = min(bh * blockSize, imageHeight);
            colStart = (bw-1) * blockSize + 1;
            colEnd = min(bw * blockSize, imageWidth);
            block = img(rowStart:rowEnd, colStart:colEnd);
            X2(i, featIdx) = mean(block(:));
            X2(i, featIdx+1) = std(block(:));
            X2(i, featIdx+2) = min(block(:));
            X2(i, featIdx+3) = max(block(:));
            featIdx = featIdx + 4;
        end
    end
end

X = cell(1, 2);
X{1} = X1;
X{2} = X2;
y = labels;

% Normalize
for v = 1:2
    X{v} = NormalizeFea(X{v}, 0);
end

numCluster = numSubjects;
fprintf('  Features constructed and normalized\n\n');

%% ==================== Parameter Search Space 参数搜索空间 ====================
fprintf('Step 3: Defining parameter search space...\n');

% 搜索空间定义
% Layer structures to try 尝试的层结构
layer_configs = {
    [500, 200, 100],    % 3层
    [400, 150],         % 2层
    [600, 300, 150],    % 3层（更宽）
    [300, 100],         % 2层（较窄）
    [500, 250, 125]     % 3层（递减）
};

% Lambda1 (HSIC diversity) 多样性系数
lambda1_range = [0.00001, 0.0001, 0.001, 0.01];

% Lambda2 (orthogonal constraint) 正交约束系数
lambda2_range = [0.00001, 0.0001, 0.001, 0.01];

% Beta (graph regularization) 图正则化系数
beta_range = [0.0001, 0.001, 0.01, 0.1];

% Gamma (view weight parameter) 视图权重参数
gamma_range = [1.2, 1.5, 1.8, 2.0];

% Graph k (number of neighbors) 邻居数
k_range = [3, 5, 7, 10];

fprintf('  Total configurations to search:\n');
fprintf('    Layer configs: %d\n', length(layer_configs));
fprintf('    Lambda1 values: %d\n', length(lambda1_range));
fprintf('    Lambda2 values: %d\n', length(lambda2_range));
fprintf('    Beta values: %d\n', length(beta_range));
fprintf('    Gamma values: %d\n', length(gamma_range));
fprintf('    K values: %d\n', length(k_range));

% 为了减少计算量,我们使用随机搜索策略
% 从每个参数空间随机采样,总共测试n_trials次
n_trials = 50;  % 可以根据时间调整
fprintf('  Using random search with %d trials\n\n', n_trials);

%% ==================== Random Parameter Search 随机参数搜索 ====================
fprintf('Step 4: Starting parameter search...\n');
fprintf('========================================\n');

% 初始化结果记录
results = struct();
results.configs = cell(n_trials, 1);
results.ACC = zeros(n_trials, 1);
results.NMI = zeros(n_trials, 1);
results.Purity = zeros(n_trials, 1);
results.time = zeros(n_trials, 1);
results.alpha = cell(n_trials, 1);

% 目标性能
target_ACC = 87.80;
target_NMI = 93.81;
target_Purity = 89.70;

% 最佳结果跟踪
best_ACC = 0;
best_NMI = 0;
best_Purity = 0;
best_config = [];
best_trial = 0;

% 随机搜索
rng(42);  % 设置随机种子以便复现

for trial = 1:n_trials
    fprintf('\n--- Trial %d/%d ---\n', trial, n_trials);
    
    % 随机选择参数
    layer_idx = randi(length(layer_configs));
    layers = layer_configs{layer_idx};
    
    lambda1 = lambda1_range(randi(length(lambda1_range)));
    lambda2 = lambda2_range(randi(length(lambda2_range)));
    beta = beta_range(randi(length(beta_range)));
    gamma = gamma_range(randi(length(gamma_range)));
    graph_k = k_range(randi(length(k_range)));
    
    % 打印当前配置
    fprintf('  Layers: [');
    for i = 1:length(layers)
        fprintf('%d', layers(i));
        if i < length(layers), fprintf(', '); end
    end
    fprintf(', %d]\n', numCluster);
    fprintf('  λ1=%.5f, λ2=%.5f, β=%.4f, γ=%.2f, k=%d\n', ...
            lambda1, lambda2, beta, gamma, graph_k);
    
    % 设置参数
    options = struct();
    options.lambda1 = lambda1;
    options.lambda2 = lambda2;
    options.beta = beta;
    options.gamma = gamma;
    options.graph_k = graph_k;
    options.maxIter = 100;
    options.tol = 1e-5;
    
    % 运行GDMFC
    try
        tic;
        [H, Z, alpha, obj_values] = GDMFC(X, numCluster, layers, options);
        elapsed = toc;
        
        % 聚类
        H_final = H;
        S = H_final * H_final';
        S = (S + S') / 2;
        S = max(S, 0);
        predict_labels = SpectralClustering(S, numCluster);
        
        % 评估
        res = bestMap(y, predict_labels);
        ACC = length(find(y == res)) / length(y) * 100;  % 转换为百分比
        NMI = MutualInfo(y, predict_labels) * 100;       % 转换为百分比
        Purity = compute_purity(y, predict_labels) * 100;
        
        % 保存结果
        results.configs{trial} = struct('layers', layers, 'lambda1', lambda1, ...
                                        'lambda2', lambda2, 'beta', beta, ...
                                        'gamma', gamma, 'k', graph_k);
        results.ACC(trial) = ACC;
        results.NMI(trial) = NMI;
        results.Purity(trial) = Purity;
        results.time(trial) = elapsed;
        results.alpha{trial} = alpha;
        
        fprintf('  Results: ACC=%.2f%%, NMI=%.2f%%, Purity=%.2f%% (%.1fs)\n', ...
                ACC, NMI, Purity, elapsed);
        fprintf('  View weights: [%.4f, %.4f]\n', alpha(1), alpha(2));
        
        % 更新最佳结果
        if ACC > best_ACC
            best_ACC = ACC;
            best_NMI = NMI;
            best_Purity = Purity;
            best_config = results.configs{trial};
            best_trial = trial;
            fprintf('  >>> NEW BEST ACC! <<<\n');
        end
        
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        results.ACC(trial) = 0;
        results.NMI(trial) = 0;
        results.Purity(trial) = 0;
    end
end

fprintf('\n========================================\n');
fprintf('Parameter Search Completed!\n');
fprintf('========================================\n\n');

%% ==================== Results Analysis 结果分析 ====================
fprintf('Step 5: Analyzing results...\n\n');

% 找到有效的试验
valid_trials = results.ACC > 0;
n_valid = sum(valid_trials);

fprintf('Valid trials: %d/%d\n\n', n_valid, n_trials);

% 统计信息
fprintf('Performance Statistics:\n');
fprintf('  ACC:    Mean=%.2f%%, Std=%.2f%%, Max=%.2f%%, Min=%.2f%%\n', ...
        mean(results.ACC(valid_trials)), std(results.ACC(valid_trials)), ...
        max(results.ACC(valid_trials)), min(results.ACC(valid_trials)));
fprintf('  NMI:    Mean=%.2f%%, Std=%.2f%%, Max=%.2f%%, Min=%.2f%%\n', ...
        mean(results.NMI(valid_trials)), std(results.NMI(valid_trials)), ...
        max(results.NMI(valid_trials)), min(results.NMI(valid_trials)));
fprintf('  Purity: Mean=%.2f%%, Std=%.2f%%, Max=%.2f%%, Min=%.2f%%\n\n', ...
        mean(results.Purity(valid_trials)), std(results.Purity(valid_trials)), ...
        max(results.Purity(valid_trials)), min(results.Purity(valid_trials)));

% 最佳配置
fprintf('========================================\n');
fprintf('BEST CONFIGURATION (Trial %d):\n', best_trial);
fprintf('========================================\n');
fprintf('  Layers: [');
for i = 1:length(best_config.layers)
    fprintf('%d', best_config.layers(i));
    if i < length(best_config.layers), fprintf(', '); end
end
fprintf(', %d]\n', numCluster);
fprintf('  Lambda1 (HSIC):       %.6f\n', best_config.lambda1);
fprintf('  Lambda2 (Orthogonal): %.6f\n', best_config.lambda2);
fprintf('  Beta (Graph):         %.6f\n', best_config.beta);
fprintf('  Gamma (View weight):  %.2f\n', best_config.gamma);
fprintf('  K (Neighbors):        %d\n', best_config.k);
fprintf('\n');
fprintf('Best Performance:\n');
fprintf('  ACC    = %.2f%% (Target: %.2f%%)\n', best_ACC, target_ACC);
fprintf('  NMI    = %.2f%% (Target: %.2f%%)\n', best_NMI, target_NMI);
fprintf('  Purity = %.2f%% (Target: %.2f%%)\n', best_Purity, target_Purity);
fprintf('  View weights: [%.4f, %.4f]\n', ...
        results.alpha{best_trial}(1), results.alpha{best_trial}(2));
fprintf('========================================\n\n');

% Top 5 configurations
fprintf('Top 5 Configurations by ACC:\n');
fprintf('----------------------------------------\n');
[sorted_ACC, sort_idx] = sort(results.ACC, 'descend');
for i = 1:min(5, n_valid)
    idx = sort_idx(i);
    if results.ACC(idx) == 0, continue; end
    
    cfg = results.configs{idx};
    fprintf('%d. ACC=%.2f%%, NMI=%.2f%%, Purity=%.2f%%\n', i, ...
            results.ACC(idx), results.NMI(idx), results.Purity(idx));
    fprintf('   Layers=[');
    for j = 1:length(cfg.layers)
        fprintf('%d', cfg.layers(j));
        if j < length(cfg.layers), fprintf(','); end
    end
    fprintf(',%d], λ1=%.5f, λ2=%.5f, β=%.4f, γ=%.2f, k=%d\n', ...
            numCluster, cfg.lambda1, cfg.lambda2, cfg.beta, cfg.gamma, cfg.k);
end
fprintf('\n');

%% ==================== Visualization 可视化 ====================
fprintf('Step 6: Generating visualizations...\n');

figure('Name', 'Parameter Search Results', 'Position', [100, 100, 1200, 800]);

% Plot 1: Performance distribution
subplot(2, 3, 1);
scatter(results.ACC(valid_trials), results.NMI(valid_trials), 50, 'filled');
hold on;
plot(best_ACC, best_NMI, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
plot(target_ACC, target_NMI, 'gs', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('ACC (%)', 'FontSize', 11);
ylabel('NMI (%)', 'FontSize', 11);
title('ACC vs NMI', 'FontSize', 12);
legend('Trials', 'Best', 'Target', 'Location', 'best');
grid on;

% Plot 2: ACC vs Purity
subplot(2, 3, 2);
scatter(results.ACC(valid_trials), results.Purity(valid_trials), 50, 'filled');
hold on;
plot(best_ACC, best_Purity, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
plot(target_ACC, target_Purity, 'gs', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('ACC (%)', 'FontSize', 11);
ylabel('Purity (%)', 'FontSize', 11);
title('ACC vs Purity', 'FontSize', 12);
legend('Trials', 'Best', 'Target', 'Location', 'best');
grid on;

% Plot 3: Performance over trials
subplot(2, 3, 3);
plot(1:n_trials, results.ACC, 'b-o', 'MarkerSize', 4);
hold on;
plot(1:n_trials, results.NMI, 'r-s', 'MarkerSize', 4);
plot(1:n_trials, results.Purity, 'g-^', 'MarkerSize', 4);
yline(target_ACC, 'b--', 'LineWidth', 1.5);
yline(target_NMI, 'r--', 'LineWidth', 1.5);
yline(target_Purity, 'g--', 'LineWidth', 1.5);
xlabel('Trial', 'FontSize', 11);
ylabel('Performance (%)', 'FontSize', 11);
title('Performance over Trials', 'FontSize', 12);
legend('ACC', 'NMI', 'Purity', 'Location', 'best');
grid on;

% Plot 4-6: Parameter influence (box plots)
% Extract parameters from valid trials
lambda1_vals = cellfun(@(x) x.lambda1, results.configs(valid_trials));
beta_vals = cellfun(@(x) x.beta, results.configs(valid_trials));
gamma_vals = cellfun(@(x) x.gamma, results.configs(valid_trials));

subplot(2, 3, 4);
boxplot(results.ACC(valid_trials), lambda1_vals);
xlabel('Lambda1', 'FontSize', 11);
ylabel('ACC (%)', 'FontSize', 11);
title('Lambda1 vs ACC', 'FontSize', 12);
grid on;

subplot(2, 3, 5);
boxplot(results.ACC(valid_trials), beta_vals);
xlabel('Beta', 'FontSize', 11);
ylabel('ACC (%)', 'FontSize', 11);
title('Beta vs ACC', 'FontSize', 12);
grid on;

subplot(2, 3, 6);
boxplot(results.ACC(valid_trials), gamma_vals);
xlabel('Gamma', 'FontSize', 11);
ylabel('ACC (%)', 'FontSize', 11);
title('Gamma vs ACC', 'FontSize', 12);
grid on;

fprintf('  Visualization completed.\n\n');

%% ==================== Save Results 保存结果 ====================
fprintf('Step 7: Saving results...\n');

save('GDMFC_parameter_search_results_ORL.mat', 'results', 'best_config', ...
     'best_ACC', 'best_NMI', 'best_Purity', 'target_ACC', 'target_NMI', 'target_Purity');

% 保存最佳配置到文本文件
fid = fopen('best_config_ORL.txt', 'w');
fprintf(fid, 'GDMFC Best Configuration for ORL Dataset\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Layers: [');
for i = 1:length(best_config.layers)
    fprintf(fid, '%d', best_config.layers(i));
    if i < length(best_config.layers), fprintf(fid, ', '); end
end
fprintf(fid, ', %d]\n', numCluster);
fprintf(fid, 'Lambda1: %.6f\n', best_config.lambda1);
fprintf(fid, 'Lambda2: %.6f\n', best_config.lambda2);
fprintf(fid, 'Beta: %.6f\n', best_config.beta);
fprintf(fid, 'Gamma: %.2f\n', best_config.gamma);
fprintf(fid, 'K: %d\n\n', best_config.k);
fprintf(fid, 'Performance:\n');
fprintf(fid, '  ACC    = %.2f%%\n', best_ACC);
fprintf(fid, '  NMI    = %.2f%%\n', best_NMI);
fprintf(fid, '  Purity = %.2f%%\n', best_Purity);
fclose(fid);

fprintf('  Results saved to:\n');
fprintf('    - GDMFC_parameter_search_results_ORL.mat\n');
fprintf('    - best_config_ORL.txt\n\n');

fprintf('========================================\n');
fprintf('Parameter Search Completed Successfully!\n');
fprintf('========================================\n');
