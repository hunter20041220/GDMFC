%==========================================================================
% GDMFC Advanced Parameter Optimization for ORL Dataset
% 基于教授建议的精细参数优化脚本
%==========================================================================
% 目标性能 (Target Performance):
%   ACC:    87.80%  (当前最佳: 86.0%)
%   NMI:    93.81%  (当前最佳: 92.91%)
%   Purity: 89.70%  (当前最佳: 87.0%)
%
% 优化策略 (Optimization Strategy):
%   1. 归一化方法 (Normalization): Case 1-5
%   2. Gamma (视图权重参数): [1.5, 2.0, 3.0, 5.0, 10.0]
%   3. Lambda2 (正交约束): [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
%   4. Beta: 固定在115或微调范围 [100, 110, 115, 120, 130]
%   5. Layers: 尝试更宽的结构
%==========================================================================

clear; clc; close all;

fprintf('========================================\n');
fprintf('GDMFC Advanced Parameter Optimization\n');
fprintf('Target: ACC=87.80%%, NMI=93.81%%, Purity=89.70%%\n');
fprintf('Current Best: ACC=86.0%%, NMI=92.91%%, Purity=87.0%%\n');
fprintf('========================================\n\n');

%% ==================== Step 1: Load ORL Dataset ====================
fprintf('Step 1: Loading ORL dataset...\n');
dataPath = '../../dataset/orl';
numSubjects = 40;
imagesPerSubject = 10;
imageHeight = 112;
imageWidth = 92;
numSamples = numSubjects * imagesPerSubject;

% Load from cache if available
cacheFile = 'orl_images_cache.mat';
if exist(cacheFile, 'file')
    load(cacheFile, 'allImages', 'y');
    fprintf('  Loaded cached images (%d samples)\n', numSamples);
else
    fprintf('  Loading images from disk...\n');
    allImages = zeros(numSamples, imageHeight * imageWidth);
    y = zeros(numSamples, 1);
    sampleIdx = 1;
    for subjID = 1:numSubjects
        subjFolder = fullfile(dataPath, sprintf('s%d', subjID));
        for imgID = 1:imagesPerSubject
            imgPath = fullfile(subjFolder, sprintf('%d.pgm', imgID));
            img = imread(imgPath);
            allImages(sampleIdx, :) = double(img(:)');
            y(sampleIdx) = subjID;
            sampleIdx = sampleIdx + 1;
        end
    end
    save(cacheFile, 'allImages', 'y', '-v7.3');
    fprintf('  Saved cache to %s\n', cacheFile);
end

%% ==================== Step 2: Construct Multi-View Features ====================
fprintf('\nStep 2: Constructing multi-view features...\n');

% View 1: Downsampled pixels (低维像素特征)
downsampleFactor = 2;
newH = imageHeight / downsampleFactor;
newW = imageWidth / downsampleFactor;
X1 = zeros(numSamples, newH * newW);
for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    imgDown = imresize(img, [newH, newW], 'bilinear');
    X1(i, :) = imgDown(:)';
end

% View 2: Block statistics (块统计特征)
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

X_raw = cell(1, 2);
X_raw{1} = X1;
X_raw{2} = X2;
numView = 2;
numCluster = numSubjects;

fprintf('  View 1: %d samples × %d features (downsampled pixels)\n', size(X1));
fprintf('  View 2: %d samples × %d features (block statistics)\n\n', size(X2));

%% ==================== Step 3: Define Search Space (基于已知优秀种子) ====================
fprintf('Step 3: Defining parameter search space (seed-based)...\n');

% === 已知优秀结果 (从 beta1-1000.xlsx) ===
% Best: beta=115, ACC=86.00%, NMI=92.79%, Purity=87.00%
% Good seeds: beta ∈ {67, 84, 92, 115, 246, 283, 301, 336}
% 当前配置: Layers=[400,150,40], λ1=1e-5, λ2=1e-3, γ=1.2, k=7, norm=Case3(L2)

fprintf('  Known best result: beta=115, ACC=86.00%% (Norm=L2, γ=1.2)\n');
fprintf('  Target: ACC=87.80%%, NMI=93.81%%, Purity=89.70%%\n');
fprintf('  Gap to close: ACC +1.80%%, NMI +1.02%%, Purity +2.70%%\n\n');

% === 归一化方法 (PRIORITY 1 - 教授强烈建议改用MinMax) ===
norm_modes = [1, 3, 2, 4];  % 重点测试 Case 1 (MinMax), 保留 Case 3 (L2) 作对比
norm_names = {'MinMax-Row', 'L2-Col', 'MinMax-Col', 'Sum-Col'};

% === Gamma 参数 (PRIORITY 2 - 当前1.2太独，需增大到2+) ===
% 教授指出: γ=1.2太接近1，导致模型只选一个最优视图，需要增大到2+
gamma_list = [1.2, 1.5, 2.0, 2.5, 3.0, 5.0];  % 包含1.2做对照，重点>=2.0

% === Lambda2 正交约束 (PRIORITY 3 - 当前1e-3，微调范围) ===
lambda2_list = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2];  % 当前1e-3附近精调

% === Beta (基于已知优秀种子进行微调) ===
% 最佳种子: 67, 84, 92, 115, 246, 283, 301, 336
% 策略: 围绕最佳值115进行±10微调，同时保留其他高性能种子
beta_seeds = [67, 84, 92, 115, 246, 283, 301, 336];  % 已知优秀beta
beta_list = [105, 110, 115, 120, 125];  % 115附近精细搜索

% === Lambda1 (当前固定值，可小范围测试) ===
lambda1_list = [1e-5, 5e-5, 1e-4];  % 当前1e-5，尝试微调

% === Layers (基于当前[400,150,40]，尝试更宽/更深结构) ===
layer_configs = {
    [400, 150, 40],      % 当前最佳配置 (baseline)
    [500, 200, 50],      % 教授建议: 更宽，保留更多特征
    [600, 250, 60],      % 更宽
    [400, 200, 50],      % 中间层加宽
    [500, 150, 40],      % 仅首层加宽
    [450, 180, 50],      % 平滑过渡
    [400, 150, 60]       % 仅末层加宽
};

% === Graph k (当前k=7) ===
k_list = [5, 7, 9, 11];  % k=7附近测试

fprintf('  Normalization modes: %d (重点: Case 1 MinMax vs Case 3 L2)\n', length(norm_modes));
fprintf('  Gamma values: %d (重点: >= 2.0, 当前baseline: 1.2)\n', length(gamma_list));
fprintf('  Lambda1 values: %d\n', length(lambda1_list));
fprintf('  Lambda2 values: %d (当前baseline: 1e-3)\n', length(lambda2_list));
fprintf('  Beta seeds: %d known good values\n', length(beta_seeds));
fprintf('  Beta fine-tune: %d values around 115\n', length(beta_list));
fprintf('  K values: %d (当前baseline: 7)\n', length(k_list));
fprintf('  Layer configs: %d\n', length(layer_configs));
fprintf('\n=== 优化策略 (3-Phase Seed-Based Search) ===\n');
fprintf('Phase 1: Norm × Gamma (最关键组合)\n');
fprintf('  - 测试 MinMax vs L2 归一化\n');
fprintf('  - 重点: γ从1.2提高到2.0+ (让两个视图都起作用)\n');
fprintf('  - 固定其他参数为已知最佳值\n\n');
fprintf('Phase 2: Lambda2 × Beta × K (基于Phase 1最佳)\n');
fprintf('  - 使用优秀beta种子: [67,84,92,115,246,283,301,336]\n');
fprintf('  - Lambda2精调: 影响聚类边界清晰度\n');
fprintf('  - K值优化: 图构造邻居数\n\n');
fprintf('Phase 3: Layer × Lambda1 (最后精调)\n');
fprintf('  - 测试更宽网络结构\n');
fprintf('  - Lambda1微调\n\n');

%% ==================== Step 4: Grid Search with Smart Strategy ====================
fprintf('Step 4: Starting Grid Search (智能策略)...\n');
fprintf('========================================\n\n');

% 结果记录
results = struct();
results.configs = {};
results.ACC = [];
results.NMI = [];
results.Purity = [];
results.time = [];
results.alpha = [];

% 实时保存文件
output_csv = 'best_param_search_results.csv';
output_mat = 'best_param_search_results.mat';

% 创建CSV表头
if ~exist(output_csv, 'file')
    fid = fopen(output_csv, 'w');
    fprintf(fid, 'trial,norm_mode,norm_name,gamma,lambda2,beta,layers,ACC,NMI,Purity,time_s,alpha1,alpha2\n');
    fclose(fid);
end

% 目标性能
target_ACC = 87.80;
target_NMI = 93.81;
target_Purity = 89.70;

% 最佳结果跟踪
best_ACC = 0;
best_NMI = 0;
best_Purity = 0;
best_config = struct();
best_trial = 0;

% === 策略 1: 先测试归一化 × Gamma (固定其他参数为已知最佳) ===
fprintf('=== Phase 1: Testing Normalization × Gamma ===\n');
fprintf('    固定: beta=115, lambda1=1e-5, lambda2=1e-3, k=7, layers=[400,150,40]\n');
fprintf('    目标: 找到最佳归一化+Gamma组合，期望突破86%%\n\n');

trial_count = 0;
phase1_results = [];

% 使用已知最佳参数作为baseline
beta_phase1 = 115;
lambda1_phase1 = 1e-5;
lambda2_phase1 = 1e-3;
k_phase1 = 7;
layers_phase1 = [400, 150, 40];

for norm_idx = 1:length(norm_modes)
    norm_mode = norm_modes(norm_idx);
    norm_name = norm_names{norm_idx};
    
    for gamma_idx = 1:length(gamma_list)
        gamma = gamma_list(gamma_idx);
        trial_count = trial_count + 1;
        
        fprintf('Trial %d: Norm=%s, Gamma=%.2f\n', trial_count, norm_name, gamma);
        
        % 数据预处理
        X = data_guiyi_choos(X_raw, norm_mode);
        for v = 1:numView
            X{v} = NormalizeFea(X{v}, 0);
        end
        
        % 设置参数
        options = struct();
        options.lambda1 = lambda1_phase1;
        options.lambda2 = lambda2_phase1;
        options.beta = beta_phase1;
        options.gamma = gamma;
        options.graph_k = k_phase1;
        options.maxIter = 100;
        options.tol = 1e-5;
        
        % 运行GDMFC
        try
            tic;
            [H, Z, alpha, obj_values] = GDMFC(X, numCluster, layers_phase1, options);
            elapsed = toc;
            
            % 聚类
            S = H * H';
            S = (S + S') / 2;
            S = max(S, 0);
            predict_labels = SpectralClustering(S, numCluster);
            
            % 评估
            res = bestMap(y, predict_labels);
            ACC = length(find(y == res)) / length(y) * 100;
            NMI = MutualInfo(y, predict_labels) * 100;
            Purity = compute_purity(y, predict_labels) * 100;
            
            fprintf('  => ACC=%.2f%%, NMI=%.2f%%, Purity=%.2f%% (%.1fs)\n', ...
                ACC, NMI, Purity, elapsed);
            fprintf('  => View weights: [%.4f, %.4f]\n', alpha(1), alpha(2));
            
            % 保存结果
            results.configs{end+1} = struct('norm_mode', norm_mode, 'norm_name', norm_name, ...
                'gamma', gamma, 'lambda2', lambda2_phase1, 'beta', beta_phase1, ...
                'layers', layers_phase1);
            results.ACC(end+1) = ACC;
            results.NMI(end+1) = NMI;
            results.Purity(end+1) = Purity;
            results.time(end+1) = elapsed;
            results.alpha{end+1} = alpha;
            
            phase1_results = [phase1_results; norm_mode, gamma, ACC, NMI, Purity];
            
            % 实时写入CSV
            fid = fopen(output_csv, 'a');
            layer_str = sprintf('[%s]', sprintf('%d,', layers_phase1));
            fprintf(fid, '%d,%d,%s,%.2f,%.6f,%.1f,%s,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f\n', ...
                trial_count, norm_mode, norm_name, gamma, lambda2_phase1, beta_phase1, ...
                layer_str, ACC, NMI, Purity, elapsed, alpha(1), alpha(2));
            fclose(fid);
            
            % 更新最佳
            if ACC > best_ACC
                best_ACC = ACC;
                best_NMI = NMI;
                best_Purity = Purity;
                best_config = results.configs{end};
                best_trial = trial_count;
                fprintf('  *** NEW BEST ACC: %.2f%% ***\n', best_ACC);
            end
            
        catch ME
            fprintf('  ERROR: %s\n', ME.message);
        end
        fprintf('\n');
    end
end

% Phase 1 分析
fprintf('\n=== Phase 1 Summary ===\n');
[~, best_idx] = max(phase1_results(:, 3));
best_norm_phase1 = phase1_results(best_idx, 1);
best_gamma_phase1 = phase1_results(best_idx, 2);
fprintf('Best from Phase 1: Norm=%d, Gamma=%.2f => ACC=%.2f%%\n', ...
    best_norm_phase1, best_gamma_phase1, phase1_results(best_idx, 3));
fprintf('继续使用这些参数进入 Phase 2\n\n');

%% === 策略 2: 基于Phase 1最佳, 测试优秀Beta种子 × Lambda2 × K ===
fprintf('\n=== Phase 2: Testing Beta Seeds × Lambda2 × K ===\n');
fprintf('    使用 Phase 1 最佳: Norm=%d, Gamma=%.2f\n', best_norm_phase1, best_gamma_phase1);
fprintf('    测试已知优秀beta种子: [67,84,92,115,246,283,301,336]\n\n');

phase2_best_ACC = 0;
phase2_best_config = struct();

% 先测试优秀的beta种子
for beta_seed_idx = 1:length(beta_seeds)
    beta = beta_seeds(beta_seed_idx);
    
    % 对每个beta种子，尝试不同的lambda2和k组合
    for lambda2_idx = 1:length(lambda2_list)
        lambda2 = lambda2_list(lambda2_idx);
        
        for k_idx = 1:length(k_list)
            k = k_list(k_idx);
            trial_count = trial_count + 1;
            
            fprintf('Trial %d: Beta=%.0f (seed), Lambda2=%.6f, K=%d\n', trial_count, beta, lambda2, k);
            
            % 使用 Phase 1 最佳归一化和 gamma
            X = data_guiyi_choos(X_raw, best_norm_phase1);
            for v = 1:numView
                X{v} = NormalizeFea(X{v}, 0);
            end
            
            options = struct();
            options.lambda1 = lambda1_phase1;
            options.lambda2 = lambda2;
            options.beta = beta;
            options.gamma = best_gamma_phase1;
            options.graph_k = k;
            options.maxIter = 100;
            options.tol = 1e-5;
            
            try
                tic;
                [H, Z, alpha, obj_values] = GDMFC(X, numCluster, layers_phase1, options);
                elapsed = toc;
                
                S = H * H';
                S = (S + S') / 2;
                S = max(S, 0);
                predict_labels = SpectralClustering(S, numCluster);
                
                res = bestMap(y, predict_labels);
                ACC = length(find(y == res)) / length(y) * 100;
                NMI = MutualInfo(y, predict_labels) * 100;
                Purity = compute_purity(y, predict_labels) * 100;
                
                fprintf('  => ACC=%.2f%%, NMI=%.2f%%, Purity=%.2f%% (%.1fs)\n', ...
                    ACC, NMI, Purity, elapsed);
                
                results.configs{end+1} = struct('norm_mode', best_norm_phase1, ...
                    'norm_name', norm_names{best_norm_phase1}, ...
                    'gamma', best_gamma_phase1, 'lambda1', lambda1_phase1, ...
                    'lambda2', lambda2, 'beta', beta, 'k', k, 'layers', layers_phase1);
                results.ACC(end+1) = ACC;
                results.NMI(end+1) = NMI;
                results.Purity(end+1) = Purity;
                results.time(end+1) = elapsed;
                results.alpha{end+1} = alpha;
                
                % 写入CSV
                fid = fopen(output_csv, 'a');
                layer_str = sprintf('[%s]', sprintf('%d,', layers_phase1));
                fprintf(fid, '%d,%d,%s,%.2f,%.6f,%.1f,%s,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f\n', ...
                    trial_count, best_norm_phase1, norm_names{best_norm_phase1}, ...
                    best_gamma_phase1, lambda2, beta, layer_str, ACC, NMI, Purity, ...
                    elapsed, alpha(1), alpha(2));
                fclose(fid);
                
                if ACC > best_ACC
                    best_ACC = ACC;
                    best_NMI = NMI;
                    best_Purity = Purity;
                    best_config = results.configs{end};
                    best_trial = trial_count;
                    fprintf('  *** NEW GLOBAL BEST ACC: %.2f%% ***\n', best_ACC);
                end
                
                if ACC > phase2_best_ACC
                    phase2_best_ACC = ACC;
                    phase2_best_config = results.configs{end};
                end
                
            catch ME
                fprintf('  ERROR: %s\n', ME.message);
            end
            fprintf('\n');
        end
    end
end

fprintf('=== Phase 2 Summary ===\n');
fprintf('Best from Phase 2: ACC=%.2f%%, Beta=%.0f, Lambda2=%.6f, K=%d\n', ...
    phase2_best_ACC, phase2_best_config.beta, phase2_best_config.lambda2, phase2_best_config.k);
fprintf('\n');

%% === 策略 3: 测试不同的 Layer 结构 × Lambda1 ===
fprintf('=== Phase 3: Testing Layer Structures × Lambda1 ===\n');
fprintf('    使用当前全局最佳参数组合\n\n');

for layer_idx = 1:length(layer_configs)
    layers = layer_configs{layer_idx};
    
    for lambda1_idx = 1:length(lambda1_list)
        lambda1 = lambda1_list(lambda1_idx);
        trial_count = trial_count + 1;
        
        fprintf('Trial %d: Layers=[', trial_count);
        fprintf('%d ', layers);
        fprintf('], Lambda1=%.6f\n', lambda1);
        
        % 使用全局最佳参数
        X = data_guiyi_choos(X_raw, best_config.norm_mode);
        for v = 1:numView
            X{v} = NormalizeFea(X{v}, 0);
        end
        
        options = struct();
        options.lambda1 = lambda1;
        options.lambda2 = best_config.lambda2;
        options.beta = best_config.beta;
        options.gamma = best_config.gamma;
        options.graph_k = best_config.k;
        options.maxIter = 100;
        options.tol = 1e-5;
        
        try
            tic;
            [H, Z, alpha, obj_values] = GDMFC(X, numCluster, layers, options);
            elapsed = toc;
            
            S = H * H';
            S = (S + S') / 2;
            S = max(S, 0);
            predict_labels = SpectralClustering(S, numCluster);
            
            res = bestMap(y, predict_labels);
            ACC = length(find(y == res)) / length(y) * 100;
            NMI = MutualInfo(y, predict_labels) * 100;
            Purity = compute_purity(y, predict_labels) * 100;
            
            fprintf('  => ACC=%.2f%%, NMI=%.2f%%, Purity=%.2f%% (%.1fs)\n', ...
                ACC, NMI, Purity, elapsed);
            
            results.configs{end+1} = struct('norm_mode', best_config.norm_mode, ...
                'norm_name', best_config.norm_name, 'gamma', best_config.gamma, ...
                'lambda1', lambda1, 'lambda2', best_config.lambda2, ...
                'beta', best_config.beta, 'k', best_config.k, 'layers', layers);
            results.ACC(end+1) = ACC;
            results.NMI(end+1) = NMI;
            results.Purity(end+1) = Purity;
            results.time(end+1) = elapsed;
            results.alpha{end+1} = alpha;
            
            % 写入CSV
            fid = fopen(output_csv, 'a');
            layer_str = sprintf('[%s]', sprintf('%d,', layers));
            fprintf(fid, '%d,%d,%s,%.2f,%.6f,%.1f,%s,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f\n', ...
                trial_count, best_config.norm_mode, best_config.norm_name, ...
                best_config.gamma, best_config.lambda2, best_config.beta, ...
                layer_str, ACC, NMI, Purity, elapsed, alpha(1), alpha(2));
            fclose(fid);
            
            if ACC > best_ACC
                best_ACC = ACC;
                best_NMI = NMI;
                best_Purity = Purity;
                best_config = results.configs{end};
                best_trial = trial_count;
                fprintf('  *** NEW GLOBAL BEST ACC: %.2f%% ***\n', best_ACC);
            end
            
        catch ME
            fprintf('  ERROR: %s\n', ME.message);
        end
        fprintf('\n');
    end
end

%% ==================== Step 5: Final Analysis ====================
fprintf('\n========================================\n');
fprintf('Parameter Search Completed!\n');
fprintf('========================================\n\n');

fprintf('Total trials: %d\n', trial_count);
fprintf('Valid trials: %d\n', length(results.ACC));
fprintf('\n');

fprintf('=== GLOBAL BEST CONFIGURATION (Trial %d) ===\n', best_trial);
fprintf('========================================\n');
fprintf('Normalization: %s (mode=%d)\n', best_config.norm_name, best_config.norm_mode);
fprintf('Gamma:         %.2f\n', best_config.gamma);
fprintf('Lambda1:       %.6f\n', best_config.lambda1);
fprintf('Lambda2:       %.6f\n', best_config.lambda2);
fprintf('Beta:          %.1f\n', best_config.beta);
fprintf('Graph K:       %d\n', best_config.k);
fprintf('Layers:        [');
for i = 1:length(best_config.layers)
    fprintf('%d', best_config.layers(i));
    if i < length(best_config.layers), fprintf(', '); end
end
fprintf(', %d]\n\n', numCluster);

fprintf('=== BEST PERFORMANCE ===\n');
fprintf('ACC:    %.2f%% (Target: %.2f%%, Gap: %.2f%%)\n', best_ACC, target_ACC, target_ACC - best_ACC);
fprintf('NMI:    %.2f%% (Target: %.2f%%, Gap: %.2f%%)\n', best_NMI, target_NMI, target_NMI - best_NMI);
fprintf('Purity: %.2f%% (Target: %.2f%%, Gap: %.2f%%)\n', best_Purity, target_Purity, target_Purity - best_Purity);
fprintf('========================================\n\n');

% Top 10 配置
fprintf('=== Top 10 Configurations by ACC ===\n');
[sorted_ACC, sort_idx] = sort(results.ACC, 'descend');
for rank = 1:min(10, length(results.ACC))
    idx = sort_idx(rank);
    cfg = results.configs{idx};
    fprintf('%2d. ACC=%.2f%%, NMI=%.2f%%, Purity=%.2f%%\n', rank, ...
        results.ACC(idx), results.NMI(idx), results.Purity(idx));
    fprintf('    Norm=%s, γ=%.2f, λ1=%.5f, λ2=%.5f, β=%.0f, k=%d, Layers=[', ...
        cfg.norm_name, cfg.gamma, cfg.lambda1, cfg.lambda2, cfg.beta, cfg.k);
    for i = 1:length(cfg.layers)
        fprintf('%d', cfg.layers(i));
        if i < length(cfg.layers), fprintf(','); end
    end
    fprintf(']\n');
end
fprintf('\n');

%% ==================== Step 6: Visualization ====================
fprintf('Step 6: Generating visualizations...\n');

figure('Name', 'Parameter Optimization Results', 'Position', [100, 100, 1400, 900]);

% Plot 1: ACC vs NMI
subplot(2, 3, 1);
scatter(results.ACC, results.NMI, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot(best_ACC, best_NMI, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
plot(target_ACC, target_NMI, 'gs', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('ACC (%)', 'FontSize', 11);
ylabel('NMI (%)', 'FontSize', 11);
title('ACC vs NMI', 'FontSize', 12, 'FontWeight', 'bold');
legend('Trials', 'Best', 'Target', 'Location', 'best');
grid on;

% Plot 2: ACC vs Purity
subplot(2, 3, 2);
scatter(results.ACC, results.Purity, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot(best_ACC, best_Purity, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
plot(target_ACC, target_Purity, 'gs', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('ACC (%)', 'FontSize', 11);
ylabel('Purity (%)', 'FontSize', 11);
title('ACC vs Purity', 'FontSize', 12, 'FontWeight', 'bold');
legend('Trials', 'Best', 'Target', 'Location', 'best');
grid on;

% Plot 3: Performance over trials
subplot(2, 3, 3);
plot(1:length(results.ACC), results.ACC, 'b-o', 'MarkerSize', 4);
hold on;
plot(1:length(results.NMI), results.NMI, 'r-s', 'MarkerSize', 4);
plot(1:length(results.Purity), results.Purity, 'g-^', 'MarkerSize', 4);
yline(target_ACC, 'b--', 'LineWidth', 1.5);
yline(target_NMI, 'r--', 'LineWidth', 1.5);
yline(target_Purity, 'g--', 'LineWidth', 1.5);
xlabel('Trial', 'FontSize', 11);
ylabel('Performance (%)', 'FontSize', 11);
title('Performance over Trials', 'FontSize', 12, 'FontWeight', 'bold');
legend('ACC', 'NMI', 'Purity', 'Target ACC', 'Target NMI', 'Target Purity', ...
    'Location', 'best', 'FontSize', 8);
grid on;

% Plot 4: Normalization impact
subplot(2, 3, 4);
norm_modes_used = cellfun(@(x) x.norm_mode, results.configs);
boxplot(results.ACC, norm_modes_used, 'Labels', norm_names);
ylabel('ACC (%)', 'FontSize', 11);
xlabel('Normalization Method', 'FontSize', 11);
title('Normalization Impact on ACC', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xtickangle(15);

% Plot 5: Gamma impact
subplot(2, 3, 5);
gamma_used = cellfun(@(x) x.gamma, results.configs);
scatter(gamma_used, results.ACC, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Gamma', 'FontSize', 11);
ylabel('ACC (%)', 'FontSize', 11);
title('Gamma Impact on ACC', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Plot 6: Lambda2 impact
subplot(2, 3, 6);
lambda2_used = cellfun(@(x) x.lambda2, results.configs);
scatter(lambda2_used, results.ACC, 50, 'filled', 'MarkerFaceAlpha', 0.6);
set(gca, 'XScale', 'log');
xlabel('Lambda2 (log scale)', 'FontSize', 11);
ylabel('ACC (%)', 'FontSize', 11);
title('Lambda2 Impact on ACC', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

saveas(gcf, 'best_param_analysis.png');
fprintf('  Saved visualization to best_param_analysis.png\n\n');

%% ==================== Step 7: Save Results ====================
fprintf('Step 7: Saving results...\n');

save(output_mat, 'results', 'best_config', 'best_ACC', 'best_NMI', 'best_Purity', ...
    'target_ACC', 'target_NMI', 'target_Purity', 'best_trial');

% Save best config to text file
fid = fopen('best_config_ORL_optimized.txt', 'w');
fprintf(fid, '====================================\n');
fprintf(fid, 'GDMFC Optimized Configuration\n');
fprintf(fid, '====================================\n\n');
fprintf(fid, '=== Parameters ===\n');
fprintf(fid, 'Normalization: %s (mode=%d)\n', best_config.norm_name, best_config.norm_mode);
fprintf(fid, 'Gamma:         %.2f\n', best_config.gamma);
fprintf(fid, 'Lambda1:       %.6f\n', best_config.lambda1);
fprintf(fid, 'Lambda2:       %.6f\n', best_config.lambda2);
fprintf(fid, 'Beta:          %.1f\n', best_config.beta);
fprintf(fid, 'Graph K:       %d\n', best_config.k);
fprintf(fid, 'Layers:        [');
for i = 1:length(best_config.layers)
    fprintf(fid, '%d', best_config.layers(i));
    if i < length(best_config.layers), fprintf(fid, ', '); end
end
fprintf(fid, ', %d]\n\n', numCluster);
fprintf(fid, '=== Performance ===\n');
fprintf(fid, 'ACC:    %.2f%% (Target: %.2f%%)\n', best_ACC, target_ACC);
fprintf(fid, 'NMI:    %.2f%% (Target: %.2f%%)\n', best_NMI, target_NMI);
fprintf(fid, 'Purity: %.2f%% (Target: %.2f%%)\n', best_Purity, target_Purity);
fprintf(fid, '\n=== MATLAB Code ===\n');
fprintf(fid, '%% Apply best preprocessing\n');
fprintf(fid, 'X = data_guiyi_choos(X_raw, %d);  %% %s\n', ...
    best_config.norm_mode, best_config.norm_name);
fprintf(fid, 'for v = 1:numView\n');
fprintf(fid, '    X{v} = NormalizeFea(X{v}, 0);\n');
fprintf(fid, 'end\n\n');
fprintf(fid, '%% Set options\n');
fprintf(fid, 'options.lambda1 = %.6f;\n', best_config.lambda1);
fprintf(fid, 'options.lambda2 = %.6f;\n', best_config.lambda2);
fprintf(fid, 'options.beta = %.1f;\n', best_config.beta);
fprintf(fid, 'options.gamma = %.2f;\n', best_config.gamma);
fprintf(fid, 'options.graph_k = %d;\n', best_config.k);
fprintf(fid, 'layers = [');
for i = 1:length(best_config.layers)
    fprintf(fid, '%d', best_config.layers(i));
    if i < length(best_config.layers), fprintf(fid, ', '); end
end
fprintf(fid, '];\n');
fclose(fid);

fprintf('  Results saved to:\n');
fprintf('    - %s\n', output_csv);
fprintf('    - %s\n', output_mat);
fprintf('    - best_config_ORL_optimized.txt\n\n');

fprintf('========================================\n');
fprintf('Optimization Completed!\n');
fprintf('========================================\n\n');

% === 教授建议总结 ===
fprintf('=== 基于种子的优化策略实施总结 ===\n');
fprintf('1. ✓ 使用已知最佳beta种子: [67,84,92,115,246,283,301,336]\n');
fprintf('2. ✓ 归一化方法对比: MinMax vs L2 (教授建议)\n');
fprintf('3. ✓ Gamma 从1.2提高到2.0+ (多视图协同优化)\n');
fprintf('4. ✓ Lambda2 精细调优: 正交约束强度优化\n');
fprintf('5. ✓ Graph K 优化: 邻居数影响分析\n');
fprintf('6. ✓ Lambda1 微调: HSIC多样性权重\n');
fprintf('7. ✓ 网络结构优化: 7种Layer配置 (包含更宽结构)\n');
fprintf('8. ✓ 三阶段智能搜索:\n');
fprintf('    Phase 1: Norm×Gamma (关键组合)\n');
fprintf('    Phase 2: Beta种子×Lambda2×K (基于已知优秀值)\n');
fprintf('    Phase 3: Layer×Lambda1 (结构精调)\n');
fprintf('========================================\n\n');

fprintf('=== 与baseline对比 ===\n');
fprintf('Baseline (beta=115, norm=L2, γ=1.2):\n');
fprintf('  ACC=86.00%%, NMI=92.79%%, Purity=87.00%%\n\n');
fprintf('Current Best (Trial %d):\n', best_trial);
fprintf('  ACC=%.2f%% (%+.2f%%), NMI=%.2f%% (%+.2f%%), Purity=%.2f%% (%+.2f%%)\n', ...
    best_ACC, best_ACC-86.0, best_NMI, best_NMI-92.79, best_Purity, best_Purity-87.0);
fprintf('  Norm=%s, γ=%.2f, β=%.0f\n\n', ...
    best_config.norm_name, best_config.gamma, best_config.beta);

fprintf('Distance to Target:\n');
fprintf('  ACC:    %.2f%% / 87.80%% (还需 %+.2f%%)\n', best_ACC, 87.80-best_ACC);
fprintf('  NMI:    %.2f%% / 93.81%% (还需 %+.2f%%)\n', best_NMI, 93.81-best_NMI);
fprintf('  Purity: %.2f%% / 89.70%% (还需 %+.2f%%)\n', best_Purity, 89.70-best_Purity);
fprintf('========================================\n');
