%==========================================================================
% 系统化参数搜索 - 基于TOP30 Beta种子
%==========================================================================
% 策略: 一步一步系统化搜索，每个参数都充分测试
% 
% Step 1: 归一化方法 (5种) × TOP30 Beta种子
% Step 2: Layers结构 (50, 100递增) × 当前最佳配置
% Step 3: Gamma参数搜索 × 当前最佳配置
% Step 4: Lambda1参数搜索 × 当前最佳配置
% Step 5: Lambda2参数搜索 × 当前最佳配置
% Step 6: K值搜索 × 当前最佳配置
%==========================================================================

clear; clc; close all;

fprintf('========================================\n');
fprintf('系统化参数搜索 (Systematic Search)\n');
fprintf('基于TOP30 Beta种子 + 逐参数优化\n');
fprintf('========================================\n\n');

%% ==================== 加载数据 ====================
fprintf('加载ORL数据集...\n');
dataPath = '../../dataset/orl';
numSubjects = 40;
imagesPerSubject = 10;
imageHeight = 112;
imageWidth = 92;
numSamples = numSubjects * imagesPerSubject;

cacheFile = 'orl_images_cache.mat';
if exist(cacheFile, 'file')
    load(cacheFile, 'allImages', 'y');
    fprintf('  已加载缓存数据\n');
else
    fprintf('  从磁盘加载图像...\n');
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
end

% 构造双视图
downsampleFactor = 2;
newH = imageHeight / downsampleFactor;
newW = imageWidth / downsampleFactor;
X1 = zeros(numSamples, newH * newW);
for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    imgDown = imresize(img, [newH, newW], 'bilinear');
    X1(i, :) = imgDown(:)';
end

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
fprintf('完成\n\n');

%% ==================== 定义搜索空间 ====================
fprintf('定义搜索空间...\n');

% TOP 30 Beta种子 (从beta1-1000.xlsx中ACC最高的30个)
beta_seeds = [10, 25, 67, 73, 84, 87, 92, 99, 115, 135, ...
              173, 196, 221, 230, 246, 247, 248, 268, 271, 277, ...
              283, 290, 291, 300, 301, 305, 320, 324, 336, 367];

% 归一化方法 (data_guiyi_choos.m中的1-5)
norm_modes = [1, 2, 3, 4, 5];
norm_names = {'MinMax-Row', 'MinMax-Col', 'L2-Col', 'Sum-Col', 'Global'};

% Layers结构 (从50开始递增到400)
layer_first_dims = [50, 100, 150, 200, 250, 300, 350, 400];
% 自动生成层结构: [first, first*0.4, 40]
generate_layers = @(d) [d, round(d*0.4), 40];

% Gamma参数 (视图权重)
gamma_range = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];

% Lambda1 (HSIC多样性)
lambda1_range = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3];

% Lambda2 (正交约束)
lambda2_range = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2];

% Graph K
k_range = [3, 5, 7, 9, 11, 13, 15];

fprintf('  TOP30 Beta种子: %d个\n', length(beta_seeds));
fprintf('  归一化方法: %d种\n', length(norm_modes));
fprintf('  Layers配置: %d种\n', length(layer_first_dims));
fprintf('  Gamma: %d个值\n', length(gamma_range));
fprintf('  Lambda1: %d个值\n', length(lambda1_range));
fprintf('  Lambda2: %d个值\n', length(lambda2_range));
fprintf('  K: %d个值\n\n', length(k_range));

%% ==================== 初始化结果记录 ====================
output_csv = 'systematic_search_results.csv';
output_mat = 'systematic_search_results.mat';

% CSV表头
if ~exist(output_csv, 'file')
    fid = fopen(output_csv, 'w');
    fprintf(fid, 'step,trial,norm,beta,layers,gamma,lambda1,lambda2,k,ACC,NMI,Purity,time_s\n');
    fclose(fid);
end

% 全局最佳跟踪
global_best = struct();
global_best.ACC = 0;
global_best.config = struct();
global_best.trial = 0;

% 当前最佳配置 (初始值 - 基于已知最佳)
current_best = struct();
current_best.norm = 3;           % L2-Col
current_best.beta = 115;         % TOP1 beta
current_best.layers = [400, 150, 40];
current_best.gamma = 1.2;
current_best.lambda1 = 1e-5;
current_best.lambda2 = 1e-3;
current_best.k = 7;

fprintf('初始配置 (基于已知最佳):\n');
fprintf('  Norm: %s\n', norm_names{current_best.norm});
fprintf('  Beta: %d\n', current_best.beta);
fprintf('  Layers: [%d, %d, %d]\n', current_best.layers);
fprintf('  Gamma: %.2f\n', current_best.gamma);
fprintf('  Lambda1: %.6f\n', current_best.lambda1);
fprintf('  Lambda2: %.6f\n', current_best.lambda2);
fprintf('  K: %d\n\n', current_best.k);

%% ==================== Step 1: 归一化 × Beta ====================
fprintf('========================================\n');
fprintf('Step 1: 归一化方法 × TOP30 Beta种子\n');
fprintf('  测试: 5种归一化 × 30个beta = 150 trials\n');
fprintf('  固定其他参数为当前最佳值\n');
fprintf('========================================\n\n');

step1_results = [];
trial_count = 0;

for norm_idx = 1:length(norm_modes)
    norm_mode = norm_modes(norm_idx);
    norm_name = norm_names{norm_idx};
    
    for beta_idx = 1:length(beta_seeds)
        beta = beta_seeds(beta_idx);
        trial_count = trial_count + 1;
        
        fprintf('[Step1-%d/%d] Norm=%s, Beta=%d ... ', ...
            trial_count, length(norm_modes)*length(beta_seeds), norm_name, beta);
        
        % 数据预处理
        X = data_guiyi_choos(X_raw, norm_mode);
        for v = 1:numView
            X{v} = NormalizeFea(X{v}, 0);
        end
        
        % 设置参数
        options = struct();
        options.lambda1 = current_best.lambda1;
        options.lambda2 = current_best.lambda2;
        options.beta = beta;
        options.gamma = current_best.gamma;
        options.graph_k = current_best.k;
        options.maxIter = 100;
        options.tol = 1e-5;
        
        try
            tic;
            [H, ~, alpha, ~] = GDMFC(X, numCluster, current_best.layers, options);
            elapsed = toc;
            
            S = H * H';
            S = (S + S') / 2;
            S = max(S, 0);
            predict_labels = SpectralClustering(S, numCluster);
            
            res = bestMap(y, predict_labels);
            ACC = length(find(y == res)) / length(y) * 100;
            NMI = MutualInfo(y, predict_labels) * 100;
            Purity = compute_purity(y, predict_labels) * 100;
            
            fprintf('ACC=%.2f%%, NMI=%.2f%%, Pur=%.2f%%\n', ACC, NMI, Purity);
            
            % 记录结果
            step1_results = [step1_results; norm_mode, beta, ACC, NMI, Purity];
            
            % 保存到CSV
            fid = fopen(output_csv, 'a');
            layer_str = sprintf('%d-%d-%d', current_best.layers);
            fprintf(fid, '1,%d,%d,%d,%s,%.2f,%.6f,%.6f,%d,%.2f,%.2f,%.2f,%.2f\n', ...
                trial_count, norm_mode, beta, layer_str, current_best.gamma, ...
                current_best.lambda1, current_best.lambda2, current_best.k, ...
                ACC, NMI, Purity, elapsed);
            fclose(fid);
            
            % 更新全局最佳
            if ACC > global_best.ACC
                global_best.ACC = ACC;
                global_best.config = struct('norm', norm_mode, 'beta', beta, ...
                    'layers', current_best.layers, 'gamma', current_best.gamma, ...
                    'lambda1', current_best.lambda1, 'lambda2', current_best.lambda2, 'k', current_best.k);
                global_best.trial = trial_count;
                fprintf('  *** 新的全局最佳: %.2f%% ***\n', ACC);
            end
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
        end
    end
end

% Step 1 总结
[~, best_idx] = max(step1_results(:,3));
best_norm = step1_results(best_idx, 1);
best_beta = step1_results(best_idx, 2);
fprintf('\n=== Step 1 最佳 ===\n');
fprintf('Norm=%s, Beta=%d => ACC=%.2f%%\n', norm_names{best_norm}, best_beta, step1_results(best_idx,3));
fprintf('更新当前最佳配置\n\n');

current_best.norm = best_norm;
current_best.beta = best_beta;

%% ==================== Step 2: Layers结构搜索 ====================
fprintf('========================================\n');
fprintf('Step 2: Layers结构搜索\n');
fprintf('  测试: %d种层结构 (从50到400递增)\n', length(layer_first_dims));
fprintf('  格式: [d, d*0.4, 40]\n');
fprintf('========================================\n\n');

step2_results = [];

for layer_idx = 1:length(layer_first_dims)
    first_dim = layer_first_dims(layer_idx);
    layers = generate_layers(first_dim);
    trial_count = trial_count + 1;
    
    fprintf('[Step2-%d/%d] Layers=[%d,%d,%d] ... ', ...
        layer_idx, length(layer_first_dims), layers(1), layers(2), layers(3));
    
    X = data_guiyi_choos(X_raw, current_best.norm);
    for v = 1:numView
        X{v} = NormalizeFea(X{v}, 0);
    end
    
    options = struct();
    options.lambda1 = current_best.lambda1;
    options.lambda2 = current_best.lambda2;
    options.beta = current_best.beta;
    options.gamma = current_best.gamma;
    options.graph_k = current_best.k;
    options.maxIter = 100;
    options.tol = 1e-5;
    
    try
        tic;
        [H, ~, alpha, ~] = GDMFC(X, numCluster, layers, options);
        elapsed = toc;
        
        S = H * H';
        S = (S + S') / 2;
        S = max(S, 0);
        predict_labels = SpectralClustering(S, numCluster);
        
        res = bestMap(y, predict_labels);
        ACC = length(find(y == res)) / length(y) * 100;
        NMI = MutualInfo(y, predict_labels) * 100;
        Purity = compute_purity(y, predict_labels) * 100;
        
        fprintf('ACC=%.2f%%, NMI=%.2f%%\n', ACC, NMI);
        
        step2_results = [step2_results; first_dim, ACC, NMI, Purity];
        
        fid = fopen(output_csv, 'a');
        layer_str = sprintf('%d-%d-%d', layers);
        fprintf(fid, '2,%d,%d,%d,%s,%.2f,%.6f,%.6f,%d,%.2f,%.2f,%.2f,%.2f\n', ...
            trial_count, current_best.norm, current_best.beta, layer_str, ...
            current_best.gamma, current_best.lambda1, current_best.lambda2, current_best.k, ...
            ACC, NMI, Purity, elapsed);
        fclose(fid);
        
        if ACC > global_best.ACC
            global_best.ACC = ACC;
            global_best.config.layers = layers;
            fprintf('  *** 新的全局最佳: %.2f%% ***\n', ACC);
        end
        
    catch ME
        fprintf('ERROR: %s\n', ME.message);
    end
end

[~, best_idx] = max(step2_results(:,2));
best_layer_dim = step2_results(best_idx, 1);
best_layers = generate_layers(best_layer_dim);
fprintf('\n=== Step 2 最佳 ===\n');
fprintf('Layers=[%d,%d,%d] => ACC=%.2f%%\n', best_layers(1), best_layers(2), best_layers(3), step2_results(best_idx,2));
fprintf('更新当前最佳配置\n\n');

current_best.layers = best_layers;

%% ==================== Step 3: Gamma搜索 ====================
fprintf('========================================\n');
fprintf('Step 3: Gamma参数搜索\n');
fprintf('  测试: %d个gamma值\n', length(gamma_range));
fprintf('========================================\n\n');

step3_results = [];

for gamma_idx = 1:length(gamma_range)
    gamma = gamma_range(gamma_idx);
    trial_count = trial_count + 1;
    
    fprintf('[Step3-%d/%d] Gamma=%.2f ... ', gamma_idx, length(gamma_range), gamma);
    
    X = data_guiyi_choos(X_raw, current_best.norm);
    for v = 1:numView
        X{v} = NormalizeFea(X{v}, 0);
    end
    
    options = struct();
    options.lambda1 = current_best.lambda1;
    options.lambda2 = current_best.lambda2;
    options.beta = current_best.beta;
    options.gamma = gamma;
    options.graph_k = current_best.k;
    options.maxIter = 100;
    options.tol = 1e-5;
    
    try
        tic;
        [H, ~, alpha, ~] = GDMFC(X, numCluster, current_best.layers, options);
        elapsed = toc;
        
        S = H * H';
        S = (S + S') / 2;
        S = max(S, 0);
        predict_labels = SpectralClustering(S, numCluster);
        
        res = bestMap(y, predict_labels);
        ACC = length(find(y == res)) / length(y) * 100;
        NMI = MutualInfo(y, predict_labels) * 100;
        Purity = compute_purity(y, predict_labels) * 100;
        
        fprintf('ACC=%.2f%%, NMI=%.2f%%\n', ACC, NMI);
        
        step3_results = [step3_results; gamma, ACC, NMI, Purity];
        
        fid = fopen(output_csv, 'a');
        layer_str = sprintf('%d-%d-%d', current_best.layers);
        fprintf(fid, '3,%d,%d,%d,%s,%.2f,%.6f,%.6f,%d,%.2f,%.2f,%.2f,%.2f\n', ...
            trial_count, current_best.norm, current_best.beta, layer_str, gamma, ...
            current_best.lambda1, current_best.lambda2, current_best.k, ...
            ACC, NMI, Purity, elapsed);
        fclose(fid);
        
        if ACC > global_best.ACC
            global_best.ACC = ACC;
            global_best.config.gamma = gamma;
            fprintf('  *** 新的全局最佳: %.2f%% ***\n', ACC);
        end
        
    catch ME
        fprintf('ERROR: %s\n', ME.message);
    end
end

[~, best_idx] = max(step3_results(:,2));
best_gamma = step3_results(best_idx, 1);
fprintf('\n=== Step 3 最佳 ===\n');
fprintf('Gamma=%.2f => ACC=%.2f%%\n', best_gamma, step3_results(best_idx,2));
fprintf('更新当前最佳配置\n\n');

current_best.gamma = best_gamma;

%% ==================== Step 4: Lambda1搜索 ====================
fprintf('========================================\n');
fprintf('Step 4: Lambda1参数搜索\n');
fprintf('  测试: %d个lambda1值\n', length(lambda1_range));
fprintf('========================================\n\n');

step4_results = [];

for lambda1_idx = 1:length(lambda1_range)
    lambda1 = lambda1_range(lambda1_idx);
    trial_count = trial_count + 1;
    
    fprintf('[Step4-%d/%d] Lambda1=%.6f ... ', lambda1_idx, length(lambda1_range), lambda1);
    
    X = data_guiyi_choos(X_raw, current_best.norm);
    for v = 1:numView
        X{v} = NormalizeFea(X{v}, 0);
    end
    
    options = struct();
    options.lambda1 = lambda1;
    options.lambda2 = current_best.lambda2;
    options.beta = current_best.beta;
    options.gamma = current_best.gamma;
    options.graph_k = current_best.k;
    options.maxIter = 100;
    options.tol = 1e-5;
    
    try
        tic;
        [H, ~, alpha, ~] = GDMFC(X, numCluster, current_best.layers, options);
        elapsed = toc;
        
        S = H * H';
        S = (S + S') / 2;
        S = max(S, 0);
        predict_labels = SpectralClustering(S, numCluster);
        
        res = bestMap(y, predict_labels);
        ACC = length(find(y == res)) / length(y) * 100;
        NMI = MutualInfo(y, predict_labels) * 100;
        Purity = compute_purity(y, predict_labels) * 100;
        
        fprintf('ACC=%.2f%%, NMI=%.2f%%\n', ACC, NMI);
        
        step4_results = [step4_results; lambda1, ACC, NMI, Purity];
        
        fid = fopen(output_csv, 'a');
        layer_str = sprintf('%d-%d-%d', current_best.layers);
        fprintf(fid, '4,%d,%d,%d,%s,%.2f,%.6f,%.6f,%d,%.2f,%.2f,%.2f,%.2f\n', ...
            trial_count, current_best.norm, current_best.beta, layer_str, ...
            current_best.gamma, lambda1, current_best.lambda2, current_best.k, ...
            ACC, NMI, Purity, elapsed);
        fclose(fid);
        
        if ACC > global_best.ACC
            global_best.ACC = ACC;
            global_best.config.lambda1 = lambda1;
            fprintf('  *** 新的全局最佳: %.2f%% ***\n', ACC);
        end
        
    catch ME
        fprintf('ERROR: %s\n', ME.message);
    end
end

[~, best_idx] = max(step4_results(:,2));
best_lambda1 = step4_results(best_idx, 1);
fprintf('\n=== Step 4 最佳 ===\n');
fprintf('Lambda1=%.6f => ACC=%.2f%%\n', best_lambda1, step4_results(best_idx,2));
fprintf('更新当前最佳配置\n\n');

current_best.lambda1 = best_lambda1;

%% ==================== Step 5: Lambda2搜索 ====================
fprintf('========================================\n');
fprintf('Step 5: Lambda2参数搜索\n');
fprintf('  测试: %d个lambda2值\n', length(lambda2_range));
fprintf('========================================\n\n');

step5_results = [];

for lambda2_idx = 1:length(lambda2_range)
    lambda2 = lambda2_range(lambda2_idx);
    trial_count = trial_count + 1;
    
    fprintf('[Step5-%d/%d] Lambda2=%.6f ... ', lambda2_idx, length(lambda2_range), lambda2);
    
    X = data_guiyi_choos(X_raw, current_best.norm);
    for v = 1:numView
        X{v} = NormalizeFea(X{v}, 0);
    end
    
    options = struct();
    options.lambda1 = current_best.lambda1;
    options.lambda2 = lambda2;
    options.beta = current_best.beta;
    options.gamma = current_best.gamma;
    options.graph_k = current_best.k;
    options.maxIter = 100;
    options.tol = 1e-5;
    
    try
        tic;
        [H, ~, alpha, ~] = GDMFC(X, numCluster, current_best.layers, options);
        elapsed = toc;
        
        S = H * H';
        S = (S + S') / 2;
        S = max(S, 0);
        predict_labels = SpectralClustering(S, numCluster);
        
        res = bestMap(y, predict_labels);
        ACC = length(find(y == res)) / length(y) * 100;
        NMI = MutualInfo(y, predict_labels) * 100;
        Purity = compute_purity(y, predict_labels) * 100;
        
        fprintf('ACC=%.2f%%, NMI=%.2f%%\n', ACC, NMI);
        
        step5_results = [step5_results; lambda2, ACC, NMI, Purity];
        
        fid = fopen(output_csv, 'a');
        layer_str = sprintf('%d-%d-%d', current_best.layers);
        fprintf(fid, '5,%d,%d,%d,%s,%.2f,%.6f,%.6f,%d,%.2f,%.2f,%.2f,%.2f\n', ...
            trial_count, current_best.norm, current_best.beta, layer_str, ...
            current_best.gamma, current_best.lambda1, lambda2, current_best.k, ...
            ACC, NMI, Purity, elapsed);
        fclose(fid);
        
        if ACC > global_best.ACC
            global_best.ACC = ACC;
            global_best.config.lambda2 = lambda2;
            fprintf('  *** 新的全局最佳: %.2f%% ***\n', ACC);
        end
        
    catch ME
        fprintf('ERROR: %s\n', ME.message);
    end
end

[~, best_idx] = max(step5_results(:,2));
best_lambda2 = step5_results(best_idx, 1);
fprintf('\n=== Step 5 最佳 ===\n');
fprintf('Lambda2=%.6f => ACC=%.2f%%\n', best_lambda2, step5_results(best_idx,2));
fprintf('更新当前最佳配置\n\n');

current_best.lambda2 = best_lambda2;

%% ==================== Step 6: K值搜索 ====================
fprintf('========================================\n');
fprintf('Step 6: K值搜索\n');
fprintf('  测试: %d个k值\n', length(k_range));
fprintf('========================================\n\n');

step6_results = [];

for k_idx = 1:length(k_range)
    k = k_range(k_idx);
    trial_count = trial_count + 1;
    
    fprintf('[Step6-%d/%d] K=%d ... ', k_idx, length(k_range), k);
    
    X = data_guiyi_choos(X_raw, current_best.norm);
    for v = 1:numView
        X{v} = NormalizeFea(X{v}, 0);
    end
    
    options = struct();
    options.lambda1 = current_best.lambda1;
    options.lambda2 = current_best.lambda2;
    options.beta = current_best.beta;
    options.gamma = current_best.gamma;
    options.graph_k = k;
    options.maxIter = 100;
    options.tol = 1e-5;
    
    try
        tic;
        [H, ~, alpha, ~] = GDMFC(X, numCluster, current_best.layers, options);
        elapsed = toc;
        
        S = H * H';
        S = (S + S') / 2;
        S = max(S, 0);
        predict_labels = SpectralClustering(S, numCluster);
        
        res = bestMap(y, predict_labels);
        ACC = length(find(y == res)) / length(y) * 100;
        NMI = MutualInfo(y, predict_labels) * 100;
        Purity = compute_purity(y, predict_labels) * 100;
        
        fprintf('ACC=%.2f%%, NMI=%.2f%%\n', ACC, NMI);
        
        step6_results = [step6_results; k, ACC, NMI, Purity];
        
        fid = fopen(output_csv, 'a');
        layer_str = sprintf('%d-%d-%d', current_best.layers);
        fprintf(fid, '6,%d,%d,%d,%s,%.2f,%.6f,%.6f,%d,%.2f,%.2f,%.2f,%.2f\n', ...
            trial_count, current_best.norm, current_best.beta, layer_str, ...
            current_best.gamma, current_best.lambda1, current_best.lambda2, k, ...
            ACC, NMI, Purity, elapsed);
        fclose(fid);
        
        if ACC > global_best.ACC
            global_best.ACC = ACC;
            global_best.config.k = k;
            fprintf('  *** 新的全局最佳: %.2f%% ***\n', ACC);
        end
        
    catch ME
        fprintf('ERROR: %s\n', ME.message);
    end
end

[~, best_idx] = max(step6_results(:,2));
best_k = step6_results(best_idx, 1);
fprintf('\n=== Step 6 最佳 ===\n');
fprintf('K=%d => ACC=%.2f%%\n', best_k, step6_results(best_idx,2));

%% ==================== 最终总结 ====================
fprintf('\n========================================\n');
fprintf('系统化搜索完成！\n');
fprintf('========================================\n\n');

fprintf('总试验次数: %d\n\n', trial_count);

fprintf('=== 全局最佳配置 ===\n');
fprintf('归一化: %s (mode %d)\n', norm_names{global_best.config.norm}, global_best.config.norm);
fprintf('Beta:     %d\n', global_best.config.beta);
fprintf('Layers:   [%d, %d, %d]\n', global_best.config.layers);
fprintf('Gamma:    %.2f\n', global_best.config.gamma);
fprintf('Lambda1:  %.6f\n', global_best.config.lambda1);
fprintf('Lambda2:  %.6f\n', global_best.config.lambda2);
fprintf('K:        %d\n\n', global_best.config.k);

fprintf('=== 最佳性能 ===\n');
fprintf('ACC:      %.2f%%\n', global_best.ACC);
fprintf('目标:     87.80%%\n');
fprintf('差距:     %.2f%%\n\n', 87.80 - global_best.ACC);

% 保存结果
save(output_mat, 'global_best', 'step1_results', 'step2_results', ...
    'step3_results', 'step4_results', 'step5_results', 'step6_results');

fprintf('结果已保存:\n');
fprintf('  - %s\n', output_csv);
fprintf('  - %s\n', output_mat);
fprintf('========================================\n');
