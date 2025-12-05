%==========================================================================
% GDMFC Refined Parameter Search (基于实验反馈v2)
%==========================================================================
% 当前最佳 (Phase 1结果):
%   ACC: 85.50% @ γ=5.0, λ2=0.002, β=246, k=7, Norm=L2-Col
%   Gap: ACC +2.30%, NMI +1.69%, Purity +3.20%
%
% 核心发现:
%   ✓ γ=5.0 >> γ=1.2-3.0 (提升明显!)
%   ✓ β=246,283,84 表现最佳
%   ✓ λ2=0.002~0.01 优于 0.001
%   ✓ L2-Col归一化意外最优
%   ✓ [400,150,40]网络最优
%
% 新策略: 聚焦最有潜力的参数组合
%==========================================================================

clear; clc; close all;

fprintf('========================================\n');
fprintf('GDMFC Refined Parameter Search v2\n');
fprintf('Current Best: ACC=85.50%%, Target=87.80%%\n');
fprintf('========================================\n\n');

%% ====================  Load Data ====================
fprintf('Loading ORL dataset...\n');
dataPath = '../../dataset/orl';
numSubjects = 40;
imagesPerSubject = 10;
imageHeight = 112;
imageWidth = 92;
numSamples = numSubjects * imagesPerSubject;

cacheFile = 'orl_images_cache.mat';
if exist(cacheFile, 'file')
    load(cacheFile, 'allImages', 'y');
else
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

% Construct views
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

fprintf('Done.\n\n');

%% ====================  Refined Search Space ====================
fprintf('Defining refined search space...\n');

% 基于实验反馈的精简参数
gamma_list = [4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0];  % γ=5.0附近密集搜索
lambda2_list = [1.5e-3, 2e-3, 2.5e-3, 3e-3, 4e-3, 5e-3, 8e-3, 1e-2];  % 2e-3~1e-2
beta_fine = [230, 235, 240, 246, 250, 255, 260, 270, 275, 280, 283, 290];  % 246/283附近
k_list = [5, 6, 7];  % k=5,7都好
lambda1_list = [8e-6, 1e-5, 1.2e-5];  % 1e-5附近微调

% 固定配置
norm_mode = 3;  % L2-Col (实验证明最优)
layers = [400, 150, 40];  % 实验证明最优

fprintf('  Gamma: %d values (4.0-8.0, focus on 5.0)\n', length(gamma_list));
fprintf('  Lambda2: %d values (1.5e-3 to 1e-2)\n', length(lambda2_list));
fprintf('  Beta: %d values (230-290, focus on 246/283)\n', length(beta_fine));
fprintf('  K: %d values (5,6,7)\n', length(k_list));
fprintf('  Lambda1: %d values (around 1e-5)\n\n', length(lambda1_list));

fprintf('=== 3-Phase Refined Strategy ===\n');
fprintf('Phase 1: Gamma × Lambda2 (核心组合, ~56 trials)\n');
fprintf('Phase 2: Beta × K (基于Phase1最佳, ~36 trials)\n');
fprintf('Phase 3: Lambda1 微调 (~3 trials)\n');
fprintf('Total: ~95 trials (高效聚焦)\n\n');

%% ====================  Initialize Results ====================
results = struct();
results.configs = {};
results.ACC = [];
results.NMI = [];
results.Purity = [];
results.time = [];
results.alpha = [];

output_csv = 'best_param_v2_results.csv';
if ~exist(output_csv, 'file')
    fid = fopen(output_csv, 'w');
    fprintf(fid, 'trial,gamma,lambda1,lambda2,beta,k,ACC,NMI,Purity,time_s,alpha1,alpha2\n');
    fclose(fid);
end

target_ACC = 87.80;
best_ACC = 0;
best_config = struct();
best_trial = 0;
trial_count = 0;

%% ====================  Phase 1: Gamma × Lambda2 ====================
fprintf('========================================\n');
fprintf('Phase 1: Gamma × Lambda2 Optimization\n');
fprintf('  Fixed: β=246, k=7, λ1=1e-5\n');
fprintf('========================================\n\n');

beta_p1 = 246;
k_p1 = 7;
lambda1_p1 = 1e-5;

phase1_results = [];

for gamma_idx = 1:length(gamma_list)
    gamma = gamma_list(gamma_idx);
    
    for lambda2_idx = 1:length(lambda2_list)
        lambda2 = lambda2_list(lambda2_idx);
        trial_count = trial_count + 1;
        
        fprintf('[%d] γ=%.2f, λ2=%.6f ... ', trial_count, gamma, lambda2);
        
        X = data_guiyi_choos(X_raw, norm_mode);
        for v = 1:numView
            X{v} = NormalizeFea(X{v}, 0);
        end
        
        options = struct();
        options.lambda1 = lambda1_p1;
        options.lambda2 = lambda2;
        options.beta = beta_p1;
        options.gamma = gamma;
        options.graph_k = k_p1;
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
            
            fprintf('ACC=%.2f%%, NMI=%.2f%%, Pur=%.2f%% (%.1fs)\n', ACC, NMI, Purity, elapsed);
            
            results.configs{end+1} = struct('gamma', gamma, 'lambda1', lambda1_p1, ...
                'lambda2', lambda2, 'beta', beta_p1, 'k', k_p1);
            results.ACC(end+1) = ACC;
            results.NMI(end+1) = NMI;
            results.Purity(end+1) = Purity;
            results.time(end+1) = elapsed;
            results.alpha{end+1} = alpha;
            
            phase1_results = [phase1_results; gamma, lambda2, ACC];
            
            fid = fopen(output_csv, 'a');
            fprintf(fid, '%d,%.2f,%.6f,%.6f,%.0f,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f\n', ...
                trial_count, gamma, lambda1_p1, lambda2, beta_p1, k_p1, ACC, NMI, Purity, elapsed, alpha(1), alpha(2));
            fclose(fid);
            
            if ACC > best_ACC
                best_ACC = ACC;
                best_config = results.configs{end};
                best_trial = trial_count;
                fprintf('  *** NEW BEST: %.2f%% ***\n', best_ACC);
            end
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
        end
    end
end

[~, idx] = max(phase1_results(:,3));
best_gamma = phase1_results(idx,1);
best_lambda2 = phase1_results(idx,2);
fprintf('\nPhase 1 Best: γ=%.2f, λ2=%.6f => ACC=%.2f%%\n\n', ...
    best_gamma, best_lambda2, phase1_results(idx,3));

%% ====================  Phase 2: Beta × K ====================
fprintf('========================================\n');
fprintf('Phase 2: Beta × K Optimization\n');
fprintf('  Using: γ=%.2f, λ2=%.6f, λ1=1e-5\n', best_gamma, best_lambda2);
fprintf('========================================\n\n');

for beta_idx = 1:length(beta_fine)
    beta = beta_fine(beta_idx);
    
    for k_idx = 1:length(k_list)
        k = k_list(k_idx);
        trial_count = trial_count + 1;
        
        fprintf('[%d] β=%.0f, k=%d ... ', trial_count, beta, k);
        
        X = data_guiyi_choos(X_raw, norm_mode);
        for v = 1:numView
            X{v} = NormalizeFea(X{v}, 0);
        end
        
        options = struct();
        options.lambda1 = lambda1_p1;
        options.lambda2 = best_lambda2;
        options.beta = beta;
        options.gamma = best_gamma;
        options.graph_k = k;
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
            
            fprintf('ACC=%.2f%%, NMI=%.2f%%, Pur=%.2f%%\n', ACC, NMI, Purity);
            
            results.configs{end+1} = struct('gamma', best_gamma, 'lambda1', lambda1_p1, ...
                'lambda2', best_lambda2, 'beta', beta, 'k', k);
            results.ACC(end+1) = ACC;
            results.NMI(end+1) = NMI;
            results.Purity(end+1) = Purity;
            results.time(end+1) = elapsed;
            results.alpha{end+1} = alpha;
            
            fid = fopen(output_csv, 'a');
            fprintf(fid, '%d,%.2f,%.6f,%.6f,%.0f,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f\n', ...
                trial_count, best_gamma, lambda1_p1, best_lambda2, beta, k, ACC, NMI, Purity, elapsed, alpha(1), alpha(2));
            fclose(fid);
            
            if ACC > best_ACC
                best_ACC = ACC;
                best_config = results.configs{end};
                best_trial = trial_count;
                fprintf('  *** NEW BEST: %.2f%% ***\n', best_ACC);
            end
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
        end
    end
end

fprintf('\n');

%% ====================  Phase 3: Lambda1 Fine-tune ====================
fprintf('========================================\n');
fprintf('Phase 3: Lambda1 Fine-Tuning\n');
fprintf('  Using best config from Phase 2\n');
fprintf('========================================\n\n');

for lambda1_idx = 1:length(lambda1_list)
    lambda1 = lambda1_list(lambda1_idx);
    trial_count = trial_count + 1;
    
    fprintf('[%d] λ1=%.6f ... ', trial_count, lambda1);
    
    X = data_guiyi_choos(X_raw, norm_mode);
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
        
        fprintf('ACC=%.2f%%, NMI=%.2f%%, Pur=%.2f%%\n', ACC, NMI, Purity);
        
        results.configs{end+1} = struct('gamma', best_config.gamma, 'lambda1', lambda1, ...
            'lambda2', best_config.lambda2, 'beta', best_config.beta, 'k', best_config.k);
        results.ACC(end+1) = ACC;
        results.NMI(end+1) = NMI;
        results.Purity(end+1) = Purity;
        results.time(end+1) = elapsed;
        results.alpha{end+1} = alpha;
        
        fid = fopen(output_csv, 'a');
        fprintf(fid, '%d,%.2f,%.6f,%.6f,%.0f,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f\n', ...
            trial_count, best_config.gamma, lambda1, best_config.lambda2, ...
            best_config.beta, best_config.k, ACC, NMI, Purity, elapsed, alpha(1), alpha(2));
        fclose(fid);
        
        if ACC > best_ACC
            best_ACC = ACC;
            best_config = results.configs{end};
            best_trial = trial_count;
            fprintf('  *** NEW BEST: %.2f%% ***\n', best_ACC);
        end
        
    catch ME
        fprintf('ERROR: %s\n', ME.message);
    end
end

%% ====================  Final Summary ====================
fprintf('\n========================================\n');
fprintf('FINAL RESULTS\n');
fprintf('========================================\n\n');

fprintf('Total Trials: %d\n', trial_count);
fprintf('Best Trial: %d\n\n', best_trial);

fprintf('=== BEST CONFIGURATION ===\n');
fprintf('Gamma:   %.2f\n', best_config.gamma);
fprintf('Lambda1: %.6f\n', best_config.lambda1);
fprintf('Lambda2: %.6f\n', best_config.lambda2);
fprintf('Beta:    %.0f\n', best_config.beta);
fprintf('K:       %d\n', best_config.k);
fprintf('Layers:  [400, 150, 40]\n');
fprintf('Norm:    L2-Col (Case 3)\n\n');

[sorted_ACC, sort_idx] = sort(results.ACC, 'descend');
fprintf('=== PERFORMANCE ===\n');
fprintf('ACC:    %.2f%% (Target: 87.80%%, Gap: %.2f%%)\n', best_ACC, target_ACC - best_ACC);
best_NMI = results.NMI(sort_idx(1));
best_Purity = results.Purity(sort_idx(1));
fprintf('NMI:    %.2f%% (Target: 93.81%%)\n', best_NMI);
fprintf('Purity: %.2f%% (Target: 89.70%%)\n\n', best_Purity);

fprintf('=== Top 5 Results ===\n');
for rank = 1:min(5, length(results.ACC))
    idx = sort_idx(rank);
    cfg = results.configs{idx};
    fprintf('%d. ACC=%.2f%%, NMI=%.2f%%, Pur=%.2f%%\n', rank, ...
        results.ACC(idx), results.NMI(idx), results.Purity(idx));
    fprintf('   γ=%.2f, λ1=%.6f, λ2=%.6f, β=%.0f, k=%d\n', ...
        cfg.gamma, cfg.lambda1, cfg.lambda2, cfg.beta, cfg.k);
end

save('best_param_v2_results.mat', 'results', 'best_config', 'best_ACC');

fprintf('\n========================================\n');
fprintf('Results saved to:\n');
fprintf('  - best_param_v2_results.csv\n');
fprintf('  - best_param_v2_results.mat\n');
fprintf('========================================\n');
