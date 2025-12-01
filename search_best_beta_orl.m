% search_best_beta_orl_v2.m
% Sweep beta from 361 to 1000 on ORL dataset.
% Continues writing to 'search_beta_results_ORL.csv'.
% Finally, analyzes the best stable interval.

clear; clc; close all;
addpath(genpath('../DMF_MVC'));

fprintf('Search best beta on ORL (Target Range: 361:1000)\n');

% =========================================================================
% 1. Dataset Loading & Preprocessing (Original Logic)
% =========================================================================
dataPath = '../../dataset/orl';
numSubjects = 40;
imagesPerSubject = 10;
imageHeight = 112;
imageWidth = 92;
numSamples = numSubjects * imagesPerSubject;

% Load ORL images (cached)
cacheFile = 'orl_images_cache.mat';
if exist(cacheFile, 'file')
    load(cacheFile, 'allImages', 'y');
    fprintf('Loaded cached ORL images from %s\n', cacheFile);
else
    fprintf('Loading ORL images from disk...\n');
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

% Construct Views
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

X_orig = cell(1,2);
X_orig{1} = X1;
X_orig{2} = X2;

% Preprocessing
preprocess_mode = 3; 
for v = 1:length(X_orig)
    X_orig{v} = data_guiyi_choos({X_orig{v}}, preprocess_mode);
    X_orig{v} = X_orig{v}{1};
    X_orig{v} = NormalizeFea(X_orig{v}, 0);
end
X_template = X_orig;
numView = length(X_template);
numCluster = numSubjects;

% =========================================================================
% 2. Fixed Parameters
% =========================================================================
layers = [400, 150, 40];
options = struct();
options.lambda1 = 1e-5;
options.lambda2 = 1e-3;
options.gamma = 1.20;
options.graph_k = 7;
options.maxIter = 100;
options.tol = 1e-5;

% =========================================================================
% 3. Resume/Append Logic (Target: 341 to 1000) - UPDATED
% =========================================================================
target_range = 341:1000;  
outCSV = 'search_beta_results_ORL.csv';

% 1. 读取现有的 CSV
if exist(outCSV, 'file')
    try
        existing_table = readtable(outCSV);
        fprintf('Found existing CSV with %d rows.\n', height(existing_table));
        
        % 2. 清除目标范围内(341-1000)的旧数据
        % 只要 beta >= 341 的都删掉，防止脚本以为跑完了直接跳过
        rows_to_keep = existing_table.beta < 341; 
        
        % 如果发现有需要删除的旧数据
        if sum(~rows_to_keep) > 0
            fprintf('Cleaning up %d old records (beta >= 341) to force re-run...\n', sum(~rows_to_keep));
            existing_table = existing_table(rows_to_keep, :);
            writetable(existing_table, outCSV); % 把清理后的表格写回硬盘
            fprintf('CSV updated. Ready to start from 341.\n');
        end
        
        % 3. 确定已完成的任务 (现在应该只有 1-340)
        if ~isempty(existing_table)
            done_betas = existing_table.beta;
        else
            done_betas = [];
        end
    catch ME
        warning('Error reading CSV: %s. Starting fresh.', ME.message);
        done_betas = [];
    end
else
    % Create file with header if it doesn't exist
    headers = {'beta','ACC','NMI','Purity','time_s'};
    dummy = table([],[],[],[],[], 'VariableNames', headers);
    writetable(dummy, outCSV);
    done_betas = [];
end

% Determine missing betas
to_run = setdiff(target_range, done_betas);
n_run = length(to_run);

fprintf('Target range: %d-%d.\n', min(target_range), max(target_range));
fprintf('Remaining to run: %d betas.\n', n_run);


% =========================================================================
% 4. Main Sweep Loop
% =========================================================================
if n_run > 0
    % Optional: Parallel pool
    % if isempty(gcp('nocreate')), parpool('local'); end
    
    for idx = 1:n_run
        beta = to_run(idx);
        fprintf('\n[Run %d/%d] beta=%.4f ... ', idx, n_run, beta);
        
        % Reset X
        X = cell(1, numView);
        for v = 1:numView
            X{v} = X_template{v};
        end
        options.beta = beta;
        
        try
            tstart = tic;
            [H, ~, alpha, ~] = GDMFC(X, numCluster, layers, options);
            elapsed = toc(tstart);
            
            % Clustering
            H_final = H;
            S = (H_final * H_final');
            S = (S + S')/2;
            S = max(S, 0);
            predict_labels = SpectralClustering(S, numCluster);
            
            % Metrics
            res = bestMap(y, predict_labels);
            ACC = length(find(y == res)) / length(y) * 100;
            NMI = MutualInfo(y, predict_labels) * 100;
            Purity = compute_purity(y, predict_labels) * 100;
            
            fprintf('Done. ACC=%.2f%% (Time: %.1fs)\n', ACC, elapsed);
            
            % Save SINGLE ROW immediately to CSV (Append Mode)
            row_data = table(beta, ACC, NMI, Purity, elapsed, ...
                'VariableNames', {'beta','ACC','NMI','Purity','time_s'});
            
            writetable(row_data, outCSV, 'WriteMode', 'Append');
            
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            % Save error row as NaN
            row_data = table(beta, NaN, NaN, NaN, NaN, ...
                'VariableNames', {'beta','ACC','NMI','Purity','time_s'});
            writetable(row_data, outCSV, 'WriteMode', 'Append');
        end
    end
else
    fprintf('All betas in range 361-1000 are already processed.\n');
end

% =========================================================================
% 5. Final Analysis: Best Range & Plotting
% =========================================================================
fprintf('\nProcessing final results for Interval Analysis...\n');

% Load FULL data (including 1-360 if present, and new 361-1000)
full_results = readtable(outCSV);
% Sort by beta ensures plotting is correct
full_results = sortrows(full_results, 'beta'); 

% Remove NaNs
valid_idx = ~isnan(full_results.ACC);
betas_clean = full_results.beta(valid_idx);
acc_clean = full_results.ACC(valid_idx);

if isempty(betas_clean)
    error('No valid results found in CSV.');
end

% --- Find Best Single Point ---
[maxACC, maxIdx] = max(acc_clean);
bestBetaPoint = betas_clean(maxIdx);

% --- Calculate "Best Interval" (Moving Average Approach) ---
% We use a moving average to find a region of stability
windowSize = 20; % Adjust this window size (e.g., span of 20 betas)
if length(acc_clean) > windowSize
    coeff = ones(1, windowSize)/windowSize;
    smoothedACC = filter(coeff, 1, acc_clean);
    
    % Shift to align filter delay
    shift = floor(windowSize/2);
    smoothedACC = [smoothedACC(shift+1:end); zeros(shift,1)]; 
    % (Truncate end artifacts for analysis)
    valid_smooth_len = length(smoothedACC) - shift;
    
    % Find the peak of the smoothed curve
    [~, maxSmoothIdx] = max(smoothedACC(1:valid_smooth_len));
    centerBeta = betas_clean(maxSmoothIdx);
    
    % Define interval as +/- range around this center where ACC is high
    % e.g., within 98% of the peak smoothed value
    threshold = max(smoothedACC) * 0.98; % 
    high_region_indices = find(smoothedACC(1:valid_smooth_len) >= threshold);
    
    interval_start = betas_clean(min(high_region_indices));
    interval_end = betas_clean(max(high_region_indices));
else
    % Fallback if too few points
    interval_start = bestBetaPoint;
    interval_end = bestBetaPoint;
    smoothedACC = acc_clean;
end

fprintf('\n=== SUMMARY ===\n');
fprintf('Best Single Beta: %.1f (ACC: %.2f%%)\n', bestBetaPoint, maxACC);
fprintf('Recommended Stable Interval: [%.1f, %.1f]\n', interval_start, interval_end);

% --- Plotting ---
figure('Name','ACC vs Beta Analysis','Position',[100,100,1000,500]);
hold on;

% 1. Plot Recommended Interval (Green Zone)
y_limits = [min(acc_clean)*0.95, 100];
patch([interval_start, interval_end, interval_end, interval_start], ...
      [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
      [0.8, 1, 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
      'DisplayName', sprintf('Best Interval [%d, %d]', interval_start, interval_end));

% 2. Plot Raw Data
plot(betas_clean, acc_clean, 'Color', [0.7, 0.7, 0.9], 'LineWidth', 1, ...
    'DisplayName', 'Raw ACC');

% 3. Plot Smoothed Trend
if length(acc_clean) > windowSize
    plot(betas_clean(1:valid_smooth_len), smoothedACC(1:valid_smooth_len), ...
        'Color', 'b', 'LineWidth', 2, 'DisplayName', 'Smoothed Trend');
end

% 4. Mark Best Point
plot(bestBetaPoint, maxACC, 'r*', 'MarkerSize', 10, 'LineWidth', 2, ...
    'DisplayName', sprintf('Max: %.1f%% @ %.0f', maxACC, bestBetaPoint));

xlabel('Beta');
ylabel('Accuracy (%)');
title('Hyperparameter Sensitivity Analysis: Beta on ORL');
legend('Location', 'southeast');
grid on;
ylim([min(acc_clean)-2, 100.5]);
box on;

% Save outputs
saveas(gcf, 'ACC_Analysis_ORL.png');
save('final_analysis_ORL.mat', 'full_results', 'interval_start', 'interval_end', 'bestBetaPoint');
fprintf('Analysis plot saved to ACC_Analysis_ORL.png\n');