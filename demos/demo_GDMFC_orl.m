% demo_GDMFC_orl.m
% Demonstration script for GDMFC algorithm on ORL Face Dataset
% GDMFC算法在ORL人脸数据集上的演示脚本
%
% This script demonstrates the complete pipeline on ORL dataset:
% 本脚本演示在ORL数据集上的完整流程：
%   1. Load ORL face images from folder structure (40 subjects, 10 images each)
%      从文件夹结构加载ORL人脸图像（40个人，每人10张）
%   2. Construct multi-view features (Raw pixels, LBP, Gabor)
%      构造多视图特征（原始像素、LBP、Gabor）
%   3. Preprocess data (normalization)
%      数据预处理（归一化）
%   4. Run GDMFC algorithm
%      运行GDMFC算法
%   5. Perform spectral clustering on learned representations
%      对学习到的表示进行谱聚类
%   6. Evaluate clustering performance (ACC, NMI, Purity)
%      评估聚类性能
%
% Author: Generated for GDMFC research project
% Date: 2024

clear; clc; close all;

%% ==================== Experiment Metadata 实验元数据 ====================
% Record all experiment metadata for reproducibility
experiment_info = struct();
experiment_info.timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
experiment_info.date = datestr(now, 'yyyy-mm-dd');
experiment_info.matlab_version = version;
experiment_info.computer = computer;
experiment_info.user = getenv('USERNAME');

% Set random seed for reproducibility 设置随机种子保证可重复性
rng_seed = 2024;
rng(rng_seed);
experiment_info.random_seed = rng_seed;

%% ==================== Add Paths 添加路径 ====================
% Get the directory of this script
script_dir = fileparts(mfilename('fullpath'));
% Get GDMFC root directory (parent of demos/)
gdmfc_root = fileparts(script_dir);
% Add paths
addpath(genpath(gdmfc_root));

fprintf('========================================\n');
fprintf('GDMFC Demo on ORL Face Dataset\n');
fprintf('========================================\n');
fprintf('Experiment Time: %s\n', experiment_info.timestamp);
fprintf('MATLAB Version: %s\n', experiment_info.matlab_version);
fprintf('Random Seed: %d\n', rng_seed);
fprintf('========================================\n\n');

%% ==================== Load ORL Dataset 加载ORL数据集 ====================
fprintf('Step 1: Loading ORL face dataset...\n');

% Dataset path 数据集路径
% Navigate from GDMFC root to dataset folder
dataPath = fullfile(fileparts(fileparts(gdmfc_root)), 'dataset', 'orl');

% ORL dataset parameters ORL数据集参数
numSubjects = 40;      % 人数 (number of subjects)
imagesPerSubject = 10; % 每人图像数 (images per subject)
imageHeight = 112;     % 图像高度 (image height)
imageWidth = 92;       % 图像宽度 (image width)
numSamples = numSubjects * imagesPerSubject;  % 总样本数 (total samples)

% Load all images 加载所有图像
fprintf('  Loading %d images from %d subjects...\n', numSamples, numSubjects);
allImages = zeros(numSamples, imageHeight * imageWidth);
labels = zeros(numSamples, 1);

sampleIdx = 1;
for subjID = 1:numSubjects
    subjFolder = fullfile(dataPath, sprintf('s%d', subjID));

    for imgID = 1:imagesPerSubject
        imgPath = fullfile(subjFolder, sprintf('%d.pgm', imgID));

        % Read PGM image 读取PGM图像
        img = imread(imgPath);

        % Flatten to vector 展平为向量
        allImages(sampleIdx, :) = double(img(:)');
        labels(sampleIdx) = subjID;

        sampleIdx = sampleIdx + 1;
    end
end

fprintf('  Loaded %d images (size: %d×%d)\n', numSamples, imageHeight, imageWidth);
fprintf('  Number of classes: %d\n\n', numSubjects);

% Record dataset information
experiment_info.dataset_name = 'ORL_Face';
experiment_info.dataset_path = dataPath;
experiment_info.num_samples = numSamples;
experiment_info.num_clusters = numSubjects;
experiment_info.num_classes = numSubjects;
experiment_info.image_height = imageHeight;
experiment_info.image_width = imageWidth;

%% ==================== Construct Multi-View Features 构造多视图特征 ====================
fprintf('Step 2: Constructing multi-view features...\n');

% View 1: Raw pixel features (downsampled) 原始像素特征(降采样)
% 降采样以减少维度
downsampleFactor = 2;
newHeight = imageHeight / downsampleFactor;
newWidth = imageWidth / downsampleFactor;
X1 = zeros(numSamples, newHeight * newWidth);

for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    % Downsample by averaging 通过平均进行降采样
    imgDown = imresize(img, [newHeight, newWidth], 'bilinear');
    X1(i, :) = imgDown(:)';
end

% View 2: LBP (Local Binary Pattern) features LBP局部二值模式特征
% 简化版LBP: 将图像分块,计算每块的统计特征
blockSize = 8;
numBlocksH = floor(imageHeight / blockSize);
numBlocksW = floor(imageWidth / blockSize);
lbpFeatDim = numBlocksH * numBlocksW * 4;  % 每块4个统计量(均值,方差,最小值,最大值)
X2 = zeros(numSamples, lbpFeatDim);

for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    featIdx = 1;

    for bh = 1:numBlocksH
        for bw = 1:numBlocksW
            % Extract block 提取块
            rowStart = (bh-1) * blockSize + 1;
            rowEnd = min(bh * blockSize, imageHeight);
            colStart = (bw-1) * blockSize + 1;
            colEnd = min(bw * blockSize, imageWidth);

            block = img(rowStart:rowEnd, colStart:colEnd);

            % Compute statistics 计算统计量
            X2(i, featIdx) = mean(block(:));      % 均值
            X2(i, featIdx+1) = std(block(:));     % 标准差
            X2(i, featIdx+2) = min(block(:));     % 最小值
            X2(i, featIdx+3) = max(block(:));     % 最大值

            featIdx = featIdx + 4;
        end
    end
end

% View 3: Gabor features Gabor特征
% 使用简化的Gabor特征: 分块后的梯度统计量
X3 = zeros(numSamples, lbpFeatDim);

for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    % Compute gradients 计算梯度
    [Gx, Gy] = gradient(img);
    Gmag = sqrt(Gx.^2 + Gy.^2);  % 梯度幅值

    featIdx = 1;
    for bh = 1:numBlocksH
        for bw = 1:numBlocksW
            % Extract block from gradient magnitude
            rowStart = (bh-1) * blockSize + 1;
            rowEnd = min(bh * blockSize, imageHeight);
            colStart = (bw-1) * blockSize + 1;
            colEnd = min(bw * blockSize, imageWidth);

            block = Gmag(rowStart:rowEnd, colStart:colEnd);

            % Compute statistics 计算统计量
            X3(i, featIdx) = mean(block(:));      % 均值
            X3(i, featIdx+1) = std(block(:));     % 标准差
            X3(i, featIdx+2) = min(block(:));     % 最小值
            X3(i, featIdx+3) = max(block(:));     % 最大值

            featIdx = featIdx + 4;
        end
    end
end

% Organize into cell array 组织为cell数组
X = cell(1, 3);
X{1} = X1;
X{2} = X2;
X{3} = X3;
y = labels;

numView = length(X);
numCluster = numSubjects;

% Record feature dimensions for each view 记录每个视图的特征维度
experiment_info.num_views = numView;
experiment_info.feature_dims = zeros(numView, 1);
for v = 1:numView
    experiment_info.feature_dims(v) = size(X{v}, 2);
end

fprintf('  View 1 (Downsampled pixels): %d dimensions\n', size(X{1}, 2));
fprintf('  View 2 (LBP block statistics): %d dimensions\n', size(X{2}, 2));
fprintf('  View 3 (Gabor features): %d dimensions\n', size(X{3}, 2));
fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', numSamples);
fprintf('  Number of clusters: %d\n\n', numCluster);

%% ==================== Data Preprocessing 数据预处理 ====================
fprintf('Step 3: Preprocessing data...\n');

% Record preprocessing method 记录预处理方法
experiment_info.preprocessing_method = 'data_guiyi_choos_mode3_NormalizeFea';
experiment_info.preprocessing_style = 'Standard';

% Preprocessing: run user preprocessing function first, then normalize
% Choose preprocessing mode for data_guiyi_choos:
% 1 - min-max, 2 - min-max (transposed), 3 - column-wise L2, 4 - column sum, 5 - global
preprocess_mode = 3; % default: column-wise L2 normalization
X = data_guiyi_choos(X, preprocess_mode);

% Then apply sample-feature normalization (NormalizeFea keeps behavior consistent)
for v = 1:numView
    X{v} = NormalizeFea(X{v}, 0);  % L2 normalization (sample-wise)
    fprintf('  View %d: %d samples × %d features\n', v, size(X{v}, 1), size(X{v}, 2));
end

fprintf('  Data preprocessed (mode %d) and normalized (L2 norm).\n\n', preprocess_mode);

%% ==================== Algorithm Parameters 算法参数 ====================
fprintf('Step 4: Setting algorithm parameters...\n');

% Layer configuration 层配置
% 对于40类,使用合适的层结构
layers = [200, 100];  % hidden layers: 200 -> 100 -> 40

% Record layer configuration 记录层配置
experiment_info.layers = layers;
experiment_info.full_architecture = [experiment_info.feature_dims', layers, numCluster];

% Algorithm parameters 算法参数
% 对于更复杂的数据集,使用更小的正则化系数
options = struct();
options.lambda1 = 0.1;      % HSIC diversity coefficient
options.lambda2 = 0;        % co-orthogonal constraint coefficient
options.beta = 115;         % graph regularization coefficient
options.gamma = 5;          % view weight parameter (must be > 1)
options.graph_k = 11;       % number of neighbors for graph construction
options.maxIter = 100;      % maximum iterations
options.tol = 1e-5;         % convergence tolerance

% Record all parameters to experiment_info 记录所有参数到实验信息
experiment_info.options = options;

fprintf('  Layer structure: [');
for i = 1:length(layers)
    fprintf('%d', layers(i));
    if i < length(layers)
        fprintf(', ');
    end
end
fprintf(', %d]\n', numCluster);
fprintf('  lambda1 (diversity): %.6f\n', options.lambda1);
fprintf('  lambda2 (orthogonal): %.6f\n', options.lambda2);
fprintf('  beta (graph reg): %.4f\n', options.beta);
fprintf('  gamma (view weight): %.2f\n', options.gamma);
fprintf('  graph_k: %d\n', options.graph_k);
fprintf('  maxIter: %d, tol: %.0e\n\n', options.maxIter, options.tol);

%% ==================== Run GDMFC Algorithm 运行GDMFC算法 ====================
fprintf('Step 5: Running GDMFC algorithm...\n');
fprintf('----------------------------------------\n');

tic;
[H, Z, alpha, obj_values] = GDMFC(X, numCluster, layers, options);
elapsed_time = toc;

% Record training results 记录训练结果
experiment_info.elapsed_time = elapsed_time;
experiment_info.view_weights = alpha;
experiment_info.num_iterations = length(obj_values);
experiment_info.obj_values = obj_values;
experiment_info.converged = (length(obj_values) < options.maxIter);
experiment_info.final_obj_value = obj_values(end);

fprintf('----------------------------------------\n');
fprintf('  Time elapsed: %.2f seconds\n', elapsed_time);
fprintf('  Iterations: %d / %d\n', length(obj_values), options.maxIter);
fprintf('  Converged: %s\n', iif(experiment_info.converged, 'Yes', 'No (max iter)'));
fprintf('  Final objective: %.6e\n', obj_values(end));
fprintf('  Final view weights: [');
for v = 1:numView
    fprintf('%.4f', alpha(v));
    if v < numView
        fprintf(', ');
    end
end
fprintf(']\n\n');

%% ==================== Clustering 聚类 ====================
fprintf('Step 6: Performing spectral clustering...\n');

% GDMFC returns the weighted average of final layer representations
% H已经是加权平均后的最终表示矩阵 (n × numCluster)
H_final = H;

% Construct similarity matrix for spectral clustering 构造相似度矩阵
S = H_final * H_final';
S = (S + S') / 2;  % 确保对称
S = max(S, 0);     % 确保非负

% Perform spectral clustering 执行谱聚类
predict_labels = SpectralClustering(S, numCluster);

fprintf('  Spectral clustering completed.\n\n');

%% ==================== Evaluation 评估 ====================
fprintf('Step 7: Evaluating clustering performance...\n');
fprintf('========================================\n');

% Compute evaluation metrics 计算评估指标
% ACC: Clustering Accuracy 聚类准确率
res = bestMap(y, predict_labels);
ACC = length(find(y == res)) / length(y);

% NMI: Normalized Mutual Information 归一化互信息
NMI = MutualInfo(y, predict_labels);

% Purity: Clustering Purity 聚类纯度
Purity = compute_purity(y, predict_labels);

% Record evaluation metrics 记录评估指标
experiment_info.ACC = ACC;
experiment_info.NMI = NMI;
experiment_info.Purity = Purity;

fprintf('Results on ORL Face Dataset:\n');
fprintf('  ACC    = %.4f (%.2f%%)\n', ACC, ACC*100);
fprintf('  NMI    = %.4f (%.2f%%)\n', NMI, NMI*100);
fprintf('  Purity = %.4f (%.2f%%)\n', Purity, Purity*100);
fprintf('========================================\n\n');

%% ==================== Single Run Results ====================
% 只运行一次，不进行多次统计
% Single run only, no multiple runs for statistics
fprintf('Step 8: Single run completed.\n');

% Use single run results
ACC_runs = ACC;
NMI_runs = NMI;
Purity_runs = Purity;

% Statistics (single run, no std)
ACC_mean = ACC_runs;
ACC_std = 0;
NMI_mean = NMI_runs;
NMI_std = 0;
Purity_mean = Purity_runs;
Purity_std = 0;

fprintf('\n========================================\n');
fprintf('Final Results (single run):\n');
fprintf('  ACC    = %.4f (%.2f%%)\n', ACC_mean, ACC_mean*100);
fprintf('  NMI    = %.4f (%.2f%%)\n', NMI_mean, NMI_mean*100);
fprintf('  Purity = %.4f (%.2f%%)\n', Purity_mean, Purity_mean*100);
fprintf('========================================\n\n');

%% ==================== Visualization 可视化 ====================
fprintf('Step 9: Visualizing results...\n');

% Plot objective function convergence 绘制目标函数收敛曲线
figure('Name', 'GDMFC on ORL Dataset', 'Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
plot(1:length(obj_values), obj_values, 'b-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 12);
ylabel('Objective Value', 'FontSize', 12);
title('Objective Function Convergence', 'FontSize', 14);
grid on;

subplot(1, 3, 2);
bar(alpha);
xlabel('View Index', 'FontSize', 12);
ylabel('Weight', 'FontSize', 12);
title('Learned View Weights', 'FontSize', 14);
set(gca, 'XTick', 1:numView);
set(gca, 'XTickLabel', {'Pixel', 'LBP', 'Gabor'});
grid on;

% Plot 3: Performance comparison 性能对比
subplot(1, 3, 3);
metrics = [ACC_mean, NMI_mean, Purity_mean];
bar(metrics);
ylabel('Score', 'FontSize', 12);
title('Clustering Performance', 'FontSize', 14);
set(gca, 'XTickLabel', {'ACC', 'NMI', 'Purity'});
ylim([0, 1]);
grid on;

fprintf('  Visualization completed.\n\n');

%% ==================== Save Results 保存结果 ====================
fprintf('Step 10: Saving comprehensive results...\n');

% Create results structure with ALL information 创建包含所有信息的结果结构体
results = struct();

% Experiment metadata 实验元数据
results.experiment_info = experiment_info;

% Performance metrics 性能指标
results.ACC_mean = ACC_mean;
results.ACC_std = ACC_std;
results.NMI_mean = NMI_mean;
results.NMI_std = NMI_std;
results.Purity_mean = Purity_mean;
results.Purity_std = Purity_std;
results.ACC_runs = ACC_runs;
results.NMI_runs = NMI_runs;
results.Purity_runs = Purity_runs;

% Algorithm outputs 算法输出
results.alpha = alpha;
results.obj_values = obj_values;
results.predict_labels = predict_labels;
results.true_labels = y;
results.H_final = H_final;
results.elapsed_time = elapsed_time;

% Algorithm parameters (also in experiment_info, but keep for backward compatibility)
results.options = options;
results.layers = layers;

% Create results directory with timestamp subfolder
results_base_dir = fullfile(gdmfc_root, 'results');
if ~exist(results_base_dir, 'dir')
    mkdir(results_base_dir);
end

% Create experiment-specific subfolder
exp_folder_name = sprintf('GDMFC_ORL_%s', experiment_info.timestamp);
results_dir = fullfile(results_base_dir, exp_folder_name);
mkdir(results_dir);
fprintf('  Created experiment folder: %s\n', exp_folder_name);

% Generate filenames
base_filename = 'GDMFC_ORL';
mat_filename = [base_filename, '.mat'];
txt_filename = [base_filename, '.txt'];
excel_filename = [base_filename, '.xlsx'];
fig_filename = [base_filename, '.png'];

% Save .mat file
mat_filepath = fullfile(results_dir, mat_filename);
save(mat_filepath, 'results');
fprintf('  [1/4] MAT file saved: %s\n', mat_filename);

% Save text report
txt_filepath = fullfile(results_dir, txt_filename);
fid = fopen(txt_filepath, 'w');
fprintf(fid, '========================================\n');
fprintf(fid, 'GDMFC Experiment Report - ORL Face Dataset\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, '=== EXPERIMENT METADATA ===\n');
fprintf(fid, 'Timestamp: %s\n', experiment_info.timestamp);
fprintf(fid, 'MATLAB Version: %s\n', experiment_info.matlab_version);
fprintf(fid, 'Computer: %s\n', experiment_info.computer);
fprintf(fid, 'Random Seed: %d\n', experiment_info.random_seed);
fprintf(fid, '\n');

fprintf(fid, '=== DATASET INFORMATION ===\n');
fprintf(fid, 'Dataset: %s\n', experiment_info.dataset_name);
fprintf(fid, 'Number of Views: %d\n', experiment_info.num_views);
fprintf(fid, 'Number of Samples: %d\n', experiment_info.num_samples);
fprintf(fid, 'Number of Clusters: %d\n', experiment_info.num_clusters);
fprintf(fid, 'Image Size: %dx%d pixels\n', experiment_info.image_height, experiment_info.image_width);
fprintf(fid, 'Feature Dimensions: ');
for v = 1:experiment_info.num_views
    fprintf(fid, '%d', experiment_info.feature_dims(v));
    if v < experiment_info.num_views, fprintf(fid, ', '); end
end
fprintf(fid, '\n');
fprintf(fid, 'View Descriptions:\n');
fprintf(fid, '  View 1: Downsampled pixel features\n');
fprintf(fid, '  View 2: LBP block statistics\n');
fprintf(fid, '  View 3: Gabor features\n');
fprintf(fid, 'Preprocessing: %s (%s style)\n', experiment_info.preprocessing_method, experiment_info.preprocessing_style);
fprintf(fid, '\n');

fprintf(fid, '=== ALGORITHM PARAMETERS ===\n');
fprintf(fid, 'Layer Structure: ');
for i = 1:length(layers)
    fprintf(fid, '%d -> ', layers(i));
end
fprintf(fid, '%d\n', numCluster);
fprintf(fid, 'lambda1 (diversity): %.6f\n', options.lambda1);
fprintf(fid, 'lambda2 (orthogonal): %.6f\n', options.lambda2);
fprintf(fid, 'beta (graph regularization): %.6f\n', options.beta);
fprintf(fid, 'gamma (view weight): %.6f\n', options.gamma);
fprintf(fid, 'graph_k (neighbors): %d\n', options.graph_k);
fprintf(fid, 'maxIter: %d\n', options.maxIter);
fprintf(fid, 'tolerance: %.0e\n', options.tol);
fprintf(fid, '\n');

fprintf(fid, '=== TRAINING RESULTS ===\n');
fprintf(fid, 'Elapsed Time: %.4f seconds\n', elapsed_time);
fprintf(fid, 'Iterations: %d / %d\n', experiment_info.num_iterations, options.maxIter);
fprintf(fid, 'Converged: %s\n', iif(experiment_info.converged, 'Yes', 'No'));
fprintf(fid, 'Final Objective: %.6e\n', experiment_info.final_obj_value);
fprintf(fid, 'View Weights: ');
for v = 1:numView
    fprintf(fid, '%.6f', alpha(v));
    if v < numView, fprintf(fid, ', '); end
end
fprintf(fid, '\n\n');

fprintf(fid, '=== CLUSTERING PERFORMANCE ===\n');
fprintf(fid, 'ACC:    %.6f (%.2f%%)\n', ACC_mean, ACC_mean*100);
fprintf(fid, 'NMI:    %.6f (%.2f%%)\n', NMI_mean, NMI_mean*100);
fprintf(fid, 'Purity: %.6f (%.2f%%)\n', Purity_mean, Purity_mean*100);
fprintf(fid, '\n');

fprintf(fid, '========================================\n');
fprintf(fid, 'End of Report\n');
fprintf(fid, '========================================\n');
fclose(fid);
fprintf('  [2/4] Text report saved: %s\n', txt_filename);

% Save Excel file
excel_filepath = fullfile(results_dir, excel_filename);

% Sheet 1: Summary
summary_data = {
    'Metric', 'Value';
    '=== EXPERIMENT INFO ===', '';
    'Timestamp', experiment_info.timestamp;
    'Dataset', experiment_info.dataset_name;
    'MATLAB Version', experiment_info.matlab_version;
    'Random Seed', experiment_info.random_seed;
    '', '';
    '=== DATASET INFO ===', '';
    'Number of Views', experiment_info.num_views;
    'Number of Samples', experiment_info.num_samples;
    'Number of Clusters', experiment_info.num_clusters;
    'Image Height', experiment_info.image_height;
    'Image Width', experiment_info.image_width;
    'View 1 Dimensions', experiment_info.feature_dims(1);
    'View 2 Dimensions', experiment_info.feature_dims(2);
    'View 3 Dimensions', experiment_info.feature_dims(3);
    'Preprocessing Method', experiment_info.preprocessing_method;
    '', '';
    '=== PERFORMANCE ===', '';
    'ACC', ACC_mean;
    'ACC (%)', ACC_mean * 100;
    'NMI', NMI_mean;
    'NMI (%)', NMI_mean * 100;
    'Purity', Purity_mean;
    'Purity (%)', Purity_mean * 100;
    '', '';
    '=== TRAINING ===', '';
    'Elapsed Time (s)', elapsed_time;
    'Iterations', experiment_info.num_iterations;
    'Max Iterations', options.maxIter;
    'Converged', iif(experiment_info.converged, 'Yes', 'No');
    'Final Objective', experiment_info.final_obj_value;
};
writecell(summary_data, excel_filepath, 'Sheet', 'Summary');

% Sheet 2: Parameters
param_data = {
    'Parameter', 'Value', 'Description';
    'lambda1', options.lambda1, 'HSIC diversity coefficient';
    'lambda2', options.lambda2, 'Co-orthogonal constraint coefficient';
    'beta', options.beta, 'Graph regularization coefficient';
    'gamma', options.gamma, 'View weight parameter';
    'graph_k', options.graph_k, 'Number of neighbors for graph';
    'maxIter', options.maxIter, 'Maximum iterations';
    'tol', options.tol, 'Convergence tolerance';
};
writecell(param_data, excel_filepath, 'Sheet', 'Parameters');

% Sheet 3: Architecture
arch_header = {'Layer', 'Dimension', 'Description'};
arch_data = cell(length(experiment_info.full_architecture), 3);
for i = 1:length(experiment_info.full_architecture)
    if i <= experiment_info.num_views
        arch_data{i, 1} = sprintf('Input View %d', i);
        if i == 1
            arch_data{i, 3} = 'Downsampled pixel features';
        elseif i == 2
            arch_data{i, 3} = 'LBP block statistics';
        else
            arch_data{i, 3} = 'Gabor features';
        end
    elseif i <= experiment_info.num_views + length(layers)
        arch_data{i, 1} = sprintf('Hidden Layer %d', i - experiment_info.num_views);
        arch_data{i, 3} = 'Deep feature representation';
    else
        arch_data{i, 1} = 'Output (Cluster)';
        arch_data{i, 3} = 'Final clustering representation';
    end
    arch_data{i, 2} = experiment_info.full_architecture(i);
end
writecell([arch_header; arch_data], excel_filepath, 'Sheet', 'Architecture');

% Sheet 4: View Weights
view_weights_header = {'View', 'Weight', 'Description'};
view_weights_data = {
    'View 1', alpha(1), 'Downsampled pixel features';
    'View 2', alpha(2), 'LBP block statistics';
    'View 3', alpha(3), 'Gabor features';
};
writecell([view_weights_header; view_weights_data], excel_filepath, 'Sheet', 'ViewWeights');

% Sheet 5: Convergence
conv_header = {'Iteration', 'Objective Value'};
conv_data = cell(length(obj_values), 2);
for i = 1:length(obj_values)
    conv_data{i, 1} = i;
    conv_data{i, 2} = obj_values(i);
end
writecell([conv_header; conv_data], excel_filepath, 'Sheet', 'Convergence');

fprintf('  [3/4] Excel file saved: %s\n', excel_filename);

% Save figure
fig_filepath = fullfile(results_dir, fig_filename);
saveas(gcf, fig_filepath);
fprintf('  [4/4] Figure saved: %s\n', fig_filename);

fprintf('\n');
fprintf('========================================\n');
fprintf('All results saved to subfolder:\n');
fprintf('  %s\n', exp_folder_name);
fprintf('\nFiles in this experiment folder:\n');
fprintf('  [1] %s  (MATLAB data)\n', mat_filename);
fprintf('  [2] %s  (Text report)\n', txt_filename);
fprintf('  [3] %s  (Excel tables)\n', excel_filename);
fprintf('  [4] %s  (Figure)\n', fig_filename);
fprintf('\nFull path: %s\n', results_dir);
fprintf('========================================\n\n');

fprintf('========================================\n');
fprintf('GDMFC Demo Completed Successfully!\n');
fprintf('========================================\n');


%% ==================== Helper Functions 辅助函数 ====================
function purity = compute_purity(y_true, y_pred)
% Compute clustering purity 计算聚类纯度
%
% Input:
%   y_true: ground truth labels 真实标签
%   y_pred: predicted cluster labels 预测的聚类标签
%
% Output:
%   purity: purity score 纯度分数

n = length(y_true);
clusters = unique(y_pred);
purity_sum = 0;

for i = 1:length(clusters)
    % Find samples in this cluster 找到该聚类中的样本
    cluster_idx = (y_pred == clusters(i));
    cluster_labels = y_true(cluster_idx);

    % Find the most frequent true label in this cluster
    % 找到该聚类中最频繁的真实标签
    if ~isempty(cluster_labels)
        label_counts = histc(cluster_labels, unique(y_true));
        purity_sum = purity_sum + max(label_counts);
    end
end

purity = purity_sum / n;
end

function result = iif(condition, true_val, false_val)
% Inline if function 内联if函数
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
