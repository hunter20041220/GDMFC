% demo_GDMFC.m
% Demonstration script for GDMFC algorithm on WebKB dataset
% GDMFC算法在WebKB数据集上的演示脚本
%
% This script demonstrates the complete pipeline:
% 本脚本演示完整流程：
%   1. Load WebKB multi-view dataset
%      加载WebKB多视图数据集
%   2. Preprocess data (normalization)
%      数据预处理（归一化）
%   3. Run GDMFC algorithm
%      运行GDMFC算法
%   4. Perform spectral clustering on learned representations
%      对学习到的表示进行谱聚类
%   5. Evaluate clustering performance (ACC, NMI, Purity)
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
% Add paths to helper functions from other directories
% 添加其他目录中辅助函数的路径
addpath(genpath('../DMF_MVC/misc'));  % for evaluation functions
addpath(genpath('../DMF_MVC/approx_seminmf'));  % for Semi-NMF

fprintf('========================================\n');
fprintf('GDMFC Demo on WebKB Dataset\n');
fprintf('========================================\n');
fprintf('Experiment Time: %s\n', experiment_info.timestamp);
fprintf('MATLAB Version: %s\n', experiment_info.matlab_version);
fprintf('Random Seed: %d\n', rng_seed);
fprintf('========================================\n\n');

%% ==================== Load Dataset 加载数据集 ====================
fprintf('Step 1: Loading WebKB dataset...\n');
dataPath = '../../dataset/WebKB.mat';
dataset_name = 'WebKB';

if ~exist(dataPath, 'file')
    error('Dataset file not found: %s\nPlease check the path.', dataPath);
end

load(dataPath);

% Dataset info: X{1} is Anchor view, X{2} is Content view
% 数据集信息：X{1}是Anchor视图，X{2}是Content视图
numView = length(X);
numSample = size(X{1}, 1);
numCluster = length(unique(y));

% Record dataset information 记录数据集详细信息
experiment_info.dataset_name = dataset_name;
experiment_info.dataset_path = dataPath;
experiment_info.num_views = numView;
experiment_info.num_samples = numSample;
experiment_info.num_clusters = numCluster;
experiment_info.num_classes = numCluster;

% Record feature dimensions for each view 记录每个视图的特征维度
experiment_info.feature_dims = zeros(numView, 1);
for v = 1:numView
    experiment_info.feature_dims(v) = size(X{v}, 2);
end

fprintf('  Dataset: %s\n', dataset_name);
fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', numSample);
fprintf('  Number of clusters: %d\n', numCluster);
fprintf('  Feature dimensions: ');
for v = 1:numView
    fprintf('%d', experiment_info.feature_dims(v));
    if v < numView, fprintf(', '); end
end
fprintf('\n\n');

%% ==================== Data Preprocessing 数据预处理 ====================
fprintf('Step 2: Preprocessing data...\n');

% Record preprocessing method 记录预处理方法
experiment_info.preprocessing_method = 'NormalizeFea_L2';
experiment_info.preprocessing_style = 'Standard';

% Normalize each view to unit length
% 将每个视图归一化为单位长度
for v = 1:numView
    X{v} = NormalizeFea(X{v}, 0);  % L2 normalization
    fprintf('  View %d: %d samples × %d features\n', v, size(X{v}, 1), size(X{v}, 2));
end
fprintf('  Data normalized (L2 norm).\n\n');

%% ==================== Algorithm Parameters 算法参数 ====================
fprintf('Step 3: Setting algorithm parameters...\n');

% Layer configuration 层配置
% For WebKB with 2 classes, we use a simple 2-layer structure
% 对于有2个类别的WebKB，使用简单的两层结构
layers = [100, 50];  % hidden layer dimensions: 100 -> 50 -> numCluster

% Record layer configuration 记录层配置
experiment_info.layers = layers;
experiment_info.full_architecture = [experiment_info.feature_dims', layers, numCluster];

% Algorithm parameters 算法参数
% 使用较小的正则化系数以保证数值稳定性
options = struct();
options.lambda1 = 0.001;    % HSIC diversity coefficient HSIC多样性系数 (降低以提高稳定性)
options.lambda2 = 0.010;    % co-orthogonal constraint coefficient 协正交约束系数 (降低)
options.beta = 115;         % graph regularization coefficient 图正则化系数 (降低)
options.gamma = 1.2;        % view weight parameter (must be > 1) 视图权重参数
options.graph_k = 7;        % number of neighbors for graph construction 图构造邻居数
options.maxIter = 100;      % maximum iterations 最大迭代次数
options.tol = 1e-5;         % convergence tolerance 收敛容差

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
fprintf('Step 4: Running GDMFC algorithm...\n');
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
fprintf('Step 5: Performing spectral clustering...\n');

% Use spectral clustering on the learned representation H
% 对学习到的表示H进行谱聚类
% H is n x k, we need to construct an n x n similarity matrix
% 构建相似度矩阵
S = H * H';  % n x n similarity matrix based on inner product
S = (S + S') / 2;  % ensure symmetry 确保对称性
S = max(S, 0);  % ensure non-negative 确保非负

predict_labels = SpectralClustering(S, numCluster);

fprintf('  Spectral clustering completed.\n\n');

%% ==================== Evaluation 评估 ====================
fprintf('Step 6: Evaluating clustering performance...\n');
fprintf('========================================\n');

% Compute evaluation metrics 计算评估指标
% ACC: Clustering Accuracy 聚类准确率
res = bestMap(y, predict_labels);
ACC = length(find(y == res)) / length(y);

% NMI: Normalized Mutual Information 归一化互信息
% 使用MutualInfo函数，它采用sqrt((MI/Hx)*(MI/Hy))归一化，更稳定
NMI = MutualInfo(y, predict_labels);

% Purity: Clustering Purity 聚类纯度
Purity = compute_purity(y, predict_labels);

% Record evaluation metrics 记录评估指标
experiment_info.ACC = ACC;
experiment_info.NMI = NMI;
experiment_info.Purity = Purity;

fprintf('Results on WebKB Dataset:\n');
fprintf('  ACC    = %.4f (%.2f%%)\n', ACC, ACC*100);
fprintf('  NMI    = %.4f (%.2f%%)\n', NMI, NMI*100);
fprintf('  Purity = %.4f (%.2f%%)\n', Purity, Purity*100);
fprintf('========================================\n\n');

%% ==================== Single Run Results ====================
% 只运行一次，不进行多次统计
% Single run only, no multiple runs for statistics
fprintf('Step 7: Single run completed.\n');

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
fprintf('Step 8: Visualizing results...\n');

% Plot objective function convergence 绘制目标函数收敛曲线
figure('Name', 'GDMFC on WebKB', 'Position', [100, 100, 1200, 400]);

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
fprintf('Step 9: Saving comprehensive results...\n');

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
results.H_final = H;
results.elapsed_time = elapsed_time;

% Algorithm parameters (also in experiment_info, but keep for backward compatibility)
results.options = options;
results.layers = layers;

% Use the save_experiment_results function
% Get script directory
script_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(script_dir);
addpath(fullfile(root_dir, 'utils'));

% Create results directory with timestamp subfolder
results_base_dir = fullfile(root_dir, 'results');
if ~exist(results_base_dir, 'dir')
    mkdir(results_base_dir);
end

% Create experiment-specific subfolder
exp_folder_name = sprintf('GDMFC_%s_%s', dataset_name, experiment_info.timestamp);
results_dir = fullfile(results_base_dir, exp_folder_name);
mkdir(results_dir);
fprintf('  Created experiment folder: %s\n', exp_folder_name);

% Generate filenames
base_filename = sprintf('GDMFC_%s', dataset_name);
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
fprintf(fid, 'GDMFC Experiment Report\n');
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
fprintf(fid, 'Feature Dimensions: ');
for v = 1:experiment_info.num_views
    fprintf(fid, '%d', experiment_info.feature_dims(v));
    if v < experiment_info.num_views, fprintf(fid, ', '); end
end
fprintf(fid, '\n');
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
arch_header = {'Layer', 'Dimension'};
arch_data = cell(length(experiment_info.full_architecture), 2);
for i = 1:length(experiment_info.full_architecture)
    if i <= experiment_info.num_views
        arch_data{i, 1} = sprintf('Input View %d', i);
    elseif i <= experiment_info.num_views + length(layers)
        arch_data{i, 1} = sprintf('Hidden Layer %d', i - experiment_info.num_views);
    else
        arch_data{i, 1} = 'Output (Cluster)';
    end
    arch_data{i, 2} = experiment_info.full_architecture(i);
end
writecell([arch_header; arch_data], excel_filepath, 'Sheet', 'Architecture');

% Sheet 4: View Weights
view_weights_header = {'View', 'Weight'};
view_weights_data = cell(numView, 2);
for v = 1:numView
    view_weights_data{v, 1} = sprintf('View %d', v);
    view_weights_data{v, 2} = alpha(v);
end
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


%% ==================== Helper Function 辅助函数 ====================
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
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
