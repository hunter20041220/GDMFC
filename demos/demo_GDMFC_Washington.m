%==========================================================================
% GDMFC Demo on Washington WebKB Dataset
%==========================================================================
% This script demonstrates the complete pipeline for GDMFC algorithm on
% Washington WebKB dataset with comprehensive experiment tracking
% 此脚本演示GDMFC算法在Washington WebKB数据集上的完整流程，带完整实验记录
%
% Washington数据集: 230个文档, 5类 (project, course, student, faculty, staff)
% Washington dataset: 230 documents, 5 classes
% 4个视图: content (文档-词), inbound/outbound/cites (链接)
% 4 views: content (doc-term), inbound/outbound/cites (links)
%
% This script follows top-tier experimental standards:
% - Complete metadata tracking (timestamp, MATLAB version, random seed)
% - Comprehensive parameter recording
% - Detailed results saving (MAT, TXT, XLSX, PNG)
% - Organized output structure
%==========================================================================

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
addpath(genpath('../DMF_MVC'));  % for evaluation and preprocessing functions

fprintf('========================================\n');
fprintf('GDMFC Demo on Washington WebKB Dataset\n');
fprintf('========================================\n');
fprintf('Experiment Time: %s\n', experiment_info.timestamp);
fprintf('MATLAB Version: %s\n', experiment_info.matlab_version);
fprintf('Random Seed: %d\n', rng_seed);
fprintf('========================================\n\n');

%% ==================== Load Washington Dataset 加载Washington数据集 ====================
fprintf('Step 1: Loading Washington WebKB dataset...\n');

% Dataset path 数据集路径
dataPath = '../../dataset/Washington';
dataset_name = 'Washington';

% 读取标签
labelsFile = fullfile(dataPath, 'washington_act.txt');
if ~exist(labelsFile, 'file')
    error('Labels file not found: %s\nPlease check the path.', labelsFile);
end
fid = fopen(labelsFile, 'r');
labels = fscanf(fid, '%d');
fclose(fid);

numSamples = length(labels);
numCluster = length(unique(labels));

% 读取类别名称
labelNamesFile = fullfile(dataPath, 'labels.txt');
fid = fopen(labelNamesFile, 'r');
labelNames = textscan(fid, '%s');
labelNames = labelNames{1};
fclose(fid);

fprintf('  Dataset: %s\n', dataset_name);
fprintf('  Number of documents: %d\n', numSamples);
fprintf('  Number of classes: %d\n', numCluster);
fprintf('  Classes: %s\n', strjoin(labelNames, ', '));

%% ==================== Load Multi-View Features 加载多视图特征 ====================
fprintf('\nStep 2: Loading multi-view features...\n');

% View 1: Content (文档-词矩阵)
contentFile = fullfile(dataPath, 'washington_content.mtx');
fprintf('  Loading content view from %s...\n', contentFile);
X1 = read_matrix_market(contentFile);
fprintf('    Content view: %d x %d (documents x words)\n', size(X1, 1), size(X1, 2));

% View 2: Inbound links (入链)
inboundFile = fullfile(dataPath, 'washington_inbound.mtx');
fprintf('  Loading inbound view from %s...\n', inboundFile);
X2 = read_matrix_market(inboundFile);
fprintf('    Inbound view: %d x %d\n', size(X2, 1), size(X2, 2));

% View 3: Outbound links (出链)
outboundFile = fullfile(dataPath, 'washington_outbound.mtx');
fprintf('  Loading outbound view from %s...\n', outboundFile);
X3 = read_matrix_market(outboundFile);
fprintf('    Outbound view: %d x %d\n', size(X3, 1), size(X3, 2));

% View 4: Cites (引用链接)
citesFile = fullfile(dataPath, 'washington_cites.mtx');
fprintf('  Loading cites view from %s...\n', citesFile);
X4 = read_matrix_market(citesFile);
fprintf('    Cites view: %d x %d\n', size(X4, 1), size(X4, 2));

% 组织为cell数组
X = cell(1, 4);
X{1} = X1;  % content
X{2} = X2;  % inbound
X{3} = X3;  % outbound
X{4} = X4;  % cites

numView = length(X);
y = labels;

% Record dataset information 记录数据集详细信息
experiment_info.dataset_name = dataset_name;
experiment_info.dataset_path = dataPath;
experiment_info.num_views = numView;
experiment_info.num_samples = numSamples;
experiment_info.num_clusters = numCluster;
experiment_info.num_classes = numCluster;
experiment_info.class_names = labelNames;
experiment_info.view_names = {'Content', 'Inbound', 'Outbound', 'Cites'};

% Record feature dimensions for each view 记录每个视图的特征维度
experiment_info.feature_dims = zeros(numView, 1);
for v = 1:numView
    experiment_info.feature_dims(v) = size(X{v}, 2);
end

fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', numSamples);
fprintf('  Number of clusters: %d\n', numCluster);
fprintf('  Feature dimensions: ');
for v = 1:numView
    fprintf('%d', experiment_info.feature_dims(v));
    if v < numView, fprintf(', '); end
end
fprintf('\n\n');

%% ==================== Data Preprocessing 数据预处理 ====================
fprintf('Step 3: Preprocessing data...\n');

% Record preprocessing method 记录预处理方法
% Two-step preprocessing: first data_guiyi_choos (mode 3), then NormalizeFea
experiment_info.preprocessing_step1 = 'data_guiyi_choos mode 3 (column-wise L2)';
experiment_info.preprocessing_step2 = 'NormalizeFea L2 (sample-wise)';
experiment_info.preprocessing_mode = 3;

% Step 1: Preprocessing with data_guiyi_choos
% Choose preprocessing mode for data_guiyi_choos:
% 1 - min-max, 2 - min-max (transposed), 3 - column-wise L2, 4 - column sum, 5 - global
preprocess_mode = 3; % default: column-wise L2 normalization
X = data_guiyi_choos(X, preprocess_mode);

% Step 2: Then apply sample-feature normalization (NormalizeFea)
for v = 1:numView
    X{v} = NormalizeFea(X{v}, 0);  % L2 normalization (sample-wise)
    fprintf('  View %d: %d samples x %d features (preprocessed)\n', v, size(X{v}, 1), size(X{v}, 2));
end

fprintf('  Data preprocessed (mode %d) and normalized (L2 norm).\n\n', preprocess_mode);

%% ==================== Algorithm Parameters 算法参数 ====================
fprintf('Step 4: Setting algorithm parameters...\n');

% Layer configuration 层配置
% 对于5类，使用适当的层结构
layers = [100, 50, 20];  % hidden layers: 100 -> 50 -> 20 -> 5

% Record layer configuration 记录层配置
experiment_info.layers = layers;
experiment_info.full_architecture = [experiment_info.feature_dims', layers, numCluster];

% Algorithm parameters 算法参数
options = struct();
options.lambda1 = 0.01;      % HSIC diversity coefficient
options.lambda2 = 0.001;     % co-orthogonal constraint coefficient
options.beta = 200;          % graph regularization coefficient
options.gamma = 1.2;         % view weight parameter (must be > 1)
options.graph_k = 7;         % number of neighbors for graph construction
options.maxIter = 100;       % maximum iterations
options.tol = 1e-5;          % convergence tolerance

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
fprintf('  lambda1 (HSIC diversity): %.6f\n', options.lambda1);
fprintf('  lambda2 (co-orthogonal): %.6f\n', options.lambda2);
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
fprintf(']\n');
fprintf('  View names: %s\n\n', strjoin(experiment_info.view_names, ', '));

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

fprintf('Results on Washington WebKB Dataset:\n');
fprintf('  ACC    = %.4f (%.2f%%)\n', ACC, ACC*100);
fprintf('  NMI    = %.4f (%.2f%%)\n', NMI, NMI*100);
fprintf('  Purity = %.4f (%.2f%%)\n', Purity, Purity*100);
fprintf('========================================\n\n');

%% ==================== Visualization 可视化 ====================
fprintf('Step 8: Visualizing results...\n');

% Create comprehensive visualization
figure('Name', 'GDMFC on Washington Dataset', 'Position', [100, 100, 1200, 400]);

% Plot 1: Objective function convergence 目标函数收敛曲线
subplot(1, 3, 1);
plot(1:length(obj_values), obj_values, 'b-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 12);
ylabel('Objective Value', 'FontSize', 12);
title('Objective Function Convergence', 'FontSize', 14);
grid on;

% Plot 2: View weights 视图权重
subplot(1, 3, 2);
bar(alpha);
xlabel('View Index', 'FontSize', 12);
ylabel('Weight', 'FontSize', 12);
title('Learned View Weights', 'FontSize', 14);
set(gca, 'XTick', 1:numView);
set(gca, 'XTickLabel', experiment_info.view_names);
xtickangle(45);
grid on;

% Plot 3: Performance metrics 性能指标
subplot(1, 3, 3);
metrics = [ACC, NMI, Purity];
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
results.ACC = ACC;
results.NMI = NMI;
results.Purity = Purity;

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

% Get script directory
script_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(script_dir);

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
fprintf(fid, 'Washington WebKB Dataset\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, '=== EXPERIMENT METADATA ===\n');
fprintf(fid, 'Timestamp: %s\n', experiment_info.timestamp);
fprintf(fid, 'MATLAB Version: %s\n', experiment_info.matlab_version);
fprintf(fid, 'Computer: %s\n', experiment_info.computer);
fprintf(fid, 'User: %s\n', experiment_info.user);
fprintf(fid, 'Random Seed: %d\n', experiment_info.random_seed);
fprintf(fid, '\n');

fprintf(fid, '=== DATASET INFORMATION ===\n');
fprintf(fid, 'Dataset: %s\n', experiment_info.dataset_name);
fprintf(fid, 'Number of Views: %d\n', experiment_info.num_views);
fprintf(fid, 'View Names: %s\n', strjoin(experiment_info.view_names, ', '));
fprintf(fid, 'Number of Samples: %d\n', experiment_info.num_samples);
fprintf(fid, 'Number of Clusters: %d\n', experiment_info.num_clusters);
fprintf(fid, 'Class Names: %s\n', strjoin(experiment_info.class_names, ', '));
fprintf(fid, 'Feature Dimensions: ');
for v = 1:experiment_info.num_views
    fprintf(fid, '%d', experiment_info.feature_dims(v));
    if v < experiment_info.num_views, fprintf(fid, ', '); end
end
fprintf(fid, '\n');
fprintf(fid, '\n');

fprintf(fid, '=== PREPROCESSING ===\n');
fprintf(fid, 'Step 1: %s\n', experiment_info.preprocessing_step1);
fprintf(fid, 'Step 2: %s\n', experiment_info.preprocessing_step2);
fprintf(fid, 'Preprocessing Mode: %d\n', experiment_info.preprocessing_mode);
fprintf(fid, '\n');

fprintf(fid, '=== ALGORITHM PARAMETERS ===\n');
fprintf(fid, 'Layer Structure: ');
for i = 1:length(layers)
    fprintf(fid, '%d -> ', layers(i));
end
fprintf(fid, '%d\n', numCluster);
fprintf(fid, 'lambda1 (HSIC diversity): %.6f\n', options.lambda1);
fprintf(fid, 'lambda2 (co-orthogonal): %.6f\n', options.lambda2);
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
fprintf(fid, 'View Weights:\n');
for v = 1:numView
    fprintf(fid, '  %s: %.6f\n', experiment_info.view_names{v}, alpha(v));
end
fprintf(fid, '\n');

fprintf(fid, '=== CLUSTERING PERFORMANCE ===\n');
fprintf(fid, 'ACC:    %.6f (%.2f%%)\n', ACC, ACC*100);
fprintf(fid, 'NMI:    %.6f (%.2f%%)\n', NMI, NMI*100);
fprintf(fid, 'Purity: %.6f (%.2f%%)\n', Purity, Purity*100);
fprintf(fid, '\n');

fprintf(fid, '========================================\n');
fprintf(fid, 'End of Report\n');
fprintf(fid, '========================================\n');
fclose(fid);
fprintf('  [2/4] Text report saved: %s\n', txt_filename);

% Save Excel file with detailed results 保存Excel文件
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
    'View Names', strjoin(experiment_info.view_names, ', ');
    'Class Names', strjoin(experiment_info.class_names, ', ');
    'Preprocessing Mode', experiment_info.preprocessing_mode;
    '', '';
    '=== PERFORMANCE ===', '';
    'ACC', ACC;
    'ACC (%)', ACC * 100;
    'NMI', NMI;
    'NMI (%)', NMI * 100;
    'Purity', Purity;
    'Purity (%)', Purity * 100;
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
        arch_data{i, 3} = experiment_info.view_names{i};
    elseif i <= experiment_info.num_views + length(layers)
        layer_idx = i - experiment_info.num_views;
        arch_data{i, 1} = sprintf('Hidden Layer %d', layer_idx);
        arch_data{i, 3} = sprintf('Hidden representation layer %d', layer_idx);
    else
        arch_data{i, 1} = 'Output (Cluster)';
        arch_data{i, 3} = 'Final clustering layer';
    end
    arch_data{i, 2} = experiment_info.full_architecture(i);
end
writecell([arch_header; arch_data], excel_filepath, 'Sheet', 'Architecture');

% Sheet 4: View Weights
view_weights_header = {'View Name', 'View Index', 'Weight'};
view_weights_data = cell(numView, 3);
for v = 1:numView
    view_weights_data{v, 1} = experiment_info.view_names{v};
    view_weights_data{v, 2} = v;
    view_weights_data{v, 3} = alpha(v);
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

%% ==================== Helper Functions 辅助函数 ====================

function A = read_matrix_market(filename)
% Read Matrix Market coordinate format file
% Returns full matrix
%
% Input:
%   filename: path to Matrix Market format file
%
% Output:
%   A: full matrix

fid = fopen(filename, 'r');
if fid == -1
    error('Cannot open file: %s', filename);
end

% Skip comment lines starting with %
while true
    line = fgetl(fid);
    if ~startsWith(line, '%')
        break;
    end
end

% Read matrix dimensions: rows cols entries
header = sscanf(line, '%d %d %d');
rows = header(1);
cols = header(2);
entries = header(3);

% Initialize sparse matrix
A = zeros(rows, cols);

% Read data triplets (row col value)
for i = 1:entries
    line = fgetl(fid);
    data = sscanf(line, '%d %d %f');
    if length(data) == 3
        A(data(1), data(2)) = data(3);
    elseif length(data) == 2
        % Binary matrix (0/1)
        A(data(1), data(2)) = 1;
    end
end

fclose(fid);
end

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
% Inline if function for conditional expressions
% 内联if函数用于条件表达式
%
% Input:
%   condition: boolean condition
%   true_val: value to return if condition is true
%   false_val: value to return if condition is false
%
% Output:
%   result: true_val if condition is true, false_val otherwise

if condition
    result = true_val;
else
    result = false_val;
end
end
