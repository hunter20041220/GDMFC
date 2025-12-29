%==========================================================================
% GDMFC Demo on Caltech101-7 Dataset
%==========================================================================
% 此脚本演示如何在Caltech101-7数据集上使用GDMFC算法
% This script demonstrates how to use GDMFC algorithm on Caltech101-7 dataset
%
% Caltech101-7数据集: 1474张图像, 7类
% Caltech101-7 dataset: 1474 images, 7 classes
% 6个视图: 不同的图像特征提取方法 (Gabor, WM, CENTRIST, HOG, GIST, LBP)
% 6 views: Different image feature extraction methods
%   View 1: 48-dim (Gabor)
%   View 2: 40-dim (WM - Wavelet Moments)
%   View 3: 254-dim (CENTRIST)
%   View 4: 1984-dim (HOG)
%   View 5: 512-dim (GIST)
%   View 6: 928-dim (LBP)
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

%% Add paths 添加路径
% Get the root directory of GDMFC
script_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(script_dir);
addpath(genpath(fullfile(root_dir, 'core')));
addpath(genpath(fullfile(root_dir, 'utils')));
addpath(genpath(fullfile(root_dir, 'solvers')));

fprintf('========================================\n');
fprintf('GDMFC Demo on Caltech101-7 Dataset\n');
fprintf('========================================\n');
fprintf('Experiment Time: %s\n', experiment_info.timestamp);
fprintf('MATLAB Version: %s\n', experiment_info.matlab_version);
fprintf('Random Seed: %d\n', rng_seed);
fprintf('========================================\n\n');

%% ==================== Load Caltech101-7 Dataset 加载Caltech101-7数据集 ====================
fprintf('Step 1: Loading Caltech101-7 dataset...\n');

% Dataset path 数据集路径
dataset_name = 'Caltech101-7';
dataPath = 'E:\research\paper\multiview\dataset\Caltech101-7.mat';

if ~exist(dataPath, 'file')
    error('Dataset file not found: %s\nPlease check the path.', dataPath);
end

load(dataPath);

% Dataset info: X contains 6 views with different image features
% 数据集信息：X包含6个视图，使用不同的图像特征
numView = length(X);
numSamples = size(X{1}, 1);
numCluster = length(unique(y));

% Record dataset information 记录数据集详细信息
experiment_info.dataset_name = dataset_name;
experiment_info.dataset_path = dataPath;
experiment_info.num_views = numView;
experiment_info.num_samples = numSamples;
experiment_info.num_clusters = numCluster;
experiment_info.num_classes = numCluster;

% Record feature dimensions for each view 记录每个视图的特征维度
experiment_info.feature_dims = zeros(numView, 1);
for v = 1:numView
    experiment_info.feature_dims(v) = size(X{v}, 2);
end

% View names for Caltech101-7
experiment_info.view_names = {'Gabor', 'WM', 'CENTRIST', 'HOG', 'GIST', 'LBP'};

fprintf('  Dataset: %s\n', dataset_name);
fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', numSamples);
fprintf('  Number of classes: %d\n', numCluster);
fprintf('  View features:\n');
for v = 1:numView
    fprintf('    View %d (%s): %d dimensions\n', v, experiment_info.view_names{v}, experiment_info.feature_dims(v));
end
fprintf('\n');

%% ==================== Data Preprocessing 数据预处理 ====================
fprintf('Step 2: Preprocessing data...\n');

% Record preprocessing method 记录预处理方法
experiment_info.preprocessing_method = 'NormalizeFea_L2';
experiment_info.preprocessing_style = 'Standard';

% Normalize each view to unit length
% 将每个视图归一化为单位长度
for v = 1:numView
    X{v} = NormalizeFea(X{v}, 0);  % L2 normalization (sample-wise)
    fprintf('  View %d (%s) preprocessed: %d samples × %d features\n', ...
        v, experiment_info.view_names{v}, size(X{v}, 1), size(X{v}, 2));
end
fprintf('  Data normalized (L2 norm).\n\n');

%% ==================== Algorithm Parameters 算法参数 ====================
fprintf('Step 3: Setting algorithm parameters...\n');

% Layer configuration 层配置
% For Caltech101-7 with 7 classes and 6 views, use appropriate layer structure
% 对于7类、6视图的Caltech101-7，使用合适的层结构
layers = [200, 100];  % hidden layers: 200 -> 100 (output layer is numCluster=7)

% Record layer configuration 记录层配置
experiment_info.layers = layers;
experiment_info.full_architecture = [experiment_info.feature_dims', layers, numCluster];

% Algorithm parameters 算法参数
options = struct();
options.lambda1 = 0.001;     % HSIC diversity coefficient HSIC多样性系数
options.lambda2 = 0.01;      % co-orthogonal constraint coefficient 协正交约束系数
options.beta = 100;          % graph regularization coefficient 图正则化系数
options.gamma = 1.2;         % view weight parameter (must be > 1) 视图权重参数
options.graph_k = 7;         % number of neighbors for graph construction 图构造邻居数
options.maxIter = 100;       % maximum iterations 最大迭代次数
options.tol = 1e-5;          % convergence tolerance 收敛容差

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
fprintf('  Final view weights:\n');
for v = 1:numView
    fprintf('    View %d (%s): %.4f\n', v, experiment_info.view_names{v}, alpha(v));
end
fprintf('\n');

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
NMI = MutualInfo(y, predict_labels);

% Purity: Clustering Purity 聚类纯度
Purity = compute_purity(y, predict_labels);

% Record evaluation metrics 记录评估指标
experiment_info.ACC = ACC;
experiment_info.NMI = NMI;
experiment_info.Purity = Purity;

fprintf('Results on Caltech101-7 Dataset:\n');
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
figure('Name', 'GDMFC on Caltech101-7', 'Position', [100, 100, 1400, 500]);

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
set(gca, 'XTickLabel', experiment_info.view_names);
xtickangle(45);
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

% Create results directory with timestamp subfolder 创建带时间戳子文件夹的results目录
results_base_dir = fullfile(root_dir, 'results');
if ~exist(results_base_dir, 'dir')
    mkdir(results_base_dir);
end

% Create experiment-specific subfolder 创建实验专用子文件夹
exp_folder_name = sprintf('GDMFC_%s_%s', dataset_name, experiment_info.timestamp);
results_dir = fullfile(results_base_dir, exp_folder_name);
mkdir(results_dir);
fprintf('  Created experiment folder: %s\n', exp_folder_name);

% Generate filenames 生成文件名（不带时间戳，因为文件夹已经有了）
base_filename = sprintf('GDMFC_%s', dataset_name);
mat_filename = [base_filename, '.mat'];
txt_filename = [base_filename, '.txt'];
excel_filename = [base_filename, '.xlsx'];
fig_filename = [base_filename, '.png'];

% Save .mat file 保存.mat文件
mat_filepath = fullfile(results_dir, mat_filename);
save(mat_filepath, 'results');
fprintf('  [1/4] MAT file saved: %s\n', mat_filename);

% Save text report 保存文本报告
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
fprintf(fid, 'View Features:\n');
for v = 1:numView
    fprintf(fid, '  View %d (%s): %d dimensions\n', v, experiment_info.view_names{v}, experiment_info.feature_dims(v));
end
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
fprintf(fid, 'View Weights:\n');
for v = 1:numView
    fprintf(fid, '  %s: %.6f\n', experiment_info.view_names{v}, alpha(v));
end
fprintf(fid, '\n');

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

% Save Excel file with detailed results 保存Excel文件
excel_filepath = fullfile(results_dir, excel_filename);

% Sheet 1: Summary 摘要
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

% Sheet 2: Parameters 参数
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

% Sheet 3: Architecture 架构
arch_header = {'Layer', 'Dimension'};
arch_data = cell(length(experiment_info.full_architecture), 2);
for i = 1:length(experiment_info.full_architecture)
    if i <= experiment_info.num_views
        arch_data{i, 1} = sprintf('Input View %d (%s)', i, experiment_info.view_names{i});
    elseif i <= experiment_info.num_views + length(layers)
        arch_data{i, 1} = sprintf('Hidden Layer %d', i - experiment_info.num_views);
    else
        arch_data{i, 1} = 'Output (Cluster)';
    end
    arch_data{i, 2} = experiment_info.full_architecture(i);
end
writecell([arch_header; arch_data], excel_filepath, 'Sheet', 'Architecture');

% Sheet 4: View Weights 视图权重
view_weights_header = {'View', 'Feature Type', 'Dimension', 'Weight'};
view_weights_data = cell(numView, 4);
for v = 1:numView
    view_weights_data{v, 1} = sprintf('View %d', v);
    view_weights_data{v, 2} = experiment_info.view_names{v};
    view_weights_data{v, 3} = experiment_info.feature_dims(v);
    view_weights_data{v, 4} = alpha(v);
end
writecell([view_weights_header; view_weights_data], excel_filepath, 'Sheet', 'ViewWeights');

% Sheet 5: Convergence 收敛过程
conv_header = {'Iteration', 'Objective Value'};
conv_data = cell(length(obj_values), 2);
for i = 1:length(obj_values)
    conv_data{i, 1} = i;
    conv_data{i, 2} = obj_values(i);
end
writecell([conv_header; conv_data], excel_filepath, 'Sheet', 'Convergence');

fprintf('  [3/4] Excel file saved: %s\n', excel_filename);

% Save figure as image 保存图表为图片
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
fprintf('GDMFC Demo on Caltech101-7 Completed Successfully!\n');
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
% Inline if function 内联条件函数
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
