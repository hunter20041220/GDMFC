%==========================================================================
% GDMFC Improved Demo on ORL40_3_400 Dataset
%==========================================================================
% 此脚本结合HDDMF的优点改进GDMFC，在保持GDMFC核心算法不变的前提下：
% 1. 借鉴HDDMF的数据预处理方法（NormalizeFea列归一化）
% 2. 借鉴HDDMF的图构建方法（PKN概率k近邻）
% 3. 使用ORL40_3_400.mat数据集
%
% Improvements from HDDMF:
% - Data preprocessing: NormalizeFea column normalization
% - Graph construction: PKN (Probabilistic k-Nearest Neighbors)
% - Optimization: L2 normalization approach
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
root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'core')));
addpath(genpath(fullfile(root_dir, 'utils')));
addpath(genpath(fullfile(root_dir, 'solvers')));

fprintf('========================================\n');
fprintf('GDMFC Improved Demo on ORL40_3_400 Dataset\n');
fprintf('Combining HDDMF advantages with GDMFC core algorithm\n');
fprintf('========================================\n');
fprintf('Experiment Time: %s\n', experiment_info.timestamp);
fprintf('MATLAB Version: %s\n', experiment_info.matlab_version);
fprintf('Random Seed: %d\n', rng_seed);
fprintf('========================================\n\n');

%% ==================== Load ORL40_3_400 Dataset 加载数据集 ====================
fprintf('Step 1: Loading ORL40_3_400 dataset...\n');

% Dataset path 数据集路径
root_dir = fileparts(fileparts(mfilename('fullpath')));
datadir = fullfile(root_dir, 'data');
dataset = 'ORL40_3_400';
dataf = fullfile(datadir, [dataset, '.mat']);

if ~exist(dataf, 'file')
    error('Dataset file not found: %s\nPlease check the path.', dataf);
end

load(dataf);
C = length(unique(label));
numView = length(data);
numSamples = size(data{1}, 2);
numCluster = C;

% Record dataset information 记录数据集详细信息
experiment_info.dataset_name = dataset;
experiment_info.dataset_path = dataf;
experiment_info.num_views = numView;
experiment_info.num_samples = numSamples;
experiment_info.num_clusters = numCluster;
experiment_info.num_classes = C;

% Record feature dimensions for each view 记录每个视图的特征维度
experiment_info.feature_dims = zeros(numView, 1);
for v = 1:numView
    experiment_info.feature_dims(v) = size(data{v}, 1);
end

fprintf('  Dataset: %s\n', dataset);
fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', numSamples);
fprintf('  Number of clusters: %d\n', numCluster);
fprintf('  Feature dimensions: ');
for v = 1:numView
    fprintf('%d', experiment_info.feature_dims(v));
    if v < numView, fprintf(', '); end
end
fprintf('\n\n');

% Extract labels
y = label';

%% ==================== Data Preprocessing 数据预处理 ====================
fprintf('Step 2: Preprocessing data (HDDMF style)...\n');

% HDDMF preprocessing approach: Direct NormalizeFea column normalization
% HDDMF预处理方法：直接使用NormalizeFea进行列归一化
fprintf('  Applying HDDMF-style preprocessing (NormalizeFea column normalization)...\n');

% Record preprocessing method 记录预处理方法
experiment_info.preprocessing_method = 'NormalizeFea_column_L2';
experiment_info.preprocessing_style = 'HDDMF';

% NormalizeFea should be available from HDDMF path
% NormalizeFea应该可以从HDDMF路径获得

% Apply HDDMF preprocessing: column-wise L2 normalization
% 应用HDDMF预处理：列L2归一化
X = cell(numView, 1);
for v = 1:numView
    % HDDMF style: NormalizeFea(data{v}, 0) - column normalization
    % 注意：HDDMF中data是列向量为样本，需要转置
    X{v} = NormalizeFea(data{v}', 0);  % 转置后列归一化（样本在行）
    fprintf('  View %d: %d samples × %d features\n', v, size(X{v}, 1), size(X{v}, 2));
end

fprintf('  Data preprocessing completed (HDDMF style).\n\n');

%% ==================== Algorithm Parameters 算法参数 ====================
fprintf('Step 3: Setting algorithm parameters...\n');

% Layer configuration 层配置
% 使用和HDDMF完全相同的配置
layers = [100, 50];  % 和HDDMF一致

% Record layer configuration 记录层配置
experiment_info.layers = layers;
experiment_info.full_architecture = [experiment_info.feature_dims', layers, numCluster];

% Algorithm parameters 算法参数
options = struct();

% ========== 使用HDDMF的参数设置 ==========
options.lambda1 = 0.0001;    % mu in HDDMF (diversity)
options.lambda2 = 0;         % 不使用正交约束（HDDMF中没有）
options.beta = 0.1;          % 图正则化（和HDDMF一致）
options.gamma = 1.5;         % 视图权重参数
options.graph_k = 5;         % 图构建的邻居数
options.maxIter = 100;       % 最大迭代次数
options.tol = 1e-5;          % 收敛容忍度

% ========== HDDMF风格设置 ==========
options.use_PKN = true;              % 使用PKN方法（HDDMF风格）
options.use_heat_kernel = false;     % 不使用热核
options.use_dynamic_graph = true;    % 动态更新图（HDDMF风格）
options.use_simple_diversity = true; % 使用简单diversity项（HDDMF风格）

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
fprintf('  Graph method: %s\n', iif(options.use_PKN, 'PKN (HDDMF)', 'Heat Kernel (GDMFC)'));
fprintf('  Dynamic graph update: %s\n', iif(options.use_dynamic_graph, 'Yes', 'No'));
fprintf('  Simple diversity: %s\n', iif(options.use_simple_diversity, 'Yes (HDDMF)', 'No (HSIC)'));
fprintf('  maxIter: %d, tol: %.0e\n\n', options.maxIter, options.tol);

%% ==================== Modified GDMFC with HDDMF Graph Construction ====================
fprintf('Step 4: Running improved GDMFC (方案A - HDDMF简化风格)...\n');
fprintf('  策略: 先测试基础版本(lambda1=0, lambda2=0)\n');
fprintf('  预期: 应达到HDDMF水平(~90%% ACC)\n');
fprintf('----------------------------------------\n');

tic;

% Create a modified version of GDMFC that uses PKN for graph construction
% 创建使用PKN图构建的GDMFC改进版本
[H, Z, alpha, obj_values] = GDMFC_improved(X, numCluster, layers, options);

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

% GDMFC returns the weighted average of final layer representations
% H已经是加权平均后的最终表示矩阵 (n × numCluster)
H_final = H;

% Construct similarity matrix for spectral clustering 构造相似度矩阵
S = H_final * H_final';
S = (S + S') / 2;  % 确保对称
S = max(S, 0);     % 确保非负

% SpectralClustering should be available from utils path
% SpectralClustering应该可以从utils路径获得

predict_labels = SpectralClustering(S, numCluster);

fprintf('  Spectral clustering completed.\n\n');

%% ==================== Evaluation 评估 ====================
fprintf('Step 6: Evaluating clustering performance...\n');
fprintf('========================================\n');

% Evaluation functions should be available from utils path
% 评估函数应该可以从utils路径获得
% Check if compute_purity exists, if not use a simple version
if ~exist('compute_purity', 'file')
    % Will define inline
    compute_purity_func = @(true_labels, pred_labels) ...
        sum(true_labels == bestMap(true_labels, pred_labels)) / length(true_labels);
else
    compute_purity_func = @compute_purity;
end

% Compute evaluation metrics 计算评估指标
% ACC: Clustering Accuracy 聚类准确率
res = bestMap(y, predict_labels);
ACC = length(find(y == res)) / length(y);

% NMI: Normalized Mutual Information 归一化互信息
NMI = MutualInfo(y, predict_labels);

% Purity: Clustering Purity 聚类纯度
Purity = compute_purity_func(y, predict_labels);

% Record evaluation metrics 记录评估指标
experiment_info.ACC = ACC;
experiment_info.NMI = NMI;
experiment_info.Purity = Purity;

fprintf('Results on ORL40_3_400 Dataset:\n');
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

% Plot 1: Objective function convergence 目标函数收敛曲线
figure('Name', 'GDMFC Improved on ORL40_3_400', 'Position', [100, 100, 1200, 400]);

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
results.H_final = H_final;
results.elapsed_time = elapsed_time;

% Algorithm parameters (also in experiment_info, but keep for backward compatibility)
results.options = options;
results.layers = layers;

% Create results directory with timestamp subfolder 创建带时间戳子文件夹的results目录
root_dir = fileparts(fileparts(mfilename('fullpath')));
results_base_dir = fullfile(root_dir, 'results');
if ~exist(results_base_dir, 'dir')
    mkdir(results_base_dir);
end

% Create experiment-specific subfolder 创建实验专用子文件夹
exp_folder_name = sprintf('GDMFC_improved_%s_%s', dataset, experiment_info.timestamp);
results_dir = fullfile(results_base_dir, exp_folder_name);
mkdir(results_dir);
fprintf('  Created experiment folder: %s\n', exp_folder_name);

% Generate filenames 生成文件名（不带时间戳，因为文件夹已经有了）
base_filename = sprintf('GDMFC_improved_%s', dataset);
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
fprintf(fid, 'GDMFC Improved Experiment Report\n');
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
fprintf(fid, 'Use PKN: %d\n', options.use_PKN);
fprintf(fid, 'Use Heat Kernel: %d\n', options.use_heat_kernel);
fprintf(fid, 'Dynamic Graph: %d\n', options.use_dynamic_graph);
fprintf(fid, 'Simple Diversity: %d\n', options.use_simple_diversity);
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
    'lambda1', options.lambda1, 'Diversity coefficient';
    'lambda2', options.lambda2, 'Orthogonal constraint coefficient';
    'beta', options.beta, 'Graph regularization coefficient';
    'gamma', options.gamma, 'View weight parameter';
    'graph_k', options.graph_k, 'Number of neighbors for graph';
    'maxIter', options.maxIter, 'Maximum iterations';
    'tol', options.tol, 'Convergence tolerance';
    'use_PKN', options.use_PKN, 'Use PKN graph construction';
    'use_heat_kernel', options.use_heat_kernel, 'Use heat kernel';
    'use_dynamic_graph', options.use_dynamic_graph, 'Dynamic graph update';
    'use_simple_diversity', options.use_simple_diversity, 'Simple diversity term';
};
writecell(param_data, excel_filepath, 'Sheet', 'Parameters');

% Sheet 3: Architecture 架构
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

% Sheet 4: View Weights 视图权重
view_weights_header = {'View', 'Weight'};
view_weights_data = cell(numView, 2);
for v = 1:numView
    view_weights_data{v, 1} = sprintf('View %d', v);
    view_weights_data{v, 2} = alpha(v);
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
fprintf('GDMFC Improved Demo Completed Successfully!\n');
fprintf('========================================\n');

%% Helper function
function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

