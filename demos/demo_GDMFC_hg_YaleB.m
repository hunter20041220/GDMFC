%==========================================================================
% GDMFC-Hypergraph Demo on YaleB_650 Dataset
%==========================================================================
% 此脚本演示如何在Extended Yale Face Database B数据集上使用GDMFC_Hypergraph算法
% This script demonstrates how to use GDMFC_Hypergraph algorithm on YaleB dataset
%
% YaleB_650数据集: 650个人脸图像, 10个类别
% YaleB_650 dataset: 650 face images, 10 classes
% 3个视图: View 1 (2500维), View 2 (3304维), View 3 (6750维) - 不同图像特征
% 3 views: View 1 (2500-dim), View 2 (3304-dim), View 3 (6750-dim) - Different image features
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
experiment_info.algorithm = 'GDMFC_Hypergraph';

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
fprintf('GDMFC-Hypergraph Demo on YaleB_650 Dataset\n');
fprintf('========================================\n');
fprintf('Experiment Time: %s\n', experiment_info.timestamp);
fprintf('MATLAB Version: %s\n', experiment_info.matlab_version);
fprintf('Random Seed: %d\n', rng_seed);
fprintf('========================================\n\n');

%% ==================== Load YaleB_650 Dataset 加载YaleB数据集 ====================
fprintf('Step 1: Loading YaleB_650 dataset...\n');

% Dataset path 数据集路径
dataset_name = 'YaleB_650';
dataPath = 'E:\research\paper\multiview\dataset\YaleB_650.mat';

if ~exist(dataPath, 'file')
    error('Dataset file not found: %s\nPlease check the path.', dataPath);
end

load(dataPath);

% Dataset info: data{1}, data{2}, data{3} are three image feature views
% 数据集信息：data{1}, data{2}, data{3}是三个图像特征视图

% 获取视图数和样本数
numView = length(data);
y = label';
numCluster = length(unique(y));

% 转换为HDDMF格式：d × n (特征 × 样本)
X = cell(numView, 1);
for v = 1:numView
    X{v} = data{v};  % HDDMF格式已经是 d×n
end

numSamples = size(X{1}, 2);  % HDDMF格式：列数是样本数

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
    experiment_info.feature_dims(v) = size(X{v}, 1);  % HDDMF格式：行数是特征数
end

fprintf('  Dataset: %s\n', dataset_name);
fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', numSamples);
fprintf('  Number of classes: %d\n', numCluster);
fprintf('  Feature dimensions: ');
for v = 1:numView
    fprintf('%d', experiment_info.feature_dims(v));
    if v < numView, fprintf(', '); end
end
fprintf('\n\n');

%% ==================== Data Preprocessing 数据预处理 ====================
fprintf('Step 2: Data preprocessing...\n');
% 注意：数据归一化现在在GDMFC_Hypergraph内部完成（HDDMF方式）
fprintf('  Data normalization will be done inside GDMFC_Hypergraph (HDDMF style).\n\n');

% Record preprocessing method 记录预处理方法
experiment_info.preprocessing_method = 'HDDMF_column_normalization';
experiment_info.preprocessing_style = 'HDDMF';

%% ==================== Algorithm Parameters 算法参数 ====================
fprintf('Step 3: Setting algorithm parameters...\n');

% Layer configuration 层配置
% For YaleB_650 with 10 classes and 3 views, use appropriate layer structure
% 对于10类、3视图的YaleB_650，使用合适的层结构
layers = [200, 100];  % hidden layers: 200 -> 100 (output layer is numCluster=10)

% Record layer configuration 记录层配置
experiment_info.layers = layers;
experiment_info.full_architecture = [experiment_info.feature_dims', layers, numCluster];

% Algorithm parameters 算法参数
% Note: 使用HDDMF的默认参数设置
options = struct();
options.beta = 0.1;          % hypergraph regularization (HDDMF默认)
options.lambda1 = 0.0001;    % diversity (对应HDDMF的mu)
options.lambda2 = 0;         % orthogonal constraint (HDDMF没有用)
options.gamma = 1.5;         % view weight parameter (HDDMF默认)
options.k_hyper = 5;         % k-NN for hypergraph (HDDMF默认)
options.maxIter = 100;       % maximum iterations 最大迭代次数
options.tol = 1e-5;          % convergence tolerance 收敛容差
options.verbose = true;      % display iteration information 显示迭代信息

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
fprintf('  beta (hypergraph regularization): %g\n', options.beta);
fprintf('  lambda1 (diversity): %g\n', options.lambda1);
fprintf('  lambda2 (orthogonal): %g\n', options.lambda2);
fprintf('  gamma (view weight): %g\n', options.gamma);
fprintf('  k_hyper: %d\n', options.k_hyper);
fprintf('  maxIter: %d, tol: %.0e\n\n', options.maxIter, options.tol);

%% ==================== Run GDMFC-Hypergraph Algorithm 运行超图算法 ====================
fprintf('Step 4: Running GDMFC-Hypergraph algorithm...\n');
fprintf('----------------------------------------\n');

tic;
[H, Z, alpha, obj_values] = GDMFC_Hypergraph(X, numCluster, layers, options);
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

% Normalize H and perform spectral clustering
H_normalized = H ./ (sqrt(sum(H.^2, 2)) + eps);
S = H_normalized * H_normalized';

% Use spectral clustering to get final cluster assignments
predict_labels = SpectralClustering(S, numCluster);

fprintf('  Spectral clustering completed.\n\n');

%% ==================== Evaluation 评估 ====================
fprintf('Step 6: Evaluating clustering performance...\n');

% ACC: Clustering Accuracy 聚类准确率
res = bestMap(y, predict_labels);
ACC = length(find(y == res)) / length(y);

% NMI: Normalized Mutual Information 归一化互信息
NMI = MutualInfo(y, predict_labels);

% Purity: Clustering Purity 聚类纯度
Purity = purity(predict_labels, y);

% Record evaluation metrics 记录评估指标
experiment_info.ACC = ACC;
experiment_info.NMI = NMI;
experiment_info.Purity = Purity;

fprintf('Results on YaleB_650 Dataset (Hypergraph):\n');
fprintf('  ACC    = %.4f (%.2f%%)\n', ACC, ACC*100);
fprintf('  NMI    = %.4f (%.2f%%)\n', NMI, NMI*100);
fprintf('  Purity = %.4f (%.2f%%)\n', Purity, Purity*100);
fprintf('========================================\n\n');

%% ==================== Prepare Results Structure 准备结果结构 ====================
% Create results structure with ALL information 创建包含所有信息的结果结构体
results = struct();

% Experiment metadata 实验元数据
results.experiment_info = experiment_info;

% Performance metrics 性能指标
results.ACC_mean = ACC;
results.NMI_mean = NMI;
results.Purity_mean = Purity;

% Algorithm outputs 算法输出
results.alpha = alpha;
results.obj_values = obj_values;
results.predict_labels = predict_labels;
results.true_labels = y;
results.H_final = H;
results.H_normalized = H_normalized;
results.S_consensus = S;
results.elapsed_time = elapsed_time;

% Algorithm parameters (also in experiment_info, but keep for backward compatibility)
results.options = options;
results.layers = layers;

%% ==================== Visualization 可视化 ====================
fprintf('Step 7: Generating visualizations...\n');

% Create figure with convergence and performance plots
figure('Position', [100, 100, 1200, 400]);

% Plot 1: Convergence curve
subplot(1, 3, 1);
plot(1:length(obj_values), obj_values, 'b-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 12);
ylabel('Objective Value', 'FontSize', 12);
title('Convergence Curve (Hypergraph)', 'FontSize', 14);
grid on;

% Plot 2: View weights
subplot(1, 3, 2);
bar(alpha);
xlabel('View Index', 'FontSize', 12);
ylabel('Weight', 'FontSize', 12);
title('View Weights (Hypergraph)', 'FontSize', 14);
set(gca, 'XTick', 1:numView);
grid on;

% Plot 3: Performance metrics
subplot(1, 3, 3);
metrics = [ACC, NMI, Purity];
bar(metrics);
xlabel('Metric', 'FontSize', 12);
ylabel('Value', 'FontSize', 12);
title('Clustering Performance (Hypergraph)', 'FontSize', 14);
set(gca, 'XTickLabel', {'ACC', 'NMI', 'Purity'});
ylim([0, 1]);
grid on;

fprintf('  Visualization completed.\n\n');

%% ==================== Save Results 保存结果 ====================
fprintf('Step 8: Saving comprehensive results...\n');

% Create results directory with timestamp subfolder 创建带时间戳子文件夹的results目录
results_base_dir = fullfile(root_dir, 'results');
if ~exist(results_base_dir, 'dir')
    mkdir(results_base_dir);
end

% Create experiment-specific subfolder 创建实验专用子文件夹
exp_folder_name = sprintf('GDMFC_Hypergraph_%s_%s', dataset_name, experiment_info.timestamp);
results_dir = fullfile(results_base_dir, exp_folder_name);
mkdir(results_dir);
fprintf('  Created experiment folder: %s\n', exp_folder_name);

% Generate filenames 生成文件名（不带时间戳，因为文件夹已经有了）
base_filename = sprintf('GDMFC_Hypergraph_%s', dataset_name);
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
fprintf(fid, 'GDMFC-Hypergraph Experiment Report - YaleB Dataset\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, '=== EXPERIMENT METADATA ===\n');
fprintf(fid, 'Algorithm: %s\n', experiment_info.algorithm);
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
fprintf(fid, 'beta (hypergraph regularization): %.6f\n', options.beta);
fprintf(fid, 'lambda1 (diversity): %.6f\n', options.lambda1);
fprintf(fid, 'lambda2 (orthogonal): %.6f\n', options.lambda2);
fprintf(fid, 'gamma (view weight): %.6f\n', options.gamma);
fprintf(fid, 'k_hyper (hypergraph neighbors): %d\n', options.k_hyper);
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

% Sheet 1: Summary 摘要
summary_data = {
    'Metric', 'Value';
    '=== EXPERIMENT INFO ===', '';
    'Algorithm', experiment_info.algorithm;
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

% Sheet 2: Parameters 参数
param_data = {
    'Parameter', 'Value', 'Description';
    'beta', options.beta, 'Hypergraph regularization coefficient';
    'lambda1', options.lambda1, 'HSIC diversity coefficient';
    'lambda2', options.lambda2, 'Orthogonal constraint coefficient';
    'gamma', options.gamma, 'View weight parameter';
    'k_hyper', options.k_hyper, 'Number of neighbors for hypergraph';
    'maxIter', options.maxIter, 'Maximum iterations';
    'tol', options.tol, 'Convergence tolerance';
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
fprintf('GDMFC-Hypergraph Demo Completed Successfully!\n');
fprintf('========================================\n');
fprintf('ACC: %.2f%%, NMI: %.2f%%, Purity: %.2f%%\n', ACC*100, NMI*100, Purity*100);

%% Helper function
function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
