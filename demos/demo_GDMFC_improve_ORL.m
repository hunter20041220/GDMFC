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

%% Add paths 添加路径
% Get the root directory of GDMFC
root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'core')));
addpath(genpath(fullfile(root_dir, 'utils')));
addpath(genpath(fullfile(root_dir, 'solvers')));

fprintf('========================================\n');
fprintf('GDMFC Improved Demo on ORL40_3_400 Dataset\n');
fprintf('Combining HDDMF advantages with GDMFC core algorithm\n');
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

fprintf('  Dataset: %s\n', dataset);
fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', numSamples);
fprintf('  Number of clusters: %d\n\n', numCluster);

% Extract labels
y = label';

%% ==================== Data Preprocessing 数据预处理 ====================
fprintf('Step 2: Preprocessing data (HDDMF style)...\n');

% HDDMF preprocessing approach: Direct NormalizeFea column normalization
% HDDMF预处理方法：直接使用NormalizeFea进行列归一化
fprintf('  Applying HDDMF-style preprocessing (NormalizeFea column normalization)...\n');

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

fprintf('----------------------------------------\n');
fprintf('  Time elapsed: %.2f seconds\n', elapsed_time);
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
fprintf('Step 9: Saving results...\n');

results = struct();
results.ACC_mean = ACC_mean;
results.ACC_std = ACC_std;
results.NMI_mean = NMI_mean;
results.NMI_std = NMI_std;
results.Purity_mean = Purity_mean;
results.Purity_std = Purity_std;
results.ACC_runs = ACC_runs;
results.NMI_runs = NMI_runs;
results.Purity_runs = Purity_runs;
results.alpha = alpha;
results.obj_values = obj_values;
results.predict_labels = predict_labels;
results.true_labels = y;
results.H_final = H_final;
results.elapsed_time = elapsed_time;
results.options = options;

root_dir = fileparts(fileparts(mfilename('fullpath')));
save(fullfile(root_dir, 'results', 'GDMFC_improved_results_ORL40_3_400.mat'), 'results');
fprintf('  Results saved to GDMFC_improved_results_ORL40_3_400.mat\n\n');

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

