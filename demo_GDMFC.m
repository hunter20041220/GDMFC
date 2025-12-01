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

%% ==================== Add Paths 添加路径 ====================
% Add paths to helper functions from other directories
% 添加其他目录中辅助函数的路径
addpath(genpath('../DMF_MVC/misc'));  % for evaluation functions
addpath(genpath('../DMF_MVC/approx_seminmf'));  % for Semi-NMF

fprintf('========================================\n');
fprintf('GDMFC Demo on WebKB Dataset\n');
fprintf('========================================\n\n');

%% ==================== Load Dataset 加载数据集 ====================
fprintf('Step 1: Loading WebKB dataset...\n');
dataPath = '../../dataset/WebKB.mat';
load(dataPath);

% Dataset info: X{1} is Anchor view, X{2} is Content view
% 数据集信息：X{1}是Anchor视图，X{2}是Content视图
numView = length(X);
numSample = size(X{1}, 1);
numCluster = length(unique(y));

fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', numSample);
fprintf('  Number of clusters: %d\n', numCluster);
for v = 1:numView
    fprintf('  View %d dimension: %d\n', v, size(X{v}, 2));
end
fprintf('\n');

%% ==================== Data Preprocessing 数据预处理 ====================
fprintf('Step 2: Preprocessing data...\n');
% Normalize each view to unit length
% 将每个视图归一化为单位长度
for v = 1:numView
    X{v} = NormalizeFea(X{v}, 0);  % L2 normalization
end
fprintf('  Data normalized (L2 norm).\n\n');

%% ==================== Algorithm Parameters 算法参数 ====================
fprintf('Step 3: Setting algorithm parameters...\n');

% Layer configuration 层配置
% For WebKB with 2 classes, we use a simple 2-layer structure
% 对于有2个类别的WebKB，使用简单的两层结构
layers = [100, 50];  % hidden layer dimensions: 100 -> 50 -> numCluster

% Algorithm parameters 算法参数
% 使用较小的正则化系数以保证数值稳定性
options = struct();
options.lambda1 = 0.001;    % HSIC diversity coefficient HSIC多样性系数 (降低以提高稳定性)
options.lambda2 = 0.010;    % co-orthogonal constraint coefficient 协正交约束系数 (降低)
options.beta = 115;        % graph regularization coefficient 图正则化系数 (降低)
options.gamma = 1.2;        % view weight parameter (must be > 1) 视图权重参数
options.graph_k = 7;        % number of neighbors for graph construction 图构造邻居数
options.maxIter = 100;      % maximum iterations 最大迭代次数
options.tol = 1e-5;         % convergence tolerance 收敛容差

fprintf('  Layer structure: [');
for i = 1:length(layers)
    fprintf('%d', layers(i));
    if i < length(layers)
        fprintf(', ');
    end
end
fprintf(', %d]\n', numCluster);
fprintf('  lambda1 (graph reg): %.3f\n', options.lambda1);
fprintf('  lambda2 (HSIC): %.3f\n', options.lambda2);
fprintf('  beta (orthogonal): %.3f\n', options.beta);
fprintf('  gamma (view weight): %.2f\n', options.gamma);
fprintf('  graph_k: %d\n', options.graph_k);
fprintf('  maxIter: %d, tol: %.0e\n\n', options.maxIter, options.tol);

%% ==================== Run GDMFC Algorithm 运行GDMFC算法 ====================
fprintf('Step 4: Running GDMFC algorithm...\n');
fprintf('----------------------------------------\n');

tic;
[H, Z, alpha, obj_values] = GDMFC(X, numCluster, layers, options);
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

fprintf('Results on WebKB Dataset:\n');
fprintf('  ACC    = %.4f (%.2f%%)\n', ACC, ACC*100);
fprintf('  NMI    = %.4f\n', NMI);
fprintf('  Purity = %.4f (%.2f%%)\n', Purity, Purity*100);
fprintf('========================================\n\n');

%% ==================== Visualization 可视化 ====================
fprintf('Step 7: Visualizing results...\n');

% Plot objective function convergence 绘制目标函数收敛曲线
figure('Name', 'GDMFC Convergence', 'Position', [100, 100, 800, 400]);

subplot(1, 2, 1);
plot(1:length(obj_values), obj_values, 'b-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 12);
ylabel('Objective Value', 'FontSize', 12);
title('Objective Function Convergence', 'FontSize', 14);
grid on;

subplot(1, 2, 2);
bar(alpha);
xlabel('View Index', 'FontSize', 12);
ylabel('Weight', 'FontSize', 12);
title('Learned View Weights', 'FontSize', 14);
set(gca, 'XTick', 1:numView);
grid on;

fprintf('  Visualization completed.\n\n');

%% ==================== Save Results 保存结果 ====================
fprintf('Step 8: Saving results...\n');

results = struct();
results.ACC = ACC;
results.NMI = NMI;
results.Purity = Purity;
results.alpha = alpha;
results.predict_labels = predict_labels;
results.obj_values = obj_values;
results.elapsed_time = elapsed_time;
results.options = options;
results.layers = layers;

save('GDMFC_results_WebKB.mat', 'results');
fprintf('  Results saved to GDMFC_results_WebKB.mat\n\n');

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
