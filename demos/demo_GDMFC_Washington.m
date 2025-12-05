%==========================================================================
% GDMFC Demo on Washington WebKB Dataset
%==========================================================================
% 此脚本演示如何在Washington WebKB数据集上使用GDMFC算法
% This script demonstrates how to use GDMFC algorithm on Washington WebKB dataset
%
% Washington数据集: 230个文档, 5类 (project, course, student, faculty, staff)
% Washington dataset: 230 documents, 5 classes
% 4个视图: content (文档-词), inbound/outbound/cites (链接)
%==========================================================================

clear; clc; close all;

%% Add paths 添加路径
addpath(genpath('../DMF_MVC'));  % 添加辅助函数路径

fprintf('========================================\n');
fprintf('GDMFC Demo on Washington WebKB Dataset\n');
fprintf('========================================\n\n');

%% ==================== Load Washington Dataset 加载Washington数据集 ====================
fprintf('Step 1: Loading Washington WebKB dataset...\n');

% Dataset path 数据集路径
dataPath = '../../dataset/Washington';

% 读取标签
labelsFile = fullfile(dataPath, 'washington_act.txt');
fid = fopen(labelsFile, 'r');
labels = fscanf(fid, '%d');
fclose(fid);

numSamples = length(labels);
numCluster = length(unique(labels));

fprintf('  Number of documents: %d\n', numSamples);
fprintf('  Number of classes: %d\n', numCluster);

% 读取类别名称
labelNamesFile = fullfile(dataPath, 'labels.txt');
fid = fopen(labelNamesFile, 'r');
labelNames = textscan(fid, '%s');
labelNames = labelNames{1};
fclose(fid);

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

fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', numSamples);
fprintf('  Number of clusters: %d\n\n', numCluster);

%% ==================== Data Preprocessing 数据预处理 ====================
fprintf('Step 3: Preprocessing data...\n');

% Preprocessing: run user preprocessing function first, then normalize
% Choose preprocessing mode for data_guiyi_choos:
% 1 - min-max, 2 - min-max (transposed), 3 - column-wise L2, 4 - column sum, 5 - global
preprocess_mode = 3; % default: column-wise L2 normalization
X = data_guiyi_choos(X, preprocess_mode);

% Then apply sample-feature normalization (NormalizeFea keeps behavior consistent)
for v = 1:numView
    X{v} = NormalizeFea(X{v}, 0);  % L2 normalization (sample-wise)
end

fprintf('  Data preprocessed (mode %d) and normalized (L2 norm).\n\n', preprocess_mode);

%% ==================== Algorithm Parameters 算法参数 ====================
fprintf('Step 4: Setting algorithm parameters...\n');

% Layer configuration 层配置
% 对于5类，使用适当的层结构
 layers = [100, 50, 20];  % hidden layers: 100 -> 50 -> 20 -> 5
% layers = [450, 150, 40];

% Algorithm parameters 算法参数
options = struct();
options.lambda1 = 0.01;     % HSIC diversity coefficient
options.lambda2 = 0.001;     % co-orthogonal constraint coefficient
options.beta = 200;         % graph regularization coefficient
options.gamma = 1.2;        % view weight parameter (must be > 1)
options.graph_k = 7;        % number of neighbors for graph construction
options.maxIter = 100;      % maximum iterations
options.tol = 1e-5;         % convergence tolerance

fprintf('  Layer structure: [');
for i = 1:length(layers)
    fprintf('%d', layers(i));
    if i < length(layers)
        fprintf(', ');
    end
end
fprintf(', %d]\n', numCluster);
fprintf('  lambda1 (HSIC): %.4f\n', options.lambda1);
fprintf('  lambda2 (orthogonal): %.4f\n', options.lambda2);
fprintf('  beta (graph reg): %.4f\n', options.beta);
fprintf('  gamma (view weight): %.2f\n', options.gamma);
fprintf('  graph_k: %d\n', options.graph_k);
fprintf('  maxIter: %d, tol: %.0e\n\n', options.maxIter, options.tol);

%% ==================== Run GDMFC 运行GDMFC ====================
fprintf('Step 5: Running GDMFC algorithm...\n');
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
fprintf(']\n');
fprintf('  View names: content, inbound, outbound, cites\n\n');

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

fprintf('Results on Washington WebKB Dataset:\n');
fprintf('  ACC    = %.4f (%.2f%%)\n', ACC, ACC*100);
fprintf('  NMI    = %.4f\n', NMI);
fprintf('  Purity = %.4f (%.2f%%)\n', Purity, Purity*100);
fprintf('========================================\n\n');

%% ==================== Visualization 可视化 ====================
fprintf('Step 8: Visualizing results...\n');

% Plot 1: Objective function convergence 目标函数收敛曲线
figure('Name', 'GDMFC on Washington Dataset', 'Position', [100, 100, 1200, 400]);

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
set(gca, 'XTickLabel', {'Content', 'Inbound', 'Outbound', 'Cites'});
grid on;

% Plot 3: Confusion matrix 混淆矩阵
subplot(1, 3, 3);
confMat = confusionmat(y, res);
imagesc(confMat);
colormap('hot');
colorbar;
xlabel('Predicted Label', 'FontSize', 12);
ylabel('True Label', 'FontSize', 12);
title(sprintf('Confusion Matrix (ACC=%.2f%%)', ACC*100), 'FontSize', 14);
axis square;

fprintf('  Visualization completed.\n\n');

%% ==================== Save Results 保存结果 ====================
fprintf('Step 9: Saving results...\n');

results = struct();
results.ACC = ACC;
results.NMI = NMI;
results.Purity = Purity;
results.alpha = alpha;
results.obj_values = obj_values;
results.predict_labels = predict_labels;
results.true_labels = y;
results.H_final = H_final;
results.elapsed_time = elapsed_time;

save('GDMFC_results_Washington.mat', 'results');
fprintf('  Results saved to GDMFC_results_Washington.mat\n\n');

%% ==================== Class Distribution Analysis 类别分布分析 ====================
fprintf('Step 10: Analyzing class distribution...\n');

% 真实类别分布
figure('Name', 'Class Distribution', 'Position', [150, 150, 1000, 400]);

subplot(1, 2, 1);
true_dist = histcounts(y, 1:numCluster+1);
bar(true_dist);
xlabel('Class ID', 'FontSize', 12);
ylabel('Number of Documents', 'FontSize', 12);
title('True Class Distribution', 'FontSize', 14);
set(gca, 'XTick', 1:numCluster);
set(gca, 'XTickLabel', labelNames);
xtickangle(45);
grid on;

% 预测类别分布
subplot(1, 2, 2);
pred_dist = histcounts(res, 1:numCluster+1);
bar(pred_dist);
xlabel('Class ID', 'FontSize', 12);
ylabel('Number of Documents', 'FontSize', 12);
title('Predicted Class Distribution', 'FontSize', 14);
set(gca, 'XTick', 1:numCluster);
set(gca, 'XTickLabel', labelNames);
xtickangle(45);
grid on;

fprintf('  Class distribution analysis completed.\n\n');

fprintf('========================================\n');
fprintf('GDMFC Demo Completed Successfully!\n');
fprintf('========================================\n');

%% ==================== Helper Function ====================
function A = read_matrix_market(filename)
% Read Matrix Market coordinate format file
% Returns full matrix

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
