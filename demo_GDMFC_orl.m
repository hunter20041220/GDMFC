%==========================================================================
% GDMFC Demo on ORL Face Dataset
%==========================================================================
% 此脚本演示如何在ORL人脸数据集上使用GDMFC算法
% This script demonstrates how to use GDMFC algorithm on ORL face dataset
%
% ORL数据集: 40个人,每人10张图像 (112×92像素)
% ORL dataset: 40 subjects, 10 images per person (112×92 pixels)
%==========================================================================

clear; clc; close all;

%% Add paths 添加路径
addpath(genpath('../DMF_MVC'));  % 添加辅助函数路径

fprintf('========================================\n');
fprintf('GDMFC Demo on ORL Face Dataset\n');
fprintf('========================================\n\n');

%% ==================== Load ORL Dataset 加载ORL数据集 ====================
fprintf('Step 1: Loading ORL face dataset...\n');

% Dataset path 数据集路径
dataPath = '../../dataset/orl';

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

% Organize into cell array 组织为cell数组
X = cell(1, 2);
X{1} = X1;
X{2} = X2;
y = labels;

numView = length(X);
numCluster = numSubjects;

fprintf('  View 1 (Downsampled pixels): %d dimensions\n', size(X{1}, 2));
fprintf('  View 2 (Block statistics): %d dimensions\n', size(X{2}, 2));
fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', numSamples);
fprintf('  Number of clusters: %d\n\n', numCluster);

%% ==================== Data Preprocessing 数据预处理 ====================
fprintf('Step 3: Preprocessing data...\n');

% Normalize each view 归一化每个视图
for v = 1:numView
    X{v} = NormalizeFea(X{v}, 0);  % L2 normalization
end
fprintf('  Data normalized (L2 norm).\n\n');

%% ==================== Algorithm Parameters 算法参数 ====================
fprintf('Step 4: Setting algorithm parameters...\n');

% Layer configuration 层配置
% 对于40类,使用更深的结构
layers = [500, 200, 100];  % hidden layers: 500 -> 200 -> 100 -> 40

% Algorithm parameters 算法参数
% 对于更复杂的数据集,使用更小的正则化系数
options = struct();
options.lambda1 = 0.0001;   % HSIC diversity coefficient
options.lambda2 = 0.0001;   % co-orthogonal constraint coefficient
options.beta = 0.001;       % graph regularization coefficient
options.gamma = 1.5;        % view weight parameter (must be > 1)
options.graph_k = 5;        % number of neighbors for graph construction
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

fprintf('Results on ORL Face Dataset:\n');
fprintf('  ACC    = %.4f (%.2f%%)\n', ACC, ACC*100);
fprintf('  NMI    = %.4f\n', NMI);
fprintf('  Purity = %.4f (%.2f%%)\n', Purity, Purity*100);
fprintf('========================================\n\n');

%% ==================== Visualization 可视化 ====================
fprintf('Step 8: Visualizing results...\n');

% Plot 1: Objective function convergence 目标函数收敛曲线
figure('Name', 'GDMFC on ORL Dataset', 'Position', [100, 100, 1200, 400]);

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
set(gca, 'XTickLabel', {'Pixel', 'Block Stats'});
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

save('GDMFC_results_ORL.mat', 'results');
fprintf('  Results saved to GDMFC_results_ORL.mat\n\n');

%% ==================== Sample Visualization 样本可视化 ====================
fprintf('Step 10: Visualizing sample clustering results...\n');

% 随机选择几个类别展示
numClassesToShow = 5;
randomClasses = randperm(numCluster, numClassesToShow);

figure('Name', 'Sample Clustering Results', 'Position', [150, 150, 1000, 600]);

for idx = 1:numClassesToShow
    classID = randomClasses(idx);
    
    % 找到该类的样本
    trueClassIndices = find(y == classID);
    predClassIndices = find(res == classID);
    
    % 显示该类的一些真实样本
    subplot(2, numClassesToShow, idx);
    if ~isempty(trueClassIndices)
        sampleIdx = trueClassIndices(1);
        img = reshape(allImages(sampleIdx, :), imageHeight, imageWidth);
        imshow(uint8(img));
        title(sprintf('True Class %d', classID), 'FontSize', 10);
    end
    
    % 显示预测为该类的样本
    subplot(2, numClassesToShow, idx + numClassesToShow);
    if ~isempty(predClassIndices)
        sampleIdx = predClassIndices(1);
        img = reshape(allImages(sampleIdx, :), imageHeight, imageWidth);
        imshow(uint8(img));
        actualClass = y(sampleIdx);
        if actualClass == classID
            title(sprintf('Pred %d (✓)', classID), 'FontSize', 10, 'Color', 'g');
        else
            title(sprintf('Pred %d (✗, true=%d)', classID, actualClass), 'FontSize', 10, 'Color', 'r');
        end
    end
end

fprintf('  Sample visualization completed.\n\n');

fprintf('========================================\n');
fprintf('GDMFC Demo Completed Successfully!\n');
fprintf('========================================\n');
