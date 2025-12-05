% Run ORL training with specific parameters
% Usage: run_orl_with_config
% This script loads ORL, runs preprocessing, trains GDMFC with the specified
% parameters, performs spectral clustering, evaluates ACC/NMI/Purity and saves results.

clear; clc; close all;
addpath(genpath('../DMF_MVC'));

fprintf('Running ORL experiment with user-specified parameters...\n');

% Dataset settings
dataPath = '../../dataset/orl';
numSubjects = 40;
imagesPerSubject = 10;
imageHeight = 112;
imageWidth = 92;
numSamples = numSubjects * imagesPerSubject;

% Load images
allImages = zeros(numSamples, imageHeight * imageWidth);
y = zeros(numSamples, 1);
sampleIdx = 1;
for subjID = 1:numSubjects
    subjFolder = fullfile(dataPath, sprintf('s%d', subjID));
    for imgID = 1:imagesPerSubject
        imgPath = fullfile(subjFolder, sprintf('%d.pgm', imgID));
        img = imread(imgPath);
        allImages(sampleIdx, :) = double(img(:)');
        y(sampleIdx) = subjID;
        sampleIdx = sampleIdx + 1;
    end
end

% Construct views (same as demo)
downsampleFactor = 2;
newH = imageHeight / downsampleFactor;
newW = imageWidth / downsampleFactor;
X1 = zeros(numSamples, newH * newW);
for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    imgDown = imresize(img, [newH, newW], 'bilinear');
    X1(i, :) = imgDown(:)';
end

blockSize = 8;
numBlocksH = floor(imageHeight / blockSize);
numBlocksW = floor(imageWidth / blockSize);
lbpFeatDim = numBlocksH * numBlocksW * 4;
X2 = zeros(numSamples, lbpFeatDim);
for i = 1:numSamples
    img = reshape(allImages(i, :), imageHeight, imageWidth);
    featIdx = 1;
    for bh = 1:numBlocksH
        for bw = 1:numBlocksW
            rowStart = (bh-1) * blockSize + 1;
            rowEnd = min(bh * blockSize, imageHeight);
            colStart = (bw-1) * blockSize + 1;
            colEnd = min(bw * blockSize, imageWidth);
            block = img(rowStart:rowEnd, colStart:colEnd);
            X2(i, featIdx) = mean(block(:));
            X2(i, featIdx+1) = std(block(:));
            X2(i, featIdx+2) = min(block(:));
            X2(i, featIdx+3) = max(block(:));
            featIdx = featIdx + 4;
        end
    end
end

X = cell(1,2);
X{1} = X1;
X{2} = X2;
numView = length(X);
numCluster = numSubjects;

% Preprocessing (use default mode 3)
preprocess_mode = 3; % column-wise L2
X = data_guiyi_choos(X, preprocess_mode);
for v = 1:numView
    X{v} = NormalizeFea(X{v}, 0);
end

% User-specified parameters
layers = [400, 150, 40];
options = struct();
options.lambda1 = 0.00001;   % HSIC
options.lambda2 = 0.00100;   % orthogonal
options.beta = 0.1000;      % graph
options.gamma = 1.20;       % view weight
options.graph_k = 7;        % neighbors
options.maxIter = 100;
options.tol = 1e-5;

fprintf('Parameters:\n');
fprintf('  layers = [%s, %d]\n', sprintf('%d,', layers), numCluster);
fprintf('  lambda1 = %.5f\n', options.lambda1);
fprintf('  lambda2 = %.5f\n', options.lambda2);
fprintf('  beta = %.4f\n', options.beta);
fprintf('  gamma = %.2f\n', options.gamma);
fprintf('  graph_k = %d\n\n', options.graph_k);

% Run GDMFC
fprintf('Running GDMFC...\n');
tic;
[H, Z, alpha, obj_values] = GDMFC(X, numCluster, layers, options);
elapsed = toc;
fprintf('GDMFC finished in %.2f seconds.\n', elapsed);

% Clustering
H_final = H;
S = H_final * H_final';
S = (S + S') / 2;
S = max(S, 0);
predict_labels = SpectralClustering(S, numCluster);

% Evaluation
res = bestMap(y, predict_labels);
ACC = length(find(y == res)) / length(y) * 100;
NMI = MutualInfo(y, predict_labels) * 100;
Purity = compute_purity(y, predict_labels) * 100;

fprintf('\nResults on ORL with specified params:\n');
fprintf('  ACC    = %.2f%%\n', ACC);
fprintf('  NMI    = %.2f%%\n', NMI);
fprintf('  Purity = %.2f%%\n', Purity);
fprintf('  View weights: [');
for v = 1:numView
    fprintf('%.4f', alpha(v));
    if v < numView, fprintf(', '); end
end
fprintf(']\n');

% Save results
results = struct();
results.ACC = ACC;
results.NMI = NMI;
results.Purity = Purity;
results.alpha = alpha;
results.obj_values = obj_values;
results.predict_labels = predict_labels;
results.true_labels = y;
results.H_final = H_final;
results.elapsed_time = elapsed;
save('GDMFC_results_ORL_custom.mat', 'results');

% Save config
fid = fopen('best_config_ORL_custom.txt', 'w');
fprintf(fid, 'Layers: [%s, %d]\n', sprintf('%d,', layers), numCluster);
fprintf(fid, 'lambda1: %.6f\n', options.lambda1);
fprintf(fid, 'lambda2: %.6f\n', options.lambda2);
fprintf(fid, 'beta: %.6f\n', options.beta);
fprintf(fid, 'gamma: %.2f\n', options.gamma);
fprintf(fid, 'k: %d\n\n', options.graph_k);
fprintf(fid, 'Performance:\n  ACC=%.2f%%\n  NMI=%.2f%%\n  Purity=%.2f%%\n', ACC, NMI, Purity);
fclose(fid);

fprintf('\nSaved results to GDMFC_results_ORL_custom.mat and best_config_ORL_custom.txt\n');

% Optional: plot objective convergence
figure('Name', 'Objective Convergence');
plot(1:length(obj_values), obj_values, '-o');
xlabel('Iteration'); ylabel('Objective Value');
title('GDMFC Objective Convergence'); grid on;

fprintf('Run complete.\n');
