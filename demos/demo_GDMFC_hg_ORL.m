%==========================================================================
% GDMFC-Hypergraph Demo on ORL40_3_400 Dataset
%==========================================================================
% 此脚本演示如何在ORL人脸数据集上使用GDMFC_Hypergraph算法
% This script demonstrates how to use GDMFC_Hypergraph algorithm on ORL dataset
%
% ORL40_3_400数据集: 400个人脸图像, 40个类别
% ORL40_3_400 dataset: 400 face images, 40 classes
% 3个视图: 不同的图像特征表示
% 3 views: Different image feature representations
%
% 注意：这是HDDMF论文使用的标准数据集，可以直接对比性能
% Note: This is the standard dataset used in HDDMF paper for direct comparison
%==========================================================================

clear; clc; close all;

%% ==================== Experiment Metadata 实验元数据 ====================
experiment_info = struct();
experiment_info.timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
experiment_info.date = datestr(now, 'yyyy-mm-dd');
experiment_info.matlab_version = version;
experiment_info.computer = computer;
experiment_info.user = getenv('USERNAME');
experiment_info.algorithm = 'GDMFC_Hypergraph';

% Set random seed for reproducibility
rng_seed = 2024;
rng(rng_seed);
experiment_info.random_seed = rng_seed;

%% Add paths
script_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(script_dir);
addpath(genpath(fullfile(root_dir, 'core')));
addpath(genpath(fullfile(root_dir, 'utils')));
addpath(genpath(fullfile(root_dir, 'solvers')));

fprintf('========================================\n');
fprintf('GDMFC-Hypergraph Demo on ORL40_3_400 Dataset\n');
fprintf('========================================\n');
fprintf('Experiment Time: %s\n', experiment_info.timestamp);
fprintf('MATLAB Version: %s\n', experiment_info.matlab_version);
fprintf('Random Seed: %d\n', rng_seed);
fprintf('========================================\n\n');

%% ==================== Load ORL40_3_400 Dataset ====================
fprintf('Step 1: Loading ORL40_3_400 dataset...\n');

% Dataset path
dataset_name = 'ORL40_3_400';
dataPath = 'E:\research\paper\multiview\code\GDMFC\data\ORL40_3_400.mat';

if ~exist(dataPath, 'file')
    error('Dataset file not found: %s\nPlease check the path.', dataPath);
end

load(dataPath);

% Dataset info
numView = length(data);
y = label';
numCluster = length(unique(y));

% 转换为HDDMF格式：d × n (特征 × 样本)
X = cell(numView, 1);
for v = 1:numView
    X{v} = data{v};  % HDDMF格式已经是 d×n
end

numSamples = size(X{1}, 2);

% Record dataset information
experiment_info.dataset_name = dataset_name;
experiment_info.dataset_path = dataPath;
experiment_info.num_views = numView;
experiment_info.num_samples = numSamples;
experiment_info.num_clusters = numCluster;
experiment_info.num_classes = numCluster;

experiment_info.feature_dims = zeros(numView, 1);
for v = 1:numView
    experiment_info.feature_dims(v) = size(X{v}, 1);
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

%% ==================== Data Preprocessing ====================
fprintf('Step 2: Data preprocessing...\n');
fprintf('  Data normalization will be done inside GDMFC_Hypergraph (HDDMF style).\n\n');

experiment_info.preprocessing_method = 'HDDMF_column_normalization';
experiment_info.preprocessing_style = 'HDDMF';

%% ==================== Algorithm Parameters ====================
fprintf('Step 3: Setting algorithm parameters...\n');

% Layer configuration - 使用HDDMF论文的配置
% HDDMF paper uses: layers = [100 50] for ORL dataset
layers = [100, 50];  % hidden layers: 100 -> 50 (output is 40 classes)

experiment_info.layers = layers;
experiment_info.full_architecture = [experiment_info.feature_dims', layers, numCluster];

% Algorithm parameters - HDDMF defaults
options = struct();
options.beta = 0.1;          % hypergraph regularization (HDDMF默认)
options.lambda1 = 0.0001;    % diversity (HDDMF的mu参数)
options.lambda2 = 0;         % orthogonal constraint
options.gamma = 1.5;         % view weight parameter (HDDMF默认)
options.k_hyper = 5;         % k-NN for hypergraph (HDDMF默认)
options.maxIter = 100;       % maximum iterations
options.tol = 1e-5;          % convergence tolerance
options.verbose = true;      % display iteration information

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

%% ==================== Run GDMFC-Hypergraph Algorithm ====================
fprintf('Step 4: Running GDMFC-Hypergraph algorithm...\n');
fprintf('----------------------------------------\n');

tic;
[H, Z, alpha, obj_values] = GDMFC_Hypergraph(X, numCluster, layers, options);
elapsed_time = toc;

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

%% ==================== Clustering ====================
fprintf('Step 5: Performing spectral clustering...\n');

H_normalized = H ./ (sqrt(sum(H.^2, 2)) + eps);
S = H_normalized * H_normalized';
predict_labels = SpectralClustering(S, numCluster);

fprintf('  Spectral clustering completed.\n\n');

%% ==================== Evaluation ====================
fprintf('Step 6: Evaluating clustering performance...\n');

% ACC
res = bestMap(y, predict_labels);
ACC = length(find(y == res)) / length(y);

% NMI
NMI = MutualInfo(y, predict_labels);

% Purity
Purity = purity(y, predict_labels);

experiment_info.clustering_results = struct();
experiment_info.clustering_results.ACC = ACC;
experiment_info.clustering_results.NMI = NMI;
experiment_info.clustering_results.Purity = Purity;
experiment_info.clustering_results.predicted_labels = predict_labels;
experiment_info.clustering_results.true_labels = y;

fprintf('Results on %s Dataset (Hypergraph):\n', dataset_name);
fprintf('  ACC    = %.4f (%.2f%%)\n', ACC, ACC*100);
fprintf('  NMI    = %.4f (%.2f%%)\n', NMI, NMI*100);
fprintf('  Purity = %.4f (%.2f%%)\n', Purity, Purity*100);
fprintf('========================================\n\n');

%% ==================== Visualization ====================
fprintf('Step 7: Generating visualizations...\n');

figure('Position', [100, 100, 1200, 400]);

% Plot 1: Objective function convergence
subplot(1, 3, 1);
plot(1:length(obj_values), obj_values, 'b-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 12);
ylabel('Objective Value', 'FontSize', 12);
title('Convergence Curve', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Plot 2: View weights
subplot(1, 3, 2);
bar(alpha, 'FaceColor', [0.2 0.6 0.8]);
xlabel('View Index', 'FontSize', 12);
ylabel('Weight', 'FontSize', 12);
title('View Weights', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
ylim([0, 1]);

% Plot 3: Confusion matrix (simplified visualization)
subplot(1, 3, 3);
confMat = confusionmat(y, res);
imagesc(confMat);
colorbar;
xlabel('Predicted Label', 'FontSize', 12);
ylabel('True Label', 'FontSize', 12);
title(sprintf('Confusion Matrix (ACC=%.2f%%)', ACC*100), 'FontSize', 14, 'FontWeight', 'bold');
axis square;

fprintf('  Visualization completed.\n\n');

%% ==================== Save Results ====================
fprintf('Step 8: Saving comprehensive results...\n');

% Create timestamped folder
results_dir = fullfile(root_dir, 'results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

experiment_folder_name = sprintf('GDMFC_Hypergraph_%s_%s', dataset_name, experiment_info.timestamp);
experiment_folder = fullfile(results_dir, experiment_folder_name);
mkdir(experiment_folder);
fprintf('  Created experiment folder: %s\n', experiment_folder_name);

% Save results
base_filename = sprintf('GDMFC_Hypergraph_%s', dataset_name);

% 1. Save MAT file
mat_filename = fullfile(experiment_folder, [base_filename, '.mat']);
save(mat_filename, 'experiment_info', 'H', 'Z', 'alpha', 'obj_values', ...
    'predict_labels', 'ACC', 'NMI', 'Purity', 'X', 'y');
fprintf('  [1/4] MAT file saved: %s\n', [base_filename, '.mat']);

% 2. Save text report
txt_filename = fullfile(experiment_folder, [base_filename, '.txt']);
fid = fopen(txt_filename, 'w');
fprintf(fid, '========================================\n');
fprintf(fid, 'GDMFC-Hypergraph Experiment Report\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Dataset: %s\n', dataset_name);
fprintf(fid, 'Date: %s\n', experiment_info.date);
fprintf(fid, 'Time: %s\n', experiment_info.timestamp);
fprintf(fid, 'MATLAB Version: %s\n\n', experiment_info.matlab_version);

fprintf(fid, '--- Dataset Information ---\n');
fprintf(fid, 'Number of views: %d\n', numView);
fprintf(fid, 'Number of samples: %d\n', numSamples);
fprintf(fid, 'Number of classes: %d\n\n', numCluster);

fprintf(fid, '--- Algorithm Parameters ---\n');
fprintf(fid, 'Layer structure: [');
for i = 1:length(layers)
    fprintf(fid, '%d', layers(i));
    if i < length(layers), fprintf(fid, ', '); end
end
fprintf(fid, ', %d]\n', numCluster);
fprintf(fid, 'beta (hypergraph regularization): %.6f\n', options.beta);
fprintf(fid, 'lambda1 (diversity): %.6f\n', options.lambda1);
fprintf(fid, 'lambda2 (orthogonal): %.6f\n', options.lambda2);
fprintf(fid, 'gamma (view weight): %.6f\n', options.gamma);
fprintf(fid, 'k_hyper (hypergraph k-NN): %d\n', options.k_hyper);
fprintf(fid, 'maxIter: %d\n', options.maxIter);
fprintf(fid, 'tolerance: %.0e\n\n', options.tol);

fprintf(fid, '--- Training Results ---\n');
fprintf(fid, 'Elapsed time: %.2f seconds\n', elapsed_time);
fprintf(fid, 'Iterations: %d / %d\n', length(obj_values), options.maxIter);
fprintf(fid, 'Converged: %s\n', iif(experiment_info.converged, 'Yes', 'No'));
fprintf(fid, 'Final objective: %.6e\n', obj_values(end));
fprintf(fid, 'View weights: [');
for v = 1:numView
    fprintf(fid, '%.4f', alpha(v));
    if v < numView, fprintf(fid, ', '); end
end
fprintf(fid, ']\n\n');

fprintf(fid, '--- Clustering Performance ---\n');
fprintf(fid, 'ACC:    %.6f (%.2f%%)\n', ACC, ACC*100);
fprintf(fid, 'NMI:    %.6f (%.2f%%)\n', NMI, NMI*100);
fprintf(fid, 'Purity: %.6f (%.2f%%)\n', Purity, Purity*100);
fclose(fid);
fprintf('  [2/4] Text report saved: %s\n', [base_filename, '.txt']);

% 3. Save Excel file with detailed results
excel_filename = fullfile(experiment_folder, [base_filename, '.xlsx']);
results_table = table(...
    {'ACC'; 'NMI'; 'Purity'}, ...
    [ACC; NMI; Purity], ...
    'VariableNames', {'Metric', 'Value'});
writetable(results_table, excel_filename, 'Sheet', 'Performance');

params_table = table(...
    {'beta'; 'lambda1'; 'lambda2'; 'gamma'; 'k_hyper'; 'maxIter'}, ...
    [options.beta; options.lambda1; options.lambda2; options.gamma; options.k_hyper; options.maxIter], ...
    'VariableNames', {'Parameter', 'Value'});
writetable(params_table, excel_filename, 'Sheet', 'Parameters');
fprintf('  [3/4] Excel file saved: %s\n', [base_filename, '.xlsx']);

% 4. Save figure
fig_filename = fullfile(experiment_folder, [base_filename, '.png']);
saveas(gcf, fig_filename);
fprintf('  [4/4] Figure saved: %s\n', [base_filename, '.png']);

fprintf('\n========================================\n');
fprintf('All results saved to subfolder:\n');
fprintf('  %s\n\n', experiment_folder_name);

fprintf('Files in this experiment folder:\n');
fprintf('  [1] %s  (MATLAB data)\n', [base_filename, '.mat']);
fprintf('  [2] %s  (Text report)\n', [base_filename, '.txt']);
fprintf('  [3] %s  (Excel tables)\n', [base_filename, '.xlsx']);
fprintf('  [4] %s  (Figure)\n', [base_filename, '.png']);

fprintf('\nFull path: %s\n', experiment_folder);
fprintf('========================================\n\n');

%% ==================== Summary ====================
fprintf('========================================\n');
fprintf('GDMFC-Hypergraph Demo Completed Successfully!\n');
fprintf('========================================\n');
fprintf('ACC: %.2f%%, NMI: %.2f%%, Purity: %.2f%%\n\n', ACC*100, NMI*100, Purity*100);

%% Helper function for inline if-else
function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
