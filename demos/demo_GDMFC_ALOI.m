%% GDMFC Demo for ALOI Dataset
% Dataset: ALOI (Amsterdam Library of Object Images)
% - 10800 samples
% - 4 views with dimensions: [77, 13, 64, 125]
% - 100 classes
%
% This demo performs multi-view clustering using GDMFC algorithm
% and saves comprehensive results including metadata, parameters,
% and performance metrics.
%
% Author: GDMFC Research Project
% Date: 2024

clear; clc; close all;

%% Experiment Metadata
fprintf('========================================\n');
fprintf('GDMFC Experiment - ALOI Dataset\n');
fprintf('========================================\n\n');

experiment_info = struct();
experiment_info.timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
experiment_info.matlab_version = version;
experiment_info.computer = computer;
experiment_info.random_seed = 2024;

fprintf('Experiment Timestamp: %s\n', experiment_info.timestamp);
fprintf('MATLAB Version: %s\n', experiment_info.matlab_version);
fprintf('Random Seed: %d\n\n', experiment_info.random_seed);

% Set random seed for reproducibility
rng(experiment_info.random_seed);

%% Load Dataset
fprintf('Step 1: Loading ALOI dataset...\n');
dataset_path = 'E:\research\paper\multiview\dataset\ALOI.mat';
load(dataset_path);

numView = length(X);
numCluster = length(unique(y));
num_samples = size(X{1}, 1);

fprintf('  Number of views: %d\n', numView);
fprintf('  Number of samples: %d\n', num_samples);
fprintf('  Number of clusters: %d\n', numCluster);
fprintf('  Feature dimensions: ');
feature_dims = zeros(1, numView);
for v = 1:numView
    feature_dims(v) = size(X{v}, 2);
    fprintf('%d', feature_dims(v));
    if v < numView, fprintf(', '); end
end
fprintf('\n\n');

% Store dataset information
experiment_info.dataset_name = 'ALOI';
experiment_info.num_views = numView;
experiment_info.num_samples = num_samples;
experiment_info.num_clusters = numCluster;
experiment_info.feature_dims = feature_dims;

%% Preprocessing
fprintf('Step 2: Preprocessing data...\n');
for v = 1:numView
    X{v} = NormalizeFea(X{v}, 0);  % L2 normalization, column-wise
end
experiment_info.preprocessing_method = 'L2 normalization (NormalizeFea, column-wise)';
fprintf('  Applied L2 normalization to all views\n\n');

%% Network Architecture
fprintf('Step 3: Setting up network architecture...\n');
layers = [200, 100];  % hidden layers: 200 -> 100 (output layer is numCluster=100)

full_architecture = [feature_dims, layers, numCluster];
fprintf('  Network architecture: ');
for i = 1:length(full_architecture)
    fprintf('%d', full_architecture(i));
    if i < length(full_architecture), fprintf(' -> '); end
end
fprintf('\n');
fprintf('  Hidden layers: [%s]\n', num2str(layers));
fprintf('  Output dimension: %d (number of clusters)\n\n', numCluster);

experiment_info.layers = layers;
experiment_info.full_architecture = full_architecture;

%% Algorithm Parameters
fprintf('Step 4: Configuring algorithm parameters...\n');
options = struct();
options.alpha = 10;           % Weight for autoencoder reconstruction
options.lambda1 = 0.001;      % Diversity regularization
options.lambda2 = 0.01;       % Orthogonal constraint
options.beta = 100;           % Graph regularization (reduced for large-scale)
options.gamma = 1.2;          % View weight exponent
options.graph_k = 7;          % Number of neighbors for graph construction
options.maxIter = 50;         % Maximum iterations (reduced for large-scale)
options.mu = 1e-5;            % Convergence threshold (relaxed for faster convergence)
options.verbose = true;       % Display progress

fprintf('  alpha (reconstruction): %g\n', options.alpha);
fprintf('  lambda1 (diversity): %g\n', options.lambda1);
fprintf('  lambda2 (orthogonal): %g\n', options.lambda2);
fprintf('  beta (graph regularization): %g\n', options.beta);
fprintf('  gamma (view weight exponent): %g\n', options.gamma);
fprintf('  graph_k (neighbors): %d\n', options.graph_k);
fprintf('  maxIter: %d\n', options.maxIter);
fprintf('  convergence threshold: %g\n\n', options.mu);

experiment_info.options = options;

%% Run GDMFC Algorithm
fprintf('Step 5: Running GDMFC algorithm...\n');
fprintf('========================================\n');
tic;
[H_normalized, alpha, obj_values] = GDMFC(X, layers, numCluster, options);
elapsed_time = toc;
fprintf('========================================\n');
fprintf('Training completed in %.4f seconds\n', elapsed_time);
fprintf('Iterations: %d / %d\n', length(obj_values), options.maxIter);
fprintf('Converged: %s\n', iif(length(obj_values) < options.maxIter, 'Yes', 'No'));
fprintf('Final objective value: %.6e\n\n', obj_values(end));

% Store training results
experiment_info.elapsed_time = elapsed_time;
experiment_info.num_iterations = length(obj_values);
experiment_info.converged = (length(obj_values) < options.maxIter);
experiment_info.final_obj_value = obj_values(end);
experiment_info.obj_values = obj_values;
experiment_info.view_weights = alpha;

fprintf('View weights: ');
for v = 1:numView
    fprintf('%.6f', alpha(v));
    if v < numView, fprintf(', '); end
end
fprintf('\n\n');

%% Clustering
fprintf('Step 6: Performing spectral clustering...\n');
S_consensus = H_normalized * H_normalized';
result = SpectralClustering(S_consensus, numCluster);
fprintf('  Clustering completed\n\n');

%% Evaluation
fprintf('Step 7: Evaluating clustering performance...\n');
ACC = Accuracy(result, double(y));
[~, NMI, ~] = compute_nmi(double(y), result);
Purity = purity(result, double(y));

fprintf('  ACC:    %.6f (%.2f%%)\n', ACC, ACC*100);
fprintf('  NMI:    %.6f (%.2f%%)\n', NMI, NMI*100);
fprintf('  Purity: %.6f (%.2f%%)\n\n', Purity, Purity*100);

%% Prepare Results Structure
results = struct();
results.predicted_labels = result;
results.true_labels = double(y);
results.ACC_mean = ACC;
results.NMI_mean = NMI;
results.Purity_mean = Purity;
results.H_normalized = H_normalized;
results.S_consensus = S_consensus;
results.view_weights = alpha;
results.obj_values = obj_values;

%% Visualization
fprintf('Step 8: Generating convergence plot...\n');
figure('Position', [100, 100, 800, 500]);

subplot(1, 2, 1);
plot(1:length(obj_values), obj_values, 'b-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 12);
ylabel('Objective Value', 'FontSize', 12);
title('GDMFC Convergence Curve - ALOI', 'FontSize', 14);
grid on;

subplot(1, 2, 2);
bar(alpha);
xlabel('View Index', 'FontSize', 12);
ylabel('Weight', 'FontSize', 12);
title('View Weights', 'FontSize', 14);
grid on;
set(gca, 'XTick', 1:numView);

fprintf('  Visualization completed\n\n');

%% Save Results
% Get root directory
script_dir = pwd;
if contains(script_dir, 'demos')
    root_dir = fileparts(script_dir);
else
    root_dir = script_dir;
end

results_base_dir = fullfile(root_dir, 'results');
if ~exist(results_base_dir, 'dir')
    mkdir(results_base_dir);
end

% Create experiment-specific subfolder
dataset_name = 'ALOI';
exp_folder_name = sprintf('GDMFC_%s_%s', dataset_name, experiment_info.timestamp);
results_dir = fullfile(results_base_dir, exp_folder_name);
mkdir(results_dir);

fprintf('Step 9: Saving results to timestamped subfolder...\n');
fprintf('  Folder: %s\n\n', exp_folder_name);

% Generate filenames
base_filename = sprintf('GDMFC_%s', dataset_name);
mat_filename = [base_filename, '.mat'];
txt_filename = [base_filename, '.txt'];
excel_filename = [base_filename, '.xlsx'];
fig_filename = [base_filename, '.png'];

%% Save .mat file
mat_filepath = fullfile(results_dir, mat_filename);
save(mat_filepath, 'results');
fprintf('  [1/4] MAT file saved: %s\n', mat_filename);

%% Save text report
txt_filepath = fullfile(results_dir, txt_filename);
fid = fopen(txt_filepath, 'w');
fprintf(fid, '========================================\n');
fprintf(fid, 'GDMFC Experiment Report - ALOI Dataset\n');
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
for v = 1:numView
    fprintf(fid, '%d', experiment_info.feature_dims(v));
    if v < numView, fprintf(fid, ', '); end
end
fprintf(fid, '\n');
fprintf(fid, 'Preprocessing: %s\n', experiment_info.preprocessing_method);
fprintf(fid, '\n');

fprintf(fid, '=== ALGORITHM PARAMETERS ===\n');
fprintf(fid, 'Layer Structure: ');
for v = 1:numView
    fprintf(fid, '%d -> ', experiment_info.feature_dims(v));
end
for i = 1:length(layers)
    fprintf(fid, '%d -> ', layers(i));
end
fprintf(fid, '%d\n', numCluster);
fprintf(fid, 'Hidden Layers: [%s]\n', num2str(layers));
fprintf(fid, 'alpha: %g\n', options.alpha);
fprintf(fid, 'lambda1: %g\n', options.lambda1);
fprintf(fid, 'lambda2: %g\n', options.lambda2);
fprintf(fid, 'beta: %g\n', options.beta);
fprintf(fid, 'gamma: %g\n', options.gamma);
fprintf(fid, 'graph_k: %d\n', options.graph_k);
fprintf(fid, 'maxIter: %d\n', options.maxIter);
fprintf(fid, 'mu: %.6e\n', options.mu);
fprintf(fid, '\n');

fprintf(fid, '=== TRAINING RESULTS ===\n');
fprintf(fid, 'Elapsed Time: %.4f seconds\n', experiment_info.elapsed_time);
fprintf(fid, 'Iterations: %d / %d\n', experiment_info.num_iterations, options.maxIter);
fprintf(fid, 'Converged: %s\n', iif(experiment_info.converged, 'Yes', 'No'));
fprintf(fid, 'Final Objective: %.6e\n', experiment_info.final_obj_value);
fprintf(fid, 'View Weights: ');
for v = 1:numView
    fprintf(fid, '%.6f', experiment_info.view_weights(v));
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

%% Save Excel file
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
    'Elapsed Time (s)', experiment_info.elapsed_time;
    'Iterations', experiment_info.num_iterations;
    'Max Iterations', options.maxIter;
    'Converged', iif(experiment_info.converged, 'Yes', 'No');
    'Final Objective', experiment_info.final_obj_value;
};
writecell(summary_data, excel_filepath, 'Sheet', 'Summary');

% Sheet 2: Parameters
param_header = {'Parameter', 'Value', 'Description'};
param_data = {
    'alpha', options.alpha, 'Reconstruction weight';
    'lambda1', options.lambda1, 'Diversity regularization';
    'lambda2', options.lambda2, 'Orthogonal constraint';
    'beta', options.beta, 'Graph regularization';
    'gamma', options.gamma, 'View weight exponent';
    'graph_k', options.graph_k, 'Number of neighbors';
    'maxIter', options.maxIter, 'Maximum iterations';
    'mu', options.mu, 'Convergence threshold';
};
writecell([param_header; param_data], excel_filepath, 'Sheet', 'Parameters');

% Sheet 3: Architecture
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

% Sheet 4: View Weights
view_weights_header = {'View', 'Weight'};
view_weights_data = cell(numView, 2);
for v = 1:numView
    view_weights_data{v, 1} = sprintf('View %d', v);
    view_weights_data{v, 2} = experiment_info.view_weights(v);
end
writecell([view_weights_header; view_weights_data], excel_filepath, 'Sheet', 'ViewWeights');

% Sheet 5: Convergence
conv_header = {'Iteration', 'Objective Value'};
conv_data = cell(length(experiment_info.obj_values), 2);
for i = 1:length(experiment_info.obj_values)
    conv_data{i, 1} = i;
    conv_data{i, 2} = experiment_info.obj_values(i);
end
writecell([conv_header; conv_data], excel_filepath, 'Sheet', 'Convergence');

fprintf('  [3/4] Excel file saved: %s\n', excel_filename);

%% Save figure
fig_filepath = fullfile(results_dir, fig_filename);
saveas(gcf, fig_filepath);
fprintf('  [4/4] Figure saved: %s\n', fig_filename);

%% Summary
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

fprintf('Experiment completed successfully!\n');
fprintf('ACC: %.2f%%, NMI: %.2f%%, Purity: %.2f%%\n', ACC*100, NMI*100, Purity*100);

%% Helper function
function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
