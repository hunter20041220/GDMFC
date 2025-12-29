function save_experiment_results(results, experiment_info, dataset_name, algorithm_name, options, layers)
%SAVE_EXPERIMENT_RESULTS 保存完整的实验结果到带时间戳的子文件夹
%
% Inputs:
%   results - 包含所有结果的结构体
%   experiment_info - 实验元数据
%   dataset_name - 数据集名称
%   algorithm_name - 算法名称 (e.g., 'GDMFC', 'GDMFC_improved')
%   options - 算法参数
%   layers - 层结构
%
% Output:
%   在 results/[algorithm_dataset_timestamp]/ 创建4个文件:
%   - .mat, .txt, .xlsx, .png
%
% Author: GDMFC Research Project
% Date: 2024

fprintf('Step: Saving comprehensive results...\n');

%% Get root directory and create results structure
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

%% Create experiment-specific subfolder
exp_folder_name = sprintf('%s_%s_%s', algorithm_name, dataset_name, experiment_info.timestamp);
results_dir = fullfile(results_base_dir, exp_folder_name);
mkdir(results_dir);
fprintf('  Created experiment folder: %s\n', exp_folder_name);

%% Generate filenames
base_filename = sprintf('%s_%s', algorithm_name, dataset_name);
mat_filename = [base_filename, '.mat'];
txt_filename = [base_filename, '.txt'];
excel_filename = [base_filename, '.xlsx'];
fig_filename = [base_filename, '.png'];

%% Extract common variables
numView = experiment_info.num_views;
numCluster = experiment_info.num_clusters;

ACC_mean = results.ACC_mean;
NMI_mean = results.NMI_mean;
Purity_mean = results.Purity_mean;

%% Save .mat file
mat_filepath = fullfile(results_dir, mat_filename);
save(mat_filepath, 'results');
fprintf('  [1/4] MAT file saved: %s\n', mat_filename);

%% Save text report
txt_filepath = fullfile(results_dir, txt_filename);
fid = fopen(txt_filepath, 'w');
fprintf(fid, '========================================\n');
fprintf(fid, '%s Experiment Report\n', algorithm_name);
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
if isfield(experiment_info, 'preprocessing_method')
    fprintf(fid, 'Preprocessing: %s', experiment_info.preprocessing_method);
    if isfield(experiment_info, 'preprocessing_style')
        fprintf(fid, ' (%s style)', experiment_info.preprocessing_style);
    end
    fprintf(fid, '\n');
end
fprintf(fid, '\n');

fprintf(fid, '=== ALGORITHM PARAMETERS ===\n');
fprintf(fid, 'Layer Structure: ');
for i = 1:length(layers)
    fprintf(fid, '%d -> ', layers(i));
end
fprintf(fid, '%d\n', numCluster);

% Print all fields in options
fn = fieldnames(options);
for i = 1:length(fn)
    val = options.(fn{i});
    if isnumeric(val) && isscalar(val)
        if val < 0.001 && val > 0
            fprintf(fid, '%s: %.6e\n', fn{i}, val);
        else
            fprintf(fid, '%s: %g\n', fn{i}, val);
        end
    elseif islogical(val)
        fprintf(fid, '%s: %d\n', fn{i}, val);
    end
end
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
fprintf(fid, 'ACC:    %.6f (%.2f%%)\n', ACC_mean, ACC_mean*100);
fprintf(fid, 'NMI:    %.6f (%.2f%%)\n', NMI_mean, NMI_mean*100);
fprintf(fid, 'Purity: %.6f (%.2f%%)\n', Purity_mean, Purity_mean*100);
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
};
if isfield(experiment_info, 'preprocessing_method')
    summary_data = [summary_data; {'Preprocessing Method', experiment_info.preprocessing_method}];
end
summary_data = [summary_data; {
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
    'Elapsed Time (s)', experiment_info.elapsed_time;
    'Iterations', experiment_info.num_iterations;
    'Max Iterations', options.maxIter;
    'Converged', iif(experiment_info.converged, 'Yes', 'No');
    'Final Objective', experiment_info.final_obj_value;
}];
writecell(summary_data, excel_filepath, 'Sheet', 'Summary');

% Sheet 2: Parameters
param_header = {'Parameter', 'Value', 'Description'};
param_data = {};
fn = fieldnames(options);
for i = 1:length(fn)
    val = options.(fn{i});
    if isnumeric(val) && isscalar(val)
        param_data = [param_data; {fn{i}, val, ''}];
    elseif islogical(val)
        param_data = [param_data; {fn{i}, val, ''}];
    end
end
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

end

%% Helper function
function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
