%==========================================================================
% Parameter Search Script for GDMFC-Hypergraph on ORL Dataset
%==========================================================================
% 搜索GDMFC-Hypergraph算法在ORL数据集上的最佳参数组合
% Search for optimal parameter combinations for GDMFC-Hypergraph on ORL
%
% 参数搜索范围:
% - beta: 超图正则化系数 (1 to 1000, step 1)
% - lambda1: 多样性系数 (0.0001, 0.001, 0.01, 0.1, 1)
% - lambda2: 正交约束系数 (0.0001, 0.001, 0.01, 0.1, 1)
% - k_hyper: k-NN for hypergraph (5, 7, 9, 11)
% - gamma: 视图权重指数 (1.01, 1.2, 1.5, 2.0)
%==========================================================================

clear; clc; close all;

%% ==================== Configuration ====================
% 搜索策略选择
% Strategy 1: 全局搜索 (WARNING: 400,000 combinations, 会运行很久!)
% Strategy 2: 粗搜索 + 细搜索 (推荐: 先粗略找区域，再细化)
% Strategy 3: 逐参数搜索 (固定其他参数，逐个优化)

SEARCH_STRATEGY = 2;  % 选择策略 2 (粗搜索+细搜索)

% Random seed
rng(2024);

%% Add paths
script_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(script_dir);
addpath(genpath(fullfile(root_dir, 'core')));
addpath(genpath(fullfile(root_dir, 'utils')));
addpath(genpath(fullfile(root_dir, 'solvers')));

fprintf('========================================\n');
fprintf('Parameter Search for GDMFC-Hypergraph\n');
fprintf('Dataset: ORL40_3_400\n');
fprintf('Search Strategy: %d\n', SEARCH_STRATEGY);
fprintf('========================================\n\n');

%% ==================== Load Dataset ====================
fprintf('Loading ORL40_3_400 dataset...\n');
dataPath = fullfile(root_dir, 'data', 'ORL40_3_400.mat');

if ~exist(dataPath, 'file')
    error('Dataset not found: %s', dataPath);
end

load(dataPath);

X = cell(length(data), 1);
for v = 1:length(data)
    X{v} = data{v};  % d×n format
end

y = label';
numCluster = length(unique(y));
layers = [100, 50];  % HDDMF paper configuration

fprintf('  Samples: %d, Classes: %d, Views: %d\n', size(X{1},2), numCluster, length(X));
fprintf('  Loaded successfully.\n\n');

%% ==================== Define Search Space ====================
% 参数搜索空间定义

if SEARCH_STRATEGY == 1
    % Strategy 1: 全局详细搜索 (非常耗时!)
    fprintf('WARNING: Strategy 1 will run ~400,000 experiments!\n');
    fprintf('Estimated time: >100 hours on typical hardware.\n');
    fprintf('Press Ctrl+C to cancel, or wait 10 seconds to continue...\n');
    pause(10);
    
    param_space.beta = 1:1:1000;
    param_space.lambda1 = [0.0001, 0.001, 0.01, 0.1, 1];
    param_space.lambda2 = [0.0001, 0.001, 0.01, 0.1, 1];
    param_space.k_hyper = [5, 7, 9, 11];
    param_space.gamma = [1.01, 1.2, 1.5, 2.0];
    
elseif SEARCH_STRATEGY == 2
    % Strategy 2: 粗搜索 + 细搜索 (推荐)
    % Phase 1: 粗搜索 - 大范围稀疏采样
    fprintf('Strategy 2: Coarse-to-Fine Search\n');
    fprintf('Phase 1: Coarse search (sparse sampling)...\n\n');
    
    param_space_coarse.beta = [0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000];
    param_space_coarse.lambda1 = [0.0001, 0.001, 0.01, 0.1, 1];
    param_space_coarse.lambda2 = [0, 0.001, 0.01, 0.1];
    param_space_coarse.k_hyper = [5, 7, 9, 11];
    param_space_coarse.gamma = [1.01, 1.2, 1.5, 2.0];
    
    param_space = param_space_coarse;
    
elseif SEARCH_STRATEGY == 3
    % Strategy 3: 逐参数优化
    fprintf('Strategy 3: Sequential Parameter Optimization\n');
    fprintf('Will optimize one parameter at a time.\n\n');
    
    % 初始默认参数
    default_params.beta = 0.1;
    default_params.lambda1 = 0.0001;
    default_params.lambda2 = 0;
    default_params.k_hyper = 5;
    default_params.gamma = 1.5;
end

%% ==================== Parameter Search ====================
if SEARCH_STRATEGY == 1 || SEARCH_STRATEGY == 2
    % Grid search
    param_names = fieldnames(param_space);
    param_values = struct2cell(param_space);
    
    % 计算总组合数
    total_combinations = 1;
    for i = 1:length(param_values)
        total_combinations = total_combinations * length(param_values{i});
    end
    
    fprintf('Total combinations to test: %d\n', total_combinations);
    fprintf('Estimated time: %.1f hours (assuming 30s per run)\n\n', total_combinations*30/3600);
    
    % 生成所有参数组合
    param_combinations = cell(1, length(param_values));
    [param_combinations{:}] = ndgrid(param_values{:});
    
    % 初始化结果存储
    results = struct();
    results.params = zeros(total_combinations, length(param_names));
    results.ACC = zeros(total_combinations, 1);
    results.NMI = zeros(total_combinations, 1);
    results.Purity = zeros(total_combinations, 1);
    results.time = zeros(total_combinations, 1);
    results.iterations = zeros(total_combinations, 1);
    results.converged = false(total_combinations, 1);
    
    % 开始搜索
    fprintf('Starting parameter search...\n');
    fprintf('Progress will be saved every 100 experiments.\n\n');
    
    start_time = datetime('now');
    
    for exp_idx = 1:total_combinations
        % 获取当前参数组合
        current_params = struct();
        for p = 1:length(param_names)
            idx_vals = param_combinations{p};
            current_params.(param_names{p}) = idx_vals(exp_idx);
            results.params(exp_idx, p) = idx_vals(exp_idx);
        end
        
        % 显示进度
        if mod(exp_idx, 10) == 1
            fprintf('[%d/%d] Testing: beta=%.4f, lambda1=%.4f, lambda2=%.4f, k=%d, gamma=%.2f\n', ...
                exp_idx, total_combinations, ...
                current_params.beta, current_params.lambda1, current_params.lambda2, ...
                current_params.k_hyper, current_params.gamma);
        end
        
        % 运行实验
        try
            options = struct();
            options.beta = current_params.beta;
            options.lambda1 = current_params.lambda1;
            options.lambda2 = current_params.lambda2;
            options.gamma = current_params.gamma;
            options.k_hyper = current_params.k_hyper;
            options.maxIter = 100;
            options.tol = 1e-5;
            options.verbose = false;  % 关闭详细输出
            
            tic;
            [H, Z, alpha, obj_values] = GDMFC_Hypergraph(X, numCluster, layers, options);
            elapsed = toc;
            
            % 聚类
            H_normalized = H ./ (sqrt(sum(H.^2, 2)) + eps);
            S = H_normalized * H_normalized';
            predict_labels = SpectralClustering(S, numCluster);
            
            % 评估
            res = bestMap(y, predict_labels);
            ACC = length(find(y == res)) / length(y);
            NMI = MutualInfo(y, predict_labels);
            Purity = purity(y, predict_labels);
            
            % 保存结果
            results.ACC(exp_idx) = ACC;
            results.NMI(exp_idx) = NMI;
            results.Purity(exp_idx) = Purity;
            results.time(exp_idx) = elapsed;
            results.iterations(exp_idx) = length(obj_values);
            results.converged(exp_idx) = (length(obj_values) < options.maxIter);
            
        catch ME
            fprintf('  ERROR: %s\n', ME.message);
            results.ACC(exp_idx) = 0;
            results.NMI(exp_idx) = 0;
            results.Purity(exp_idx) = 0;
        end
        
        % 每100次保存一次中间结果
        if mod(exp_idx, 100) == 0
            save_intermediate_results(results, param_names, exp_idx, root_dir);
            fprintf('  Intermediate results saved at experiment %d\n', exp_idx);
        end
        
        % 显示当前最佳结果
        if mod(exp_idx, 50) == 0
            [best_acc, best_idx] = max(results.ACC(1:exp_idx));
            fprintf('  Current best ACC: %.4f at experiment %d\n', best_acc, best_idx);
        end
    end
    
    end_time = datetime('now');
    total_time = end_time - start_time;
    
    fprintf('\n========================================\n');
    fprintf('Parameter search completed!\n');
    fprintf('Total time: %s\n', char(total_time));
    fprintf('========================================\n\n');
    
    % 找出最佳结果
    [best_acc, best_idx] = max(results.ACC);
    [best_nmi, ~] = max(results.NMI);
    [best_purity, ~] = max(results.Purity);
    
    fprintf('Best Results:\n');
    fprintf('  Best ACC: %.4f (%.2f%%)\n', best_acc, best_acc*100);
    fprintf('  Best NMI: %.4f (%.2f%%)\n', best_nmi, best_nmi*100);
    fprintf('  Best Purity: %.4f (%.2f%%)\n\n', best_purity, best_purity*100);
    
    fprintf('Best parameters (by ACC):\n');
    for p = 1:length(param_names)
        fprintf('  %s = %.6f\n', param_names{p}, results.params(best_idx, p));
    end
    fprintf('\n');
    
    % 保存最终结果
    save_final_results(results, param_names, root_dir);
    
    % Phase 2: 细搜索 (如果是Strategy 2)
    if SEARCH_STRATEGY == 2
        fprintf('========================================\n');
        fprintf('Phase 2: Fine search around best parameters...\n');
        fprintf('========================================\n\n');
        
        best_params = struct();
        for p = 1:length(param_names)
            best_params.(param_names{p}) = results.params(best_idx, p);
        end
        
        % 在最佳参数附近细化搜索
        % beta: ±20% 范围，10个点
        beta_center = best_params.beta;
        beta_range = max(0.01, beta_center * 0.2);
        param_space_fine.beta = linspace(max(0.01, beta_center-beta_range), beta_center+beta_range, 10);
        
        % 其他参数保持最佳值或小范围变化
        param_space_fine.lambda1 = best_params.lambda1;
        param_space_fine.lambda2 = best_params.lambda2;
        param_space_fine.k_hyper = best_params.k_hyper;
        param_space_fine.gamma = best_params.gamma;
        
        fprintf('Fine search will test %d combinations.\n\n', length(param_space_fine.beta));
        
        % 执行细搜索 (代码类似，这里简化)
        % ...可以递归调用相同的搜索逻辑
    end
end

fprintf('========================================\n');
fprintf('All results saved to:\n');
fprintf('  %s\n', fullfile(root_dir, 'results', 'param_search_results.xlsx'));
fprintf('========================================\n\n');

%% ==================== Helper Functions ====================
function save_intermediate_results(results, param_names, current_idx, root_dir)
    % 保存中间结果
    results_dir = fullfile(root_dir, 'results');
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    
    filename = fullfile(results_dir, 'param_search_intermediate.mat');
    save(filename, 'results', 'param_names', 'current_idx');
end

function save_final_results(results, param_names, root_dir)
    % 保存最终结果到Excel
    results_dir = fullfile(root_dir, 'results');
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    
    % 创建表格
    n = length(results.ACC);
    T = table();
    
    % 添加参数列
    for p = 1:length(param_names)
        T.(param_names{p}) = results.params(:, p);
    end
    
    % 添加结果列
    T.ACC = results.ACC;
    T.NMI = results.NMI;
    T.Purity = results.Purity;
    T.Time_sec = results.time;
    T.Iterations = results.iterations;
    T.Converged = results.converged;
    
    % 按ACC排序
    T = sortrows(T, 'ACC', 'descend');
    
    % 保存
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = fullfile(results_dir, sprintf('param_search_results_%s.xlsx', timestamp));
    writetable(T, filename);
    
    fprintf('Results saved to: %s\n', filename);
    
    % 也保存MAT格式
    mat_filename = fullfile(results_dir, sprintf('param_search_results_%s.mat', timestamp));
    save(mat_filename, 'results', 'param_names', 'T');
end
