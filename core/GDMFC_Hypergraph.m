function [H, Z, alpha, obj_values] = GDMFC_Hypergraph(X, numCluster, layers, options)
% GDMFC_Hypergraph: Hypergraph-regularized Diversity-aware Deep Matrix Factorization for Multi-view Clustering
% 超图正则化多样性感知深度矩阵分解的多视图聚类算法
%
% Input:
%   X: cell array of multi-view data, X{v} is d_v x n matrix (HDDMF format: features x samples)
%      多视图数据的单元数组，X{v}为第v个视图的d_v x n矩阵（HDDMF格式：特征×样本）
%   numCluster: number of clusters (k)
%               聚类数目
%   layers: vector specifying the dimensions of each layer
%           每一层的维度向量，例如 [100, 50] 表示两层
%   options: struct containing algorithm parameters
%            包含算法参数的结构体
%       .beta: coefficient for hypergraph regularization (default: 1.0)
%              超图正则化系数
%       .lambda1: coefficient for HSIC diversity term (default: 0.1)
%                 HSIC多样性项系数
%       .lambda2: coefficient for orthogonal constraint (default: 0.1)
%                 正交约束系数
%       .gamma: parameter for view weight (default: 1.5, must be > 1)
%               视图权重参数
%       .k_hyper: number of neighbors for hypergraph construction (default: 5)
%                 超图构造的邻居数
%       .maxIter: maximum number of iterations (default: 100)
%                 最大迭代次数
%       .tol: convergence tolerance (default: 1e-5)
%             收敛容差
%       .verbose: display iteration information (default: true)
%                 是否显示迭代信息
%
% Output:
%   H: the final low-dimensional representation (n x k)
%      最终的低维表示
%   Z: cell array of learned matrices for each view and layer
%      每个视图每一层学习到的矩阵
%   alpha: view weights (V x 1 vector)
%          视图权重
%   obj_values: objective function values over iterations
%               目标函数值的迭代过程
%
% Reference:
%   Based on "Diverse Deep Matrix Factorization with Hypergraph Regularization"
%   and our optimization derivation in 目标函数与优化_hypergraph.md
%
% Author: Generated for GDMFC research project
% Date: 2024-12-31

% ==================== Add HDDMF reference path 添加HDDMF参考路径 ====================
current_dir = fileparts(mfilename('fullpath'));
hddmf_path = fullfile(fileparts(current_dir), 'HDDMF_reference');
if exist(hddmf_path, 'dir')
    addpath(genpath(hddmf_path));
end

% ==================== Parameter Initialization 参数初始化 ====================
numView = length(X);  % number of views 视图数量
n = size(X{1}, 2);    % number of samples (HDDMF format) 样本数量（HDDMF格式）

% Set default parameters 设置默认参数
if ~exist('options', 'var')
    options = struct();
end
if ~isfield(options, 'beta'), options.beta = 1.0; end
if ~isfield(options, 'lambda1'), options.lambda1 = 0.1; end
if ~isfield(options, 'lambda2'), options.lambda2 = 0.1; end
if ~isfield(options, 'gamma'), options.gamma = 1.5; end
if ~isfield(options, 'k_hyper'), options.k_hyper = max(5, numCluster); end
if ~isfield(options, 'maxIter'), options.maxIter = 100; end
if ~isfield(options, 'tol'), options.tol = 1e-5; end
if ~isfield(options, 'verbose'), options.verbose = true; end

beta = options.beta;
lambda1 = options.lambda1;
lambda2 = options.lambda2;
gamma = options.gamma;
k_hyper = options.k_hyper;
maxIter = options.maxIter;
tol = options.tol;
verbose = options.verbose;

if verbose
    fprintf('========================================\n');
    fprintf('GDMFC with Hypergraph Regularization\n');
    fprintf('========================================\n');
    fprintf('Number of views: %d\n', numView);
    fprintf('Number of samples: %d\n', n);
    fprintf('Number of clusters: %d\n', numCluster);
    fprintf('Parameters: beta=%.4f, lambda1=%.4f, lambda2=%.4f, gamma=%.4f\n', ...
        beta, lambda1, lambda2, gamma);
    fprintf('Hypergraph k-NN: %d\n', k_hyper);
end

% ==================== Phase 1: Data Normalization (HDDMF Style) 数据归一化 ====================
if verbose
    fprintf('\n[Phase 1] Normalizing data (HDDMF style: column normalization)...\n');
end

XX = cell(numView, 1);
for v = 1:numView
    % HDDMF方式：列归一化（每个样本归一化为单位长度）
    XX{v} = bsxfun(@rdivide, X{v}, sqrt(sum(X{v}.^2, 1)));
    if verbose
        fprintf('  View %d: %d features x %d samples\n', v, size(XX{v},1), size(XX{v},2));
    end
end

% 初始化图结构（使用PKN，超图将在迭代中动态构造）
L_graph = cell(1, numView);  % 图Laplacian (初始用PKN，后续用超图)
param_graph.k = k_hyper;

% Add the output dimension (numCluster) to the layers
% 将输出维度（聚类数）添加到层向量中
m = length(layers);  % number of layers 层数
all_layers = [layers, numCluster];  % complete layer dimensions 完整的层维度

if verbose
    fprintf('Network architecture: ');
    for i = 1:length(all_layers)
        fprintf('%d', all_layers(i));
        if i < length(all_layers)
            fprintf(' -> ');
        end
    end
    fprintf('\n');
end

% ==================== Phase 2: Layer-wise Pre-training 第二阶段：逐层预训练 ====================
if verbose
    fprintf('\n[Phase 2] Layer-wise pre-training...\n');
end

Z = cell(numView, m+1);  % Z{v, i} stores the i-th layer matrix for view v
H_pretrain = cell(numView, m+1);  % temporary storage for pre-training

% Initialize view weights uniformly 初始化视图权重为均匀分布
alpha = ones(numView, 1) / numView;

for i = 1:m+1
    if verbose
        fprintf('  Pre-training layer %d/%d...\n', i, m+1);
    end
    
    % Determine input dimension for this layer 确定当前层的输入维度
    if i == 1
        % First layer: input is original data 第一层：输入为原始数据
        for v = 1:numView
            d_output = all_layers(i);
            
            % Initialize using Semi-NMF (HDDMF格式：XX{v}是d×n，返回的H是k×n)
            [Z{v,i}, H_pretrain{v,i}, ~] = seminmf(XX{v}, d_output, 'maxiter', 100, 'verbose', 0);
        end
    else
        % Deeper layers: input is H from previous layer 深层：输入为前一层的H
        for v = 1:numView
            d_output = all_layers(i);
            
            % H from previous layer is k_{i-1} x n (HDDMF格式)
            [Z{v,i}, H_pretrain{v,i}, ~] = seminmf(H_pretrain{v,i-1}, d_output, 'maxiter', 100, 'verbose', 0);
        end
    end
end

% Initialize H from pre-training results 从预训练结果初始化H
H = cell(numView, m+1);
for v = 1:numView
    for i = 1:m+1
        H{v,i} = max(H_pretrain{v,i}, 1e-10);  % ensure non-negative and non-zero
    end
end

% 初始化图结构（使用最后一层H构造初始PKN图 - HDDMF方式）
for v = 1:numView
    H_last = H{v, m+1};  % k × n格式
    W = constructW_PKN(H_last, param_graph.k, 1);
    L_graph{v} = diag(sum(W)) - W;
end

if verbose
    fprintf('  Pre-training completed. Initial graphs constructed using PKN.\n\n');
end

% ==================== Phase 3: Joint Fine-tuning 第三阶段：联合微调 ====================
if verbose
    fprintf('[Phase 3] Joint fine-tuning with alternating optimization...\n');
end

% 标记是否是第一次迭代（HDDMF方式：第一次用PKN，之后用超图）
start_flag = 1;

obj_values = zeros(maxIter, 1);
eps_zero = 1e-10;  % small constant to avoid division by zero

for iter = 1:maxIter
    % ========== HDDMF Style: Interleaved Z and H update per layer ==========
    for v = 1:numView
        Xv = XX{v};  % d × n
        
        % Step 1: 计算H_err（反向传播）
        H_err = cell(1, m+1);
        H_err{m+1} = H{v, m+1};
        for i_layer = m:-1:1
            H_err{i_layer} = Z{v, i_layer+1} * H_err{i_layer+1};
        end
        
        % Step 2: 逐层交替更新Z和H（HDDMF核心结构）
        for i = 1:m+1
            % ===== 2.1: Update Z{i} =====
            if i == 1
                Z{v, i} = Xv * pinv(H_err{1});
                D = Z{v, 1}';  % k1 × d
            else
                Z{v, i} = pinv(D') * Xv * pinv(H_err{i});
                D = Z{v, i}' * D;  % ki × d
            end
            
            % ===== 2.2: Update H{i} (using the accumulated D) =====
            % A = D * X
            A = D * Xv;  % ki × n
            Ap = (abs(A) + A) / 2;
            An = (abs(A) - A) / 2;
            
            % B = D * D'
            B = D * D';  % ki × ki
            Bp = (abs(B) + B) / 2;
            Bn = (abs(B) - B) / 2;
            
            % Basic reconstruction update (all layers)
            H{v,i} = H{v,i} .* sqrt((Ap + Bn*H{v,i}) ./ max(An + Bp*H{v,i}, 1e-10));
            
            % Hypergraph + diversity update (only last layer)
            if i == m+1
                % Hypergraph regularization
                HmL = H{v,i} * L_graph{v};
                HmLp = (abs(HmL) + HmL) / 2;
                HmLn = (abs(HmL) - HmL) / 2;
                
                % Diversity term
                R_mat = zeros(size(H{v,i}));
                for w = 1:numView
                    if w ~= v
                        R_mat = R_mat + H{v,i} * H{w,i}' * H{w,i};
                    end
                end
                
                mu = lambda1;
                Hm_a = Ap + Bn*H{v,i} + beta*HmLn;
                Hm_b = max(An + Bp*H{v,i} + beta*HmLp, 1e-10) + mu*R_mat;
                H{v,i} = H{v,i} .* sqrt(Hm_a ./ Hm_b);
            end
            
            % Ensure non-negative
            H{v,i} = max(H{v,i}, eps_zero);
        end
    end
    
    % ========== 动态更新超图（从第二次迭代开始）==========
    if start_flag == 0
        for v = 1:numView
            H_last = H{v, m+1};
            HG = gsp_nn_hypergraph(H_last', param_graph);
            L_graph{v} = HG.L;
        end
    end
    start_flag = 0;
    
    % ========== Step 3: Update view weights alpha 更新视图权重 ==========
    R = zeros(numView, 1);
    for v = 1:numView
        % Compute reconstruction error 计算重建误差
        % XX{v}是d×n，Z是d×k，H是k×n
        if m == 0
            X_recon = Z{v,1} * H{v,1};
        else
            Phi = Z{v,1};
            for j = 2:m+1
                Phi = Phi * Z{v,j};
            end
            X_recon = Phi * H{v,m+1};
        end
        recon_err = norm(XX{v} - X_recon, 'fro')^2;
        
        % Hypergraph regularization: tr(H*L*H')
        hyper_reg = trace(H{v,m+1} * L_graph{v} * H{v,m+1}');
        
        R(v) = recon_err + beta * hyper_reg;
    end
    
    % HDDMF: alpha更新被注释掉了！保持均匀权重
    % 注意：HDDMF line 225-227显示alpha更新代码被完全注释，说明他们不更新alpha
    % 所以alpha一直保持初始的均匀分布：alpha = [1/V, 1/V, ..., 1/V]
    R = max(R, eps_zero);
    
    % Add diagnostic output for R values
    if iter <= 3 || mod(iter, 20) == 0
        fprintf('    R values: [%.4e, %.4e, %.4e]\n', R(1), R(2), R(3));
    end
    
    % 不更新alpha，保持均匀权重（HDDMF方式）
    % alpha = ones(numView, 1) / numView;  % already initialized, no update
    
    % ========== Step 4: Compute objective function 计算目标函数值 ==========
    obj = 0;
    
    % (1) Weighted reconstruction error with hypergraph regularization
    % 加权重建误差与超图正则化
    for v = 1:numView
        obj = obj + alpha(v)^gamma * R(v);
    end
    
    % (2) HSIC diversity term (negative, we want to maximize diversity)
    % HSIC多样性项（负值，我们希望最大化多样性）
    for v = 1:numView
        for w = v+1:numView
            hsic_val = computeHSIC(H{v, m+1}, H{w, m+1});
            obj = obj - lambda1 * hsic_val;
        end
    end
    
    % (3) Orthogonal constraint term
    % 正交约束项
    for v = 1:numView
        HHt = H{v, m+1} * H{v, m+1}';  % k × k
        ortho_term = norm(HHt - eye(numCluster), 'fro')^2;
        obj = obj + lambda2 * ortho_term;
    end
    
    obj_values(iter) = obj;
    
    % ========== Step 5: Check convergence 检查收敛性 ==========
    if iter > 1
        rel_change = abs(obj_values(iter) - obj_values(iter-1)) / ...
                     (abs(obj_values(iter-1)) + eps_zero);
        
        if verbose && (mod(iter, 10) == 0 || iter == 1)
            fprintf('  Iter %3d: obj = %.6f, rel_change = %.6e, alpha = [', ...
                iter, obj, rel_change);
            fprintf('%.3f ', alpha);
            fprintf(']\n');
        end
        
        if rel_change < tol
            if verbose
                fprintf('Converged at iteration %d (rel_change = %.6e < tol = %.6e)\n', ...
                    iter, rel_change, tol);
            end
            obj_values = obj_values(1:iter);
            break;
        end
    else
        if verbose
            fprintf('  Iter %3d: obj = %.6f\n', iter, obj);
        end
    end
    
    % Early stopping if objective increases significantly (divergence detection)
    % 如果目标函数显著增加则提前停止（发散检测）
    if iter > 20 && obj_values(iter) > 3 * obj_values(iter-10)
        if verbose
            fprintf('Warning: Objective function is diverging significantly. Stopping early.\n');
        end
        obj_values = obj_values(1:iter);
        break;
    end
end

% ==================== Phase 4: Construct final representation 第四阶段：构建最终表示 ====================
if verbose
    fprintf('\n[Phase 4] Constructing final representation...\n');
end

% HDDMF方式：直接求和各视图的最后一层H，然后除以视图数（平均）
% Hstar = sum_v H{v,last} / numView
H_final = zeros(numCluster, n);  % k × n格式
for v = 1:numView
    H_final = H_final + H{v, m+1};  % 直接求和，不乘alpha！
end
H_final = H_final / numView;  % HDDMF: 除以视图数！

% Normalize the final representation (按列归一化，每个样本)
H_final = max(H_final, 0);
col_norms = sqrt(sum(H_final.^2, 1)) + eps_zero;
H_final = bsxfun(@rdivide, H_final, col_norms);

% 返回时转置为n×k格式（方便后续聚类）
H = H_final';

if verbose
    fprintf('========================================\n');
    fprintf('GDMFC-Hypergraph optimization completed.\n');
    fprintf('Final view weights: [');
    fprintf('%.4f ', alpha);
    fprintf(']\n');
    fprintf('Final objective: %.6f\n', obj_values(end));
    fprintf('========================================\n');
end

end


%% ==================== Helper Function: Construct Hypergraph ====================
function [L_h, S, D_v] = constructHypergraph(X, k_hyper)
% Construct hypergraph using k-nearest neighbors
% 使用k近邻构建超图
%
% Input:
%   X: n x d data matrix
%   k_hyper: number of nearest neighbors for each hyperedge
%
% Output:
%   L_h: n x n hypergraph Laplacian (L_h = D_v - S)
%   S: n x n hypergraph affinity matrix (S = R*W*D_e^{-1}*R^T)
%   D_v: n x n vertex degree matrix (diagonal)
%
% Reference:
%   Zhou et al. "Learning with Hypergraphs: Clustering, Classification, 
%   and Embedding." NIPS 2006.

n = size(X, 1);

% Compute pairwise Euclidean distances
% 计算成对欧氏距离
D_dist = EuDist2(X, X);

% Find k nearest neighbors for each sample
% 为每个样本找k个最近邻
[sorted_dist, idx] = sort(D_dist, 2);
% idx(:,1) is the sample itself, so we take idx(:,2:k_hyper+1)
neighbors_idx = idx(:, 2:k_hyper+1);
neighbors_dist = sorted_dist(:, 2:k_hyper+1);

% Construct incidence matrix R (n x n, each row is a hyperedge)
% 构建关联矩阵R (n x n, 每行是一条超边)
% Each hyperedge e_i connects sample i and its k neighbors
R = sparse(n, n);
for i = 1:n
    R(i, i) = 1;  % include the sample itself
    R(i, neighbors_idx(i, :)) = 1;  % include its neighbors
end

% Compute hyperedge weights using Gaussian kernel
% 使用高斯核计算超边权重
% w(e_i) = exp(-mean_dist / (2*σ^2))
% This follows the standard hypergraph construction in Zhou et al. NIPS 2006

% Adaptive sigma: use median of k-th nearest neighbor distances (more robust than mean)
% 自适应sigma：使用第k个最近邻距离的中位数（比均值更鲁棒）
sigma = median(neighbors_dist(:, end));
if sigma < eps
    % Fallback: use mean of all k-NN distances
    sigma = mean(neighbors_dist(:));
    if sigma < eps
        sigma = 1;  % last resort
    end
end

% W is a diagonal matrix where W(i,i) is the weight of hyperedge e_i
W = sparse(n, n);
for i = 1:n
    % Compute mean distance to k neighbors for this hyperedge
    mean_dist = mean(neighbors_dist(i, :));
    % Gaussian kernel weight
    W(i, i) = exp(-mean_dist / (2 * sigma^2));
end

% Compute hyperedge degree matrix D_e
% 计算超边度矩阵D_e
% δ(e_j) = Σ_i r(v_i, e_j)
D_e = diag(sum(R, 1));  % sum along columns (each column is a hyperedge)

% Avoid division by zero
D_e_diag = diag(D_e);
D_e_diag(D_e_diag == 0) = 1;
D_e_inv = sparse(diag(1 ./ D_e_diag));

% Compute hypergraph affinity matrix S = R * W * D_e^{-1} * R^T
% 计算超图亲和矩阵
S = R * W * D_e_inv * R';

% Compute vertex degree matrix D_v
% 计算顶点度矩阵
% d(v_i) = Σ_j w(e_j) * r(v_i, e_j)
D_v = sparse(diag(full(sum(S, 2))));

% Compute hypergraph Laplacian L_h = D_v - S
% 计算超图拉普拉斯
L_h = D_v - S;

% Ensure symmetry and numerical stability
% 确保对称性和数值稳定性
S = (S + S') / 2;
L_h = (L_h + L_h') / 2;

end
