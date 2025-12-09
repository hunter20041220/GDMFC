function [H, Z, alpha, obj_values] = GDMFC_improved(X, numCluster, layers, options)
% GDMFC_improved: Improved GDMFC with HDDMF graph construction method
% 改进的GDMFC：使用HDDMF的PKN图构建方法，保持GDMFC核心算法不变
%
% Input:
%   X: cell array of multi-view data, X{v} is n x d_v matrix for view v
%      多视图数据的单元数组，X{v}为第v个视图的n x d_v矩阵
%   numCluster: number of clusters (k)
%               聚类数目
%   layers: vector specifying the dimensions of each layer
%           每一层的维度向量，例如 [100, 50] 表示两层
%   options: struct containing algorithm parameters
%            包含算法参数的结构体
%       .lambda1: coefficient for HSIC diversity term (default: 0.01)
%       .lambda2: coefficient for co-orthogonal constraint (default: 0.01)
%       .beta: coefficient for graph regularization (default: 0.1)
%       .gamma: parameter for view weight (default: 1.5, must be > 1)
%       .graph_k: number of neighbors for graph construction (default: 5)
%       .maxIter: maximum number of iterations (default: 100)
%       .tol: convergence tolerance (default: 1e-5)
%       .use_PKN: use PKN method from HDDMF (default: true)
%       .use_heat_kernel: use heat kernel method from GDMFC (default: false)
%       .use_dynamic_graph: dynamically update graph in iterations (default: true)
%       .use_simple_diversity: use simple HDDMF diversity instead of HSIC (default: true)
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
% Improvements from HDDMF:
%   - Graph construction: PKN (Probabilistic k-Nearest Neighbors)
%   - Data preprocessing: Already applied in demo
%
% Core algorithm: GDMFC (unchanged)
%

% ==================== Parameter Initialization 参数初始化 ====================
numView = length(X);  % number of views 视图数量
n = size(X{1}, 1);    % number of samples 样本数量

% Set default parameters 设置默认参数
if ~exist('options', 'var')
    options = struct();
end
if ~isfield(options, 'lambda1'), options.lambda1 = 0.01; end
if ~isfield(options, 'lambda2'), options.lambda2 = 0.01; end
if ~isfield(options, 'beta'), options.beta = 0.1; end
if ~isfield(options, 'gamma'), options.gamma = 1.5; end
if ~isfield(options, 'graph_k'), options.graph_k = 5; end
if ~isfield(options, 'maxIter'), options.maxIter = 100; end
if ~isfield(options, 'tol'), options.tol = 1e-5; end
if ~isfield(options, 'use_PKN'), options.use_PKN = true; end
if ~isfield(options, 'use_heat_kernel'), options.use_heat_kernel = false; end
if ~isfield(options, 'use_dynamic_graph'), options.use_dynamic_graph = true; end
if ~isfield(options, 'use_simple_diversity'), options.use_simple_diversity = true; end

lambda1 = options.lambda1;
lambda2 = options.lambda2;
beta = options.beta;
gamma = options.gamma;
graph_k = options.graph_k;
maxIter = options.maxIter;
tol = options.tol;
use_PKN = options.use_PKN;
use_dynamic_graph = options.use_dynamic_graph;
use_simple_diversity = options.use_simple_diversity;

% Construct graph Laplacian for each view 为每个视图构建图拉普拉斯矩阵
fprintf('Constructing graph Laplacian matrices...\n');
if use_PKN
    fprintf('  Using PKN method (HDDMF style)...\n');
else
    fprintf('  Using Heat Kernel method (GDMFC style)...\n');
end

L = cell(numView, 1);
for v = 1:numView
    if use_PKN
        % Use HDDMF's PKN method 使用HDDMF的PKN方法
        L{v} = constructGraphLaplacian_PKN(X{v}, graph_k);
    else
        % Use GDMFC's heat kernel method 使用GDMFC的热核方法
        L{v} = constructGraphLaplacian(X{v}, graph_k);
    end
end

% Add the output dimension (numCluster) to the layers
% 将输出维度（聚类数）添加到层向量中
m = length(layers);  % number of layers 层数
all_layers = [layers, numCluster];  % complete layer dimensions 完整的层维度

% ==================== Phase 1: Layer-wise Pre-training 第一阶段：逐层预训练 ====================
fprintf('Phase 1: Layer-wise pre-training...\n');
Z = cell(numView, m+1);  % Z{v, i} stores the i-th layer matrix for view v
H_pretrain = cell(numView, m+1);  % temporary storage for pre-training

% Initialize view weights uniformly 初始化视图权重为均匀分布
alpha = ones(numView, 1) / numView;

for i = 1:m+1
    fprintf('  Pre-training layer %d/%d...\n', i, m+1);
    
    % Determine input dimension for this layer 确定当前层的输入维度
    if i == 1
        % First layer: input is original data 第一层：输入为原始数据
        for v = 1:numView
            d_input = size(X{v}, 2);
            d_output = all_layers(i);
            
            % Initialize using Semi-NMF 使用Semi-NMF初始化
            % Z{v,i}: d_output x d_input, H{v,i}: n x d_output
            [Z{v,i}, H_tmp] = seminmf(X{v}', d_output, 'maxiter', 50, 'verbose', 0);
            H_pretrain{v,i} = H_tmp';  % transpose to n x d_output
        end
    else
        % Deeper layers: input is H from previous layer 深层：输入为前一层的H
        for v = 1:numView
            H_prev = H_pretrain{v, i-1};  % n x d_prev
            d_input = size(H_prev, 2);
            d_output = all_layers(i);
            
            % Apply Semi-NMF to previous layer output
            % 对前一层的输出应用Semi-NMF
            [Z{v,i}, H_tmp] = seminmf(H_prev', d_output, 'maxiter', 50, 'verbose', 0);
            H_pretrain{v,i} = H_tmp';  % n x d_output
        end
    end
end

% Initialize H from pre-training results 从预训练结果初始化H
H = cell(numView, m+1);
for v = 1:numView
    for i = 1:m+1
        H{v,i} = H_pretrain{v,i};
    end
end

% ==================== Phase 2: Joint Fine-tuning 第二阶段：联合微调 ====================
fprintf('Phase 2: Joint fine-tuning with alternating optimization...\n');
obj_values = zeros(maxIter, 1);

for iter = 1:maxIter
    % ========== Dynamic Graph Update (HDDMF style) 动态图更新 ==========
    % 从第2次迭代开始，使用H_m来更新图（超图或普通图）
    if use_dynamic_graph && iter > 1
        for v = 1:numView
            if use_PKN
                % 使用PKN在H_m上构建图
                L{v} = constructGraphLaplacian_PKN(H{v, m+1}, graph_k);
            else
                % 使用热核在H_m上构建图
                L{v} = constructGraphLaplacian(H{v, m+1}, graph_k);
            end
        end
    end
    
    % ========== Update Z and H (完全HDDMF实现) ==========
    for v = 1:numView
        X_v = X{v}';  % 转为HDDMF格式: d_v × n
        
        % Step 1: 计算H_err (错误传播) - HDDMF方式
        H_err = cell(1, m+1);
        H_err{m+1} = H{v, m+1}';  % k × n (转置为HDDMF格式)
        
        % 反向传播计算H_err
        for i = m:-1:1
            H_err{i} = Z{v, i+1} * H_err{i+1};  % d_i × n
        end
        
        % Step 2: 逐层更新Z和H
        D_cumul = [];
        for i = 1:m+1
            % 2.1 更新Z{i} (HDDMF方式)
            if i == 1
                Z{v, 1} = X_v * pinv(H_err{1});  % (d_v × n) * (n × d_1) = d_v × d_1
                D_cumul = Z{v, 1}';  % d_1 × d_v
            else
                Z{v, i} = pinv(D_cumul') * X_v * pinv(H_err{i});  
                D_cumul = Z{v, i}' * D_cumul;  % d_i × d_v (累积)
            end
            
            % 2.2 更新H{i} (HDDMF方式 - 每一层都更新)
            A = D_cumul * X_v;  % d_i × n
            B = D_cumul * D_cumul';  % d_i × d_i
            
            Ap = (abs(A) + A) / 2;
            An = (abs(A) - A) / 2;
            Bp = (abs(B) + B) / 2;
            Bn = (abs(B) - B) / 2;
            
            % 获取当前层H (k_i × n格式)
            H_i = H{v, i}';
            
            % 根据HDDMF：中间层和最后一层的更新方式不同
            if i < m+1
                % 中间层：只做基础更新（不含图正则化）
                H_i = H_i .* sqrt((Ap + Bn * H_i) ./ max(An + Bp * H_i, 1e-10));
            else
                % 最后一层(i == m+1)：包含图正则化和diversity
                % 图正则化项
                HmL = H_i * L{v};
                HmLp = (abs(HmL) + HmL) / 2;
                HmLn = (abs(HmL) - HmL) / 2;
                
                % 计算分子分母（完整公式，包括图项）
                Hm_a = Ap + Bn * H_i + beta * HmLn;
                Hm_b = An + Bp * H_i + beta * HmLp;
                
                % Diversity term (HDDMF style)
                if use_simple_diversity && lambda1 > 0
                    R = zeros(size(H_i));  % k × n
                    for w = 1:numView
                        if w ~= v
                            Hw = H{w, m+1}';  % k × n
                            R = R + H_i * Hw' * Hw;  % (k×n) * (n×k) * (k×n)
                        end
                    end
                    Hm_b = Hm_b + lambda1 * R;
                end
                
                % 最终更新（一次性完成）
                H_i = H_i .* sqrt(Hm_a ./ max(Hm_b, 1e-10));
            end
            
            % 转置回n × k格式存储
            H{v, i} = H_i';
        end
    end
    
    % ========== Update view weights alpha 更新视图权重 ==========
    R = zeros(numView, 1);  % reconstruction error for each view 每个视图的重建误差
    for v = 1:numView
        % Compute reconstruction error 计算重建误差
        % X ≈ H_m * Φ_m^T，其中 Φ_m = Z_1 * Z_2 * ... * Z_m
        Phi_m = Z{v, 1};
        for j = 2:m+1
            Phi_m = Phi_m * Z{v, j};
        end
        % 重构误差
        X_recon = H{v, m+1} * Phi_m';
        err = norm(X{v} - X_recon, 'fro')^2;
        
        % 加上图正则化项
        err = err + beta * trace(H{v, m+1}' * L{v} * H{v, m+1});
        
        R(v) = err;
    end
    
    % Update alpha using closed-form solution 使用闭式解更新alpha
    % α^(v) = (R^(v))^(1/(1-γ)) / Σ_w (R^(w))^(1/(1-γ))
    % 确保R都是正数
    R = max(R, 1e-10);
    R_powered = R.^(1/(1-gamma));
    % 检查数值稳定性
    R_powered(isnan(R_powered) | isinf(R_powered)) = 1;
    alpha = R_powered / sum(R_powered);
    % 确保alpha有效
    if any(isnan(alpha)) || any(isinf(alpha))
        alpha = ones(numView, 1) / numView;  % 重置为均匀分布
    end
    
    % ========== Compute objective function 计算目标函数值 ==========
    obj = 0;
    
    % Weighted reconstruction error 加权重建误差
    for v = 1:numView
        obj = obj + alpha(v)^gamma * R(v);
    end
    
    % Diversity term (simple HDDMF style)
    if use_simple_diversity && lambda1 > 0
        for v = 1:numView
            for w = v+1:numView
                % trace(H^(v)' * H^(v) * H^(w)' * H^(w))
                div_term = trace(H{v, m+1}' * H{v, m+1} * H{w, m+1}' * H{w, m+1});
                obj = obj + lambda1 * div_term;  % 注意：diversity是惩罚项，所以加号
            end
        end
    end
    
    % Co-orthogonal constraint term 协正交约束项
    if lambda2 > 0
        for v = 1:numView
            HHt = H{v, m+1} * H{v, m+1}';
            obj = obj + lambda2 * norm(HHt - eye(n), 'fro')^2;
        end
    end
    
    obj_values(iter) = obj;
    
    % Check convergence 检查收敛性
    if iter > 1
        rel_change = abs(obj_values(iter) - obj_values(iter-1)) / (abs(obj_values(iter-1)) + 1e-10);
        fprintf('  Iter %d: obj = %.6f, rel_change = %.6e\n', iter, obj, rel_change);
        
        if rel_change < tol
            fprintf('Converged at iteration %d\n', iter);
            obj_values = obj_values(1:iter);
            break;
        end
    else
        fprintf('  Iter %d: obj = %.6f\n', iter, obj);
    end
end

% Return the final consensus representation 返回最终的一致性表示
% 完全按照HDDMF方式：所有视图最后一层的简单平均
% HDDMF: Hstar = Hstar./numOfView (Hstar是k×n格式)
H_final_kn = zeros(numCluster, n);  % k × n格式
for v = 1:numView
    H_final_kn = H_final_kn + H{v, m+1}';  % 累加k×n
end
H_final_kn = H_final_kn / numView;  % 简单平均

% 转置为n×k格式返回
H = H_final_kn';  % n × k

fprintf('GDMFC_improved optimization completed.\n');
end


%% ==================== Helper Functions 辅助函数 ====================

function L = constructGraphLaplacian_PKN(X, k)
% Construct graph Laplacian using PKN (Probabilistic k-Nearest Neighbors) method
% 使用PKN（概率k近邻）方法构建图拉普拉斯矩阵
% This is from HDDMF
%
% Input:
%   X: n x d data matrix (n samples, d features)
%   k: number of neighbors
%
% Output:
%   L: n x n graph Laplacian matrix (L = D - W)

% constructW_PKN should be in solvers path
% constructW_PKN应该在solvers路径中
% Note: constructW_PKN expects columns as samples, so we transpose
W = constructW_PKN(X', k, 1);  % X' is d x n, issymmetric=1

% Fallback implementation (should not be needed if paths are set correctly)
if isempty(W) || any(isnan(W(:))) || any(isinf(W(:)))
    n = size(X, 1);
    D_dist = EuDist2(X, X);
    W = zeros(n, n);
    
    for i = 1:n
        [sorted_dist, idx] = sort(D_dist(i, :));
        neighbors = idx(2:k+2);  % k+1 neighbors (excluding self)
        di = sorted_dist(2:k+2);
        
        % PKN weight formula: (d_{k+1} - d_j) / (k*d_{k+1} - sum(d_1 to d_k) + eps)
        d_k1 = di(k+1);
        sum_d_k = sum(di(1:k));
        for j_idx = 1:length(neighbors)
            j = neighbors(j_idx);
            if j_idx <= k
                W(i, j) = (d_k1 - di(j_idx)) / (k * d_k1 - sum_d_k + eps);
            end
        end
    end
    
    % Symmetrize
    W = (W + W') / 2;
end  % End of fallback implementation

% Compute degree matrix
D_deg = diag(sum(W, 2));

% Compute Laplacian matrix
L = D_deg - W;

end


function d = EuDist2(a, b)
% Compute squared Euclidean distance matrix
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B

if nargin < 2
    b = a;
end

if (size(a,1) == 1)
    a = [a; zeros(1,size(a,2))];
    b = [b; zeros(1,size(b,2))];
end

aa = sum(a.*a); 
bb = sum(b.*b); 
ab = a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

d = real(d);
d = max(d, 0);

end

