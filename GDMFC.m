function [H, Z, alpha, obj_values] = GDMFC(X, numCluster, layers, options)
% GDMFC: Graph-regularized Diversity-aware Deep Matrix Factorization for Multi-view Clustering
% 图正则化多样性感知深度矩阵分解的多视图聚类算法
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
%       .lambda1: coefficient for graph regularization (default: 0.1)
%                 图正则化系数
%       .lambda2: coefficient for HSIC diversity term (default: 0.01)
%                 HSIC多样性项系数
%       .beta: coefficient for co-orthogonal constraint (default: 0.1)
%              协正交约束系数
%       .gamma: parameter for view weight (default: 1.5, must be > 1)
%               视图权重参数
%       .graph_k: number of neighbors for graph construction (default: 5)
%                 图构造的邻居数
%       .maxIter: maximum number of iterations (default: 100)
%                 最大迭代次数
%       .tol: convergence tolerance (default: 1e-5)
%             收敛容差
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
%   Based on optimization derivation in goal_function.md
%   基于goal_function.md中的优化推导
%
% Author: Generated for GDMFC research project
% Date: 2024

% ==================== Parameter Initialization 参数初始化 ====================
numView = length(X);  % number of views 视图数量
n = size(X{1}, 1);    % number of samples 样本数量

% Set default parameters 设置默认参数
if ~exist('options', 'var')
    options = struct();
end
if ~isfield(options, 'lambda1'), options.lambda1 = 0.1; end
if ~isfield(options, 'lambda2'), options.lambda2 = 0.01; end
if ~isfield(options, 'beta'), options.beta = 0.1; end
if ~isfield(options, 'gamma'), options.gamma = 1.5; end
if ~isfield(options, 'graph_k'), options.graph_k = 5; end
if ~isfield(options, 'maxIter'), options.maxIter = 100; end
if ~isfield(options, 'tol'), options.tol = 1e-5; end

lambda1 = options.lambda1;
lambda2 = options.lambda2;
beta = options.beta;
gamma = options.gamma;
graph_k = options.graph_k;
maxIter = options.maxIter;
tol = options.tol;

% Construct graph Laplacian for each view 为每个视图构建图拉普拉斯矩阵
fprintf('Constructing graph Laplacian matrices...\n');
L = cell(numView, 1);
for v = 1:numView
    L{v} = constructGraphLaplacian(X{v}, graph_k);
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
    % ========== Update Z matrices (all layers, all views) 更新Z矩阵 ==========
    % 矩阵形式：X^T ≈ Z_1 * Z_2 * ... * Z_m * H_m^T
    % 其中 X: n×d_v, Z_i: d_{i-1}×d_i, H_m: n×d_m
    for v = 1:numView
        % 对于第一层 Z_1: X^T ≈ Z_1 * (Z_2*...*Z_m*H_m^T)
        % 计算右侧乘积 (从H_m开始)
        right_prod = H{v, m+1}';  % d_m × n
        for j = m+1:-1:2
            right_prod = Z{v, j} * right_prod;  % 逐层左乘
        end
        % X^T (d_v × n) ≈ Z_1 (d_v × d_1) * right_prod (d_1 × n)
        Z{v, 1} = X{v}' * pinv(right_prod);  % (d_v × n) * (n × d_1) = d_v × d_1
        
        % 对于中间层 Z_i (i=2,...,m)
        for i = 2:m
            % 左侧乘积: Z_1 * ... * Z_{i-1}
            left_prod = Z{v, 1};
            for j = 2:i-1
                left_prod = left_prod * Z{v, j};
            end
            % 右侧乘积: Z_{i+1} * ... * Z_m * H_m^T
            right_prod = H{v, m+1}';
            for j = m+1:-1:i+1
                right_prod = Z{v, j} * right_prod;
            end
            % left_prod^T * X^T ≈ Z_i * right_prod
            Z{v, i} = pinv(left_prod) * X{v}' * pinv(right_prod);
        end
        
        % 对于最后一层 Z_{m+1}: (Z_1*...*Z_m)^T * X^T ≈ Z_{m+1} * H_{m+1}^T
        if m > 0
            left_prod = Z{v, 1};
            for j = 2:m
                left_prod = left_prod * Z{v, j};
            end
            Z{v, m+1} = pinv(left_prod) * X{v}' * pinv(H{v, m+1}');
        else
            % 如果没有隐藏层，直接 X^T ≈ Z * H^T
            Z{v, 1} = X{v}' * pinv(H{v, 1}');
        end
    end
    
    % ========== Update H_m (only the last layer) 只更新最后一层H_m ==========
    % 根据推导：只有H_m需要通过乘法规则更新，中间层H通过重构得到
    % 完整更新规则包含：重构项、图正则化、HSIC多样性、协正交约束
    for v = 1:numView
        % 计算 Φ_m = Z_1 * Z_2 * ... * Z_m
        % X^T ≈ Φ_m * H_m^T，所以 X ≈ H_m * Φ_m^T
        if m == 0
            Phi_m = eye(size(X{v}, 2));
        else
            Phi_m = Z{v, 1};
            for j = 2:m+1
                Phi_m = Phi_m * Z{v, j};
            end
        end
        
        % ===== 计算所有梯度项 =====
        
        % (A) 重构项：||X - H_m * Φ_m^T||^2
        XPhi = X{v} * Phi_m;  % n × d_m，对应 (Φ_m^T * X^T)^T
        PhiTPhi = Phi_m' * Phi_m;  % d_m × d_m
        HPhiTPhi = H{v, m+1} * PhiTPhi;  % n × d_m，对应 Φ_m^T * Φ_m * H_m^T (转置后)
        
        % (B) 图正则化项：tr(H_m^T * L * H_m) 的梯度是 2*L*H_m (n×n)*(n×d_m)
        LH = L{v} * H{v, m+1};  % n × d_m
        
        % (C) HSIC多样性项：需要计算 K_{-v} = Σ_{w≠v} H * K^(w) * H
        % 其中 H 是中心化矩阵 (n×n), K^(w) 是样本核矩阵 (Gram matrix)
        % 正确的核矩阵：K^(w) = H_m^(w) * H_m^(w)^T (n×n，样本间的内积)
        n_samples = size(H{v, m+1}, 1);
        H_center = eye(n_samples) - ones(n_samples) / n_samples;  % 中心化矩阵 (n×n)
        K_minus_v = zeros(n_samples, n_samples);
        for w = 1:numView
            if w ~= v
                K_w = H{w, m+1} * H{w, m+1}';  % n × n (样本核，Gram矩阵)
                K_minus_v = K_minus_v + H_center * K_w * H_center;  % n × n
            end
        end
        KH = K_minus_v * H{v, m+1};  % (n×n) * (n×d_m) = n × d_m
        
        % (D) 协正交约束项：||H_m * H_m^T - I||^2 的梯度是 4*H_m*H_m^T*H_m - 4*H_m
        HHt = H{v, m+1} * H{v, m+1}';  % n × n
        HHtH = HHt * H{v, m+1};  % n × d_m
        
        % ===== 组合梯度为分子和分母（乘法更新规则）=====
        % 根据推导：
        % ∇^+ = 2(α^γ)[Φ_m^T*Φ_m*H_m + β*D*H_m] + 4λ2*H_m*H_m^T*H_m
        % ∇^- = 2(α^γ)[Φ_m^T*X + β*W*H_m] + 2λ1*H_m*K_{-v} + 4λ2*H_m
        
        % 分解 L = D - W
        D_mat = diag(sum(abs(L{v}), 2));  % 度矩阵 (n×n对角阵)
        W_mat = D_mat - L{v};  % 权重矩阵 (n×n)
        
        % D和W是n×n矩阵，H是n×d_m，所以是 D*H 和 W*H
        DH = D_mat * H{v, m+1};  % (n×n) * (n×d_m) = n×d_m
        WH = W_mat * H{v, m+1};  % (n×n) * (n×d_m) = n×d_m
        
        % 分子（梯度负部，促进增长）
        numer = (alpha(v)^gamma) * (XPhi + beta * WH) ...
                + lambda1 * KH ...
                + 2 * lambda2 * H{v, m+1};
        
        % 分母（梯度正部，抑制增长）
        denom = (alpha(v)^gamma) * (HPhiTPhi + beta * DH) ...
                + 2 * lambda2 * HHtH;
        
        % 数值稳定性处理
        numer = max(numer, 1e-10);
        denom = max(denom, 1e-10);
        
        % 乘法更新规则：H ← H ⊙ √(∇^- / ∇^+)
        % 添加学习率 η 进行阻尼：H ← H ⊙ (√(∇^- / ∇^+))^η
        eta = 0.5;  % 学习率/阻尼因子，防止更新过大导致发散
        update_ratio = (numer ./ denom).^(eta/2);  % 等价于 (sqrt(...))^eta
        update_ratio(isnan(update_ratio) | isinf(update_ratio)) = 1;
        
        % 限制更新幅度，防止单次更新过大
        update_ratio = min(update_ratio, 2);  % 最多增长2倍
        update_ratio = max(update_ratio, 0.5);  % 最多减小到0.5倍
        
        H{v, m+1} = H{v, m+1} .* update_ratio;
        
        % 确保H非负且不为零
        H{v, m+1} = max(H{v, m+1}, 1e-10);
        
        % 重构中间层 H_i (i < m+1): H_i^T = Z_{i+1} * H_{i+1}^T
        for i = m:-1:1
            H{v, i} = (Z{v, i+1} * H{v, i+1}')';
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
    
    % Graph regularization term (already included in R)
    % 图正则化项已包含在R中
    
    % HSIC diversity term HSIC多样性项
    for v = 1:numView
        for w = v+1:numView
            obj = obj - lambda1 * computeHSIC(H{v, m+1}, H{w, m+1});
        end
    end
    
    % Co-orthogonal constraint term 协正交约束项
    for v = 1:numView
        HHt = H{v, m+1} * H{v, m+1}';
        obj = obj + lambda2 * norm(HHt - eye(n), 'fro')^2;
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
% Average the last layer representations across all views
% 对所有视图的最后一层表示取平均
H_final = zeros(n, numCluster);
for v = 1:numView
    H_final = H_final + alpha(v) * H{v, m+1};
end

% 确保H_final非负且归一化
H_final = max(H_final, 0);
% 行归一化
H_final = bsxfun(@rdivide, H_final, sqrt(sum(H_final.^2, 2)) + 1e-10);

H = H_final;

fprintf('GDMFC optimization completed.\n');
end


%% ==================== Helper Functions 辅助函数 ====================

function HSIC_grad = computeHSIC_gradient(H1, H2)
% Compute gradient of HSIC term with respect to H1
% 计算HSIC项关于H1的梯度
%
% HSIC(H1, H2) ≈ trace(K1 * K2) where K = H * H'
% ∂HSIC/∂H1 = 4 * K2 * H1 = 4 * (H2 * H2') * H1

K2 = H2 * H2';
HSIC_grad = 4 * K2 * H1;
end
