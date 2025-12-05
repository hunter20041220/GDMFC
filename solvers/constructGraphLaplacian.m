function L = constructGraphLaplacian(X, k)
% Construct graph Laplacian matrix using k-nearest neighbors
% 使用k近邻构建图拉普拉斯矩阵
%
% Input:
%   X: n x d data matrix (n samples, d features)
%      数据矩阵
%   k: number of nearest neighbors
%      近邻数量
%
% Output:
%   L: n x n graph Laplacian matrix (L = D - W)
%      图拉普拉斯矩阵
%
% Author: Generated for GDMFC research project

n = size(X, 1);

% Compute pairwise Euclidean distances 计算欧氏距离矩阵
D_dist = EuDist2(X, X);

% Construct k-NN graph 构建k近邻图
% For each sample, find k nearest neighbors and set edge weight
W = zeros(n, n);

for i = 1:n
    % Find k nearest neighbors (excluding itself)
    % 找到k个最近邻（不包括自己）
    [sorted_dist, idx] = sort(D_dist(i, :));
    % idx(1) is the point itself with distance 0, so take idx(2:k+1)
    neighbors = idx(2:k+1);
    
    % Use heat kernel to compute edge weights
    % 使用热核计算边权重: w_ij = exp(-||x_i - x_j||^2 / (2*sigma^2))
    sigma = mean(sorted_dist(2:k+1));  % adaptive sigma based on local density
    
    for j = neighbors
        W(i, j) = exp(-D_dist(i, j) / (2 * sigma^2));
    end
end

% Make the graph symmetric 使图对称化
W = max(W, W');

% Compute degree matrix 计算度矩阵
D_deg = diag(sum(W, 2));

% Compute Laplacian matrix 计算拉普拉斯矩阵
L = D_deg - W;

end
