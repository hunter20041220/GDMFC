function D = EuDist2(X, Y, squared)
% Efficiently compute Euclidean distance matrix
% 高效计算欧氏距离矩阵
%
% Input:
%   X: n1 x d matrix
%   Y: n2 x d matrix
%   squared: whether to return squared distances (default: true)
%            是否返回平方距离
%
% Output:
%   D: n1 x n2 distance matrix
%      D(i,j) = ||X(i,:) - Y(j,:)||^2 (if squared=true)
%
% Author: Adapted from misc/EuDist2.m for GDMFC

if ~exist('squared', 'var') || isempty(squared)
    squared = true;
end

% Efficient computation using: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x'*y
% 使用高效计算公式
X_norm = sum(X.^2, 2);  % n1 x 1
Y_norm = sum(Y.^2, 2);  % n2 x 1

D = bsxfun(@plus, X_norm, Y_norm') - 2 * (X * Y');

% Numerical precision: ensure non-negative
% 数值精度处理：确保非负
D = max(D, 0);

if ~squared
    D = sqrt(D);
end

end
