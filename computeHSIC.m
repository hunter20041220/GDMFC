function hsic = computeHSIC(H1, H2)
% Compute Hilbert-Schmidt Independence Criterion (HSIC) between two matrices
% 计算两个矩阵之间的希尔伯特-施密特独立性准则
%
% Input:
%   H1: n x d1 matrix (first representation)
%       第一个表示矩阵
%   H2: n x d2 matrix (second representation)
%       第二个表示矩阵
%
% Output:
%   hsic: HSIC value (scalar)
%         HSIC值
%
% Reference:
%   HSIC measures the dependence between H1 and H2
%   HSIC用于衡量H1和H2之间的依赖性
%   For diversity, we want to minimize HSIC (maximize independence)
%   为了多样性，我们希望最小化HSIC（最大化独立性）
%
% Author: Generated for GDMFC research project

% Compute kernel matrices using linear kernel: K = H * H'
% 使用线性核计算核矩阵
K1 = H1 * H1';
K2 = H2 * H2';

% Center the kernel matrices 中心化核矩阵
n = size(H1, 1);
H_mat = eye(n) - ones(n) / n;  % centering matrix 中心化矩阵

K1_c = H_mat * K1 * H_mat;
K2_c = H_mat * K2 * H_mat;

% Compute HSIC 计算HSIC
% HSIC(K1, K2) = (1/n^2) * trace(K1 * K2)
% After centering: HSIC = (1/n^2) * trace(K1_c * K2_c)
hsic = trace(K1_c * K2_c) / (n^2);

end
