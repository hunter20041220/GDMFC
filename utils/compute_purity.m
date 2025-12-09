%==========================================================================
% 聚类纯度 (Clustering Purity) 计算
%==========================================================================
% 功能描述 (Function Description):
%   计算聚类结果的纯度。纯度是聚类质量的一个简单直观的评价指标。
%   对每个簇,计算其中占多数的真实类别样本的比例,然后对所有簇取加权平均。
%
%   Calculate clustering purity. Purity is a simple and intuitive metric
%   for clustering quality. For each cluster, compute the fraction of 
%   samples from the majority true class, then take weighted average.
%
% 输入参数 (Input Parameters):
%   true_labels    - 真实标签向量 (n×1) [Ground truth labels]
%   cluster_labels - 聚类标签向量 (n×1) [Clustering labels]
%
% 输出参数 (Output Parameters):
%   purity - 聚类纯度值 [0,1] (Purity value, higher is better)
%
% 数学公式 (Mathematical Formula):
%   Purity = (1/N) * Σ_k max_j |C_k ∩ T_j|
%   其中 N是样本总数, C_k是第k个簇, T_j是第j个真实类别
%   where N is total samples, C_k is cluster k, T_j is true class j
%
% 示例 (Example):
%   true_labels = [1 1 1 2 2 2 3 3 3]';
%   cluster_labels = [1 1 2 2 2 2 3 3 3]';
%   purity = compute_purity(true_labels, cluster_labels);
%   % purity = (2 + 3 + 3) / 9 = 0.8889
%
% 参考文献 (Reference):
%   Manning et al. (2008). Introduction to Information Retrieval. 
%   Cambridge University Press.
%==========================================================================

function purity = compute_purity(true_labels, cluster_labels)

% 转换为列向量 (Convert to column vectors)
true_labels = true_labels(:);
cluster_labels = cluster_labels(:);

% 检查维度匹配 (Check dimension match)
if length(true_labels) ~= length(cluster_labels)
    error('标签向量长度必须相同! Length of label vectors must be equal!');
end

N = length(true_labels);  % 样本总数 (Total number of samples)

% 获取唯一的簇和类别 (Get unique clusters and classes)
clusters = unique(cluster_labels);
classes = unique(true_labels);

num_clusters = length(clusters);
num_classes = length(classes);

% 初始化纯度累加器 (Initialize purity accumulator)
purity_sum = 0;

% 对每个簇计算其纯度贡献 (For each cluster, compute its purity contribution)
for k = 1:num_clusters
    % 找到属于第k个簇的所有样本 (Find all samples in cluster k)
    cluster_k_indices = (cluster_labels == clusters(k));
    cluster_k_size = sum(cluster_k_indices);
    
    % 计算该簇中每个真实类别的样本数 (Count samples from each true class)
    max_class_count = 0;
    for j = 1:num_classes
        % 该簇中属于类别j的样本数 (Number of samples in cluster k from class j)
        class_j_indices = (true_labels == classes(j));
        intersection_count = sum(cluster_k_indices & class_j_indices);
        
        % 记录最大值 (Keep track of maximum)
        if intersection_count > max_class_count
            max_class_count = intersection_count;
        end
    end
    
    % 累加该簇的纯度贡献 (Add this cluster's contribution to purity)
    purity_sum = purity_sum + max_class_count;
end

% 计算最终纯度 (Compute final purity)
purity = purity_sum / N;

end
