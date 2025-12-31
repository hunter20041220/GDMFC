# HDDMF (Diverse Deep Matrix Factorization with Hypergraph Regularization) 代码详细分析

## 一、项目概述

HDDMF (Diverse Deep Matrix Factorization with Hypergraph Regularization) 是一个用于多视图数据表示的深度矩阵分解方法。该方法发表于 IEEE/CAA Journal of Automatica Sinica (JAS) 2022。

**核心特点：**
- 深度半非负矩阵分解（Deep Semi-NMF）
- 超图正则化（Hypergraph Regularization）
- 多样性约束（Diversity Constraint）
- 多视图聚类应用

## 二、代码结构

### 2.1 主要文件

1. **demo_HDDMF.m** - 主程序入口，执行完整的实验流程
2. **DiDMF_DE_hyper.m** - 核心算法实现，包含HDDMF的主要优化逻辑
3. **evalResults_multiview.m** - 多视图聚类结果评估函数
4. **approx_seminmf/** - 半非负矩阵分解相关函数
5. **solvers/** - 求解器相关函数
6. **misc/** - 工具函数（归一化、相似度矩阵构建、评估指标等）
7. **gspbox-0.7.0/** - 图信号处理工具箱（用于超图构建）

### 2.2 数据格式

数据文件（.mat格式）应包含：
- `data`: 一个cell数组，每个元素是一个视图的特征矩阵（列向量为样本）
- `label`: 真实标签向量

示例：`ORL40_3_400.mat`
- 40个类别
- 3个视图
- 400个样本

## 三、代码运行流程

### 3.1 主程序执行步骤（demo_HDDMF.m）

```matlab
1. 初始化环境
   - 添加所有子目录到路径
   - 清空工作空间

2. 数据准备
   - 读取数据文件（.mat格式）
   - 提取视图数量、样本数量、类别数量
   - 对每个视图的特征进行列归一化（NormalizeFea）

3. 参数设置
   - layers: [100 50] - 深度网络的层配置（每层隐藏单元数）
   - mu: 0.0001 - 多样性参数
   - beta: 0.1 - 超图正则化参数

4. 执行算法（运行10次取平均）
   - 调用 DiDMF_DE_hyper() 进行矩阵分解
   - 对得到的表示矩阵 H 进行谱聚类
   - 计算评估指标（ACC, NMI, Purity, ARI, F-score, Precision, Recall）

5. 结果保存
   - 保存评估结果到 .txt 文件
   - 保存收敛误差到 .mat 文件
```

### 3.2 核心算法流程（DiDMF_DE_hyper.m）

#### 阶段1：初始化（Initialization）

**步骤1.1：数据预处理**
- 对每个视图的数据进行L2归一化（列向量单位化）
- 构建初始图拉普拉斯矩阵（使用PKN方法，k=5）

**步骤1.2：深度网络初始化**
- 对每个视图，逐层初始化深度网络：
  - 第1层：从原始数据 X 分解为 Z₁ * H₁
  - 第i层（i>1）：从前一层的 H_{i-1} 分解为 Z_i * H_i
- 使用半非负矩阵分解（seminmf）进行初始化
- 如果 k < min(m,n)，使用LPinitSemiNMF进行智能初始化
- 否则使用随机初始化

**步骤1.3：计算初始误差**
- 重构误差：||X - reconstruction(Z, H)||²_F
- 图正则化项：β * trace(H_last * L * H_last')
- 多样性项：μ * trace(H_v' * H_v * H_w' * H_w)（对所有视图对）

#### 阶段2：迭代优化（Iterative Optimization）

**外层循环：迭代次数（maxiter = 100）**

对每个视图 v：

**步骤2.1：误差反向传播**
- 从最后一层 H{v, num_layers} 开始
- 逐层向前传播：H_err{v,i} = Z{v,i+1} * H_err{v,i+1}

**步骤2.2：图更新**
- 第一次迭代：使用PKN方法构建k近邻图
- 后续迭代：使用超图方法（gsp_nn_hypergraph）构建超图拉普拉斯矩阵

**步骤2.3：更新基矩阵 Z**
- 第1层：Z{v,1} = X * pinv(H_err{v,1})
- 第i层（i>1）：Z{v,i} = pinv(D') * X * pinv(H_err{v,i})
  - 其中 D = Z{v,1}' * Z{v,2}' * ... * Z{v,i-1}'

**步骤2.4：更新编码矩阵 H（使用乘法更新规则）**

对于第i层：

1. **计算中间变量：**
   - A = D * X（D是累积的基矩阵乘积）
   - B = D * D'
   - HmL = H{v,i} * L_graph{v}（图正则化项）

2. **分离正负部分：**
   - Ap = (|A| + A) / 2, An = (|A| - A) / 2
   - Bp = (|B| + B) / 2, Bn = (|B| - B) / 2
   - HmLp = (|HmL| + HmL) / 2, HmLn = (|HmL| - HmL) / 2

3. **更新规则：**
   - 对于非最后一层：
     ```
     H{v,i} = H{v,i} .* sqrt((Ap + Bn*H{v,i}) ./ max(An + Bp*H{v,i}, 1e-10))
     ```
   
   - 对于最后一层（加入图正则化和多样性约束）：
     ```
     R = Σ_{k≠v} H{v,last} * H{k,last}' * H{k,last}  （多样性项）
     Hm_a = Ap + Bn*H{v,i} + beta*HmLn
     Hm_b = An + Bp*H{v,i} + beta*HmLp + mu*R
     H{v,i} = H{v,i} .* sqrt(Hm_a ./ max(Hm_b, 1e-10))
     ```

**步骤2.5：计算共识表示**
- Hstar = (1/numOfView) * Σ_v H{v, num_layers}
- 计算所有视图的总误差

**步骤2.6：收敛判断**
- 检查误差变化是否小于容差（tolfun = 1e-5）
- 如果收敛则提前退出

#### 阶段3：输出结果
- Z: 所有视图的所有层的基矩阵
- Hstar: 共识表示矩阵（用于聚类）
- derror: 每次迭代的误差记录

### 3.3 聚类评估流程（evalResults_multiview.m）

1. **构建相似度矩阵**
   - 使用HeatKernel方法构建k近邻图（k=5, t=1）
   - 得到邻接矩阵 A

2. **谱聚类**
   - 计算归一化拉普拉斯矩阵：L = I - D^{-1/2} * A * D^{-1/2}
   - 对L进行SVD分解，取后nClass个特征向量
   - 对特征向量进行L2归一化
   - 使用K-means聚类（maxiter=1000, replicates=20）

3. **计算评估指标**
   - ACC (Accuracy): 聚类准确率
   - NMI (Normalized Mutual Information): 归一化互信息
   - Purity: 纯度
   - ARI (Adjusted Rand Index): 调整兰德指数
   - F-score: F1分数
   - Precision: 精确率
   - Recall: 召回率

## 四、关键函数详解

### 4.1 seminmf.m - 半非负矩阵分解

**功能：** 将矩阵 X 分解为 X ≈ Z * H，其中 H 非负，Z 可正可负

**算法：**
- 初始化：H随机初始化或使用LPinitSemiNMF
- 迭代更新：
  - Z = X * pinv(H)
  - H = H .* sqrt((Ap + Bn*H) ./ max(An + Bp*H, eps))
- 收敛条件：重构误差变化小于容差

### 4.2 constructW_PKN.m - 概率k近邻图构建

**功能：** 构建概率k近邻相似度矩阵

**方法：**
- 计算样本间的L2距离
- 对每个样本，找到k+1个最近邻（包括自身）
- 使用概率公式计算权重：
  ```
  W(i,id) = (d_{k+1} - d_j) / (k*d_{k+1} - Σ_{l=1}^k d_l + eps)
  ```
- 对称化：W = (W + W') / 2

### 4.3 gsp_nn_hypergraph.m - 超图构建

**功能：** 从点云构建k近邻超图

**方法：**
- 找到每个点的k个最近邻
- 为每个点创建一个超边，包含该点及其k个邻居
- 计算超边权重：w = exp(-dist² / σ²)
- 构建超图拉普拉斯矩阵 L

### 4.4 NormalizeFea.m - 特征归一化

**功能：** 对特征矩阵进行归一化

**参数：**
- row = 0: 对列归一化（每列单位向量）
- row = 1: 对行归一化（每行单位向量）

### 4.5 SpectralClustering_GPU.m - GPU加速谱聚类

**功能：** 使用GPU加速的谱聚类算法

**步骤：**
1. 计算归一化拉普拉斯矩阵
2. SVD分解获取特征向量
3. K-means聚类

## 五、数据预处理方法分析

### 5.1 HDDMF的数据预处理流程

**当前实现：**
```matlab
% 在demo_HDDMF.m中
for v = 1:length(data)
    data{v} = NormalizeFea(data{v}, 0);  % 列归一化（L2范数）
end
```

**预处理步骤：**
1. **单一归一化方法**：只使用了`NormalizeFea(data{v}, 0)`进行列归一化
   - 将每个特征（列）归一化为单位向量
   - 公式：`X(:,j) = X(:,j) / ||X(:,j)||_2`

**特点：**
- ✅ 简单直接，计算快速
- ✅ 保证每个特征的尺度一致
- ❌ 缺少更精细的预处理步骤
- ❌ 没有考虑不同视图的特征分布差异

### 5.2 数据预处理对性能的影响

数据预处理是多视图聚类中的关键步骤，直接影响：
1. **特征尺度统一**：不同视图的特征可能具有不同的数值范围
2. **数值稳定性**：避免大数值导致的数值计算问题
3. **优化收敛性**：良好的预处理可以加速算法收敛
4. **表示质量**：预处理影响学习到的表示矩阵的质量

### 5.3 改进的数据预处理方法（参考GDMFC）

**GDMFC使用的预处理流程：**
```matlab
% 步骤1：先进行特征级预处理
X = data_guiyi_choos(X, preprocess_mode);
% preprocess_mode = 3: 列L2归一化
% preprocess_mode = 1: Min-Max归一化到[0,1]
% preprocess_mode = 2: 转置后Min-Max归一化
% preprocess_mode = 4: 列和归一化
% preprocess_mode = 5: 全局归一化

% 步骤2：再进行样本级归一化
for v = 1:numView
    X{v} = NormalizeFea(X{v}, 0);  % 列归一化（L2范数）
end
```

**双重归一化的优势：**
1. **特征级归一化**：统一不同特征的尺度
2. **样本级归一化**：进一步保证数值稳定性
3. **灵活性**：可以根据数据特性选择不同的预处理模式

## 六、HDDMF vs GDMFC 对比分析

### 6.1 性能差异分析

**性能对比（ORL数据集）：**
- **HDDMF**: ACC ≈ 86%
- **GDMFC**: ACC ≈ 89%
- **性能差距**: 约3个百分点

### 6.2 核心算法差异

| 方面 | HDDMF | GDMFC |
|------|-------|-------|
| **多样性约束** | trace(H_v' * H_v * H_w' * H_w) | HSIC(H_v, H_w) |
| **视图融合** | 简单平均：Hstar = mean(H{v}) | 加权平均：H = Σ α_v * H_v |
| **视图权重** | 无（固定权重） | 自适应学习（基于重构误差） |
| **正交约束** | 无 | 协正交约束：||H*H' - I||² |
| **图构建** | PKN + 超图 | k-NN + 热核 |

### 6.3 除核心算法外的关键增强点

#### 6.3.1 数据预处理增强

**GDMFC的预处理优势：**
1. **双重归一化**：
   ```matlab
   % 先特征级归一化
   X = data_guiyi_choos(X, 3);  % 列L2归一化
   % 再样本级归一化
   X{v} = NormalizeFea(X{v}, 0);
   ```

2. **多种归一化模式可选**：
   - Mode 1: Min-Max归一化到[0,1]
   - Mode 3: 列L2归一化（推荐）
   - Mode 4: 列和归一化

**改进建议：**
```matlab
% 在demo_HDDMF.m中添加
% 步骤1：特征级预处理
preprocess_mode = 3;  % 列L2归一化
data = data_guiyi_choos(data, preprocess_mode);

% 步骤2：样本级归一化（保持原有）
for v = 1:length(data)
    data{v} = NormalizeFea(data{v}, 0);
end
```

#### 6.3.2 视图权重学习机制

**HDDMF的问题：**
- 使用固定权重（1/numOfView）融合不同视图
- 没有考虑不同视图的质量差异

**GDMFC的解决方案：**
```matlab
% 自适应视图权重学习
% α_v = (R_v)^(1/(1-γ)) / Σ_w (R_w)^(1/(1-γ))
% 其中 R_v 是视图v的重构误差
alpha = zeros(numView, 1);
for v = 1:numView
    R(v) = reconstruction_error(v) + graph_regularization(v);
end
R_powered = R.^(1/(1-gamma));
alpha = R_powered / sum(R_powered);

% 加权融合
H_final = zeros(n, numCluster);
for v = 1:numView
    H_final = H_final + alpha(v) * H{v, last};
end
```

**改进建议：**
在`DiDMF_DE_hyper.m`中添加视图权重学习：
```matlab
% 初始化视图权重
alpha = ones(numOfView, 1) / numOfView;
gamma = 5;  % 权重参数（>1）

% 在每次迭代中更新权重
for v_ind = 1:numOfView
    % 计算视图v的重构误差
    R(v_ind) = dnorm(v_ind) + dorm_diver(v_ind);
end
R = max(R, 1e-10);
R_powered = R.^(1/(1-gamma));
alpha = R_powered / sum(R_powered);

% 加权融合
Hstar = zeros(layers(num_of_layers), N);
for v_ind = 1:numOfView
    Hstar = Hstar + alpha(v_ind) * H{v_ind, num_of_layers};
end
```

#### 6.3.3 图构建方法优化

**HDDMF的图构建：**
- 第一次迭代：PKN（概率k近邻）
- 后续迭代：超图

**GDMFC的图构建：**
- 使用k-NN + 热核（Heat Kernel）
- 自适应sigma参数：`sigma = mean(local_distances)`

**改进建议：**
```matlab
% 改进的图构建函数
function L = constructGraphLaplacian_improved(X, k)
    n = size(X, 1);
    D_dist = EuDist2(X, X);
    W = zeros(n, n);
    
    for i = 1:n
        [sorted_dist, idx] = sort(D_dist(i, :));
        neighbors = idx(2:k+1);
        
        % 自适应sigma
        sigma = mean(sorted_dist(2:k+1));
        
        % 热核权重
        for j = neighbors
            W(i, j) = exp(-D_dist(i, j) / (2 * sigma^2));
        end
    end
    
    W = max(W, W');  % 对称化
    D_deg = diag(sum(W, 2));
    L = D_deg - W;
end
```

#### 6.3.4 优化算法改进

**GDMFC的优化技巧：**

1. **学习率阻尼**：
   ```matlab
   eta = 0.5;  % 阻尼因子
   update_ratio = (numer ./ denom).^(eta/2);
   update_ratio = min(update_ratio, 2);   % 限制最大增长
   update_ratio = max(update_ratio, 0.5); % 限制最大减小
   ```

2. **数值稳定性增强**：
   ```matlab
   numer = max(numer, 1e-10);
   denom = max(denom, 1e-10);
   update_ratio(isnan(update_ratio) | isinf(update_ratio)) = 1;
   ```

3. **非负性保证**：
   ```matlab
   H{v, m+1} = max(H{v, m+1}, 1e-10);
   ```

**改进建议：**
在`DiDMF_DE_hyper.m`的H更新部分添加：
```matlab
% 添加学习率阻尼
eta = 0.5;
update_ratio = sqrt((Hm_a) ./ max(Hm_b, 1e-10));
update_ratio = update_ratio.^eta;  % 阻尼
update_ratio = min(update_ratio, 2);
update_ratio = max(update_ratio, 0.5);
H{v_ind,i} = H{v_ind,i} .* update_ratio;
```

#### 6.3.5 HSIC多样性约束 vs 简单多样性约束

**HDDMF的多样性项：**
```matlab
dorm_diver = mu * trace(H_v' * H_v * H_w' * H_w);
```
- 计算简单，但可能不够精确

**GDMFC的HSIC多样性项：**
```matlab
function hsic = computeHSIC(H1, H2)
    K1 = H1 * H1';  % 核矩阵
    K2 = H2 * H2';
    H_center = eye(n) - ones(n) / n;  % 中心化矩阵
    K1_c = H_center * K1 * H_center;
    K2_c = H_center * K2 * H_center;
    hsic = trace(K1_c * K2_c) / (n^2);
end
```
- HSIC是更严格的独立性度量
- 能够更好地衡量不同视图表示的独立性

**改进建议：**
```matlab
% 在DiDMF_DE_hyper.m中添加HSIC计算
function hsic = computeHSIC_HDDMF(H1, H2)
    n = size(H1, 1);
    K1 = H1 * H1';
    K2 = H2 * H2';
    H_center = eye(n) - ones(n) / n;
    K1_c = H_center * K1 * H_center;
    K2_c = H_center * K2 * H_center;
    hsic = trace(K1_c * K2_c) / (n^2);
end

% 在目标函数中使用HSIC
for www = 1:numOfView
    if (abs(www-v_ind)>0)
        dorm_diver_www(www) = mu * computeHSIC_HDDMF(...
            H{v_ind,num_of_layers}, H{www,num_of_layers});
    end
end
```

#### 6.3.6 协正交约束

**GDMFC的协正交约束：**
```matlab
% 目标函数中添加：λ2 * ||H*H' - I||²_F
% 鼓励H的列向量接近正交
HHt = H{v, m+1} * H{v, m+1}';
obj = obj + lambda2 * norm(HHt - eye(n), 'fro')^2;
```

**作用：**
- 提高表示的判别性
- 减少冗余信息
- 改善聚类性能

**改进建议：**
```matlab
% 在DiDMF_DE_hyper.m中添加协正交约束
lambda2 = 0.001;  % 协正交约束参数

% 在H更新规则中添加
HHt = H{v_ind,i} * H{v_ind,i}';
HHtH = HHt * H{v_ind,i};
Hm_a = Hm_a + 2 * lambda2 * H{v_ind,i};
Hm_b = Hm_b + 2 * lambda2 * HHtH;
```

### 6.4 参数调优策略

**GDMFC的参数搜索策略：**
1. **分阶段搜索**：
   - Phase 1: 搜索gamma和lambda2
   - Phase 2: 搜索beta和k
   - Phase 3: 微调lambda1

2. **关键参数发现**：
   - `gamma = 5.0` 表现最佳（远优于1.2-3.0）
   - `beta = 115-283` 范围表现好
   - `lambda2 = 0.002-0.01` 优于0.001
   - `k = 7` 表现较好

**HDDMF当前参数：**
- `mu = 0.0001`（可能过小）
- `beta = 0.1`（可能过小）
- `k = 5`（可以尝试7）

**改进建议：**
```matlab
% 尝试更大的参数范围
mu_list = [0.0001, 0.001, 0.01];
beta_list = [0.1, 1, 10, 50, 100, 115, 200];
k_list = [5, 7, 10];

% 进行网格搜索
for mu = mu_list
    for beta = beta_list
        for k = k_list
            % 运行算法并记录结果
        end
    end
end
```

### 6.5 网络结构优化

**GDMFC使用的网络结构：**
```matlab
layers = [400, 150, 40];  % 对于40类问题
```

**HDDMF当前结构：**
```matlab
layers = [100, 50];  % 可能过小
```

**改进建议：**
```matlab
% 根据类别数调整网络结构
numCluster = 40;
layers = [min(400, size(data{1},1)), ...
          min(150, size(data{1},1)/2), ...
          numCluster];
```

### 6.6 总结：关键改进点

**优先级1（高影响）：**
1. ✅ **添加双重数据预处理**（data_guiyi_choos + NormalizeFea）
2. ✅ **实现自适应视图权重学习**（基于重构误差）
3. ✅ **改进图构建方法**（k-NN + 热核，自适应sigma）

**优先级2（中等影响）：**
4. ✅ **添加学习率阻尼**（防止更新过大）
5. ✅ **增强数值稳定性**（更严格的边界检查）
6. ✅ **使用HSIC多样性约束**（替代简单trace）

**优先级3（可选）：**
7. ✅ **添加协正交约束**（提高判别性）
8. ✅ **参数网格搜索**（找到最优参数组合）
9. ✅ **调整网络结构**（根据数据规模）

## 七、参数说明

### 7.1 数据集参数

**ORL40_3_400数据集：**
- 类别数：40
- 视图数：3
- 样本数：400
- 数据格式：每个视图的特征矩阵（列向量为样本）

### 7.2 算法参数

#### 7.2.1 网络结构参数

- **layers = [100, 50]**
  - 含义：深度网络的层配置
  - 第1层：100个隐藏单元
  - 第2层：50个隐藏单元（最后一层，用于聚类）
  - 说明：层数可以根据数据复杂度调整，最后一层的维度通常设置为聚类数或略大于聚类数

#### 7.2.2 正则化参数

- **mu = 0.0001**
  - 含义：多样性参数
  - 作用：控制不同视图表示之间的多样性
  - 公式：μ * trace(H_v' * H_v * H_w' * H_w)
  - 调参建议：通常较小（0.0001-0.01），过大会导致视图表示差异过大

- **beta = 0.1**
  - 含义：超图正则化参数
  - 作用：控制图结构信息在表示学习中的重要性
  - 公式：β * trace(H_last * L * H_last')
  - 调参建议：根据数据集的图结构质量调整（0.01-1.0）

#### 7.2.3 图构建参数

- **k = 5**
  - 含义：k近邻图的邻居数
  - 作用：控制图的稀疏性和局部性
  - 调参建议：通常5-15，过小会导致图过于稀疏，过大会失去局部性

#### 7.2.4 优化参数

- **maxiter = 100**
  - 含义：最大迭代次数
  - 作用：控制算法的运行时间
  - 调参建议：根据收敛速度调整，通常50-200

- **tolfun = 1e-5**
  - 含义：收敛容差
  - 作用：判断算法是否收敛
  - 调参建议：通常1e-4到1e-6

#### 5.2.5 评估参数

- **运行次数 = 10**
  - 含义：重复运行次数
  - 作用：减少随机性影响，获得稳定的统计结果
  - 输出：均值±标准差

### 5.3 其他数据集参数设置建议

对于不同的数据集，建议的参数设置：

1. **小规模数据集（样本数<1000）**
   - layers: [50, 30] 或 [30, 20]
   - mu: 0.001
   - beta: 0.1-0.5
   - k: 5-10

2. **中等规模数据集（1000-5000样本）**
   - layers: [100, 50] 或 [200, 100]
   - mu: 0.0001-0.001
   - beta: 0.1
   - k: 5-10

3. **大规模数据集（>5000样本）**
   - layers: [200, 100] 或更大
   - mu: 0.0001
   - beta: 0.05-0.1
   - k: 10-15

## 八、代码执行步骤详解

### 步骤1：环境准备

```matlab
addpath(genpath('.'));  % 添加所有子目录
clear;                   % 清空工作空间
```

### 步骤2：数据加载与预处理

```matlab
% 加载数据
load('data/ORL40_3_400.mat');

% 提取信息
C = length(unique(label));  % 类别数
numOfView = length(data);    % 视图数
numOfSample = size(data{1}, 2);  % 样本数

% 归一化每个视图
for v = 1:length(data)
    data{v} = NormalizeFea(data{v}, 0);  % 列归一化
end
```

### 步骤3：参数设置

```matlab
layers = [100 50];  % 网络层配置
mu = 0.0001;        % 多样性参数
beta = 0.1;         % 超图正则化参数
```

### 步骤4：算法执行（循环10次）

```matlab
for i = 1:10
    % 执行HDDMF算法
    [Z, H, dnorm] = DiDMF_DE_hyper(data, layers, ...
        'gnd', gnd, 'beta', beta, 'mu', mu);
    
    % 评估聚类结果
    [acc1(i), nmii1(i), pur1(i), ari1(i), ...
     f1(i), pre1(i), rec1(i)] = ...
        evalResults_multiview(H, gnd);
end
```

### 步骤5：结果统计与保存

```matlab
% 计算均值和标准差
acc1m = mean(acc1); acc1s = std(acc1);
nmii1m = mean(nmii1); nmii1s = std(nmii1);
% ... 其他指标类似

% 格式化输出
eva_spe = [acc1m,acc1s,nmii1m,nmii1s,...]*100;
eva_spe = roundn(eva_spe,-2);

% 保存结果
dlmwrite(Tname, eva_spe, '-append', ...);
save(objectname, 'dnorm');
```

## 九、算法数学原理

### 9.1 目标函数

HDDMF的目标函数为：

```
min_{Z_v, H_v} Σ_v [||X_v - reconstruction(Z_v, H_v)||²_F 
                  + β * trace(H_v^{last} * L_v * (H_v^{last})')
                  + μ * Σ_{w≠v} trace((H_v^{last})' * H_v^{last} * (H_w^{last})' * H_w^{last})]
```

其中：
- 第一项：重构误差，保证数据重构质量
- 第二项：超图正则化项，保持数据的图结构信息
- 第三项：多样性项，鼓励不同视图学习到不同的表示

### 9.2 深度分解结构

对于每个视图v，深度分解为：

```
X_v ≈ Z_v^1 * Z_v^2 * ... * Z_v^L * H_v^L
```

其中：
- Z_v^i: 第i层的基矩阵（可正可负）
- H_v^L: 最后一层的编码矩阵（非负）

### 9.3 更新规则推导

使用乘法更新规则（Multiplicative Update Rules），保证H的非负性：

对于H的更新：
```
H = H .* sqrt((∇H_positive) ./ max(∇H_negative, eps))
```

其中∇H_positive和∇H_negative分别是目标函数对H的梯度分解后的正负部分。

## 十、关键实现细节

### 10.1 数值稳定性

- 使用`max(..., 1e-10)`避免除零错误
- 使用`pinv()`（伪逆）代替`inv()`提高数值稳定性
- L2归一化防止数值溢出

### 10.2 GPU加速

- `SpectralClustering_GPU.m`使用GPU加速谱聚类
- 需要MATLAB的Parallel Computing Toolbox

### 10.3 内存优化

- 使用稀疏矩阵存储图结构
- 分批处理大规模数据（在constructW中实现）

### 10.4 初始化策略

- 使用LPinitSemiNMF进行智能初始化（当k < min(m,n)时）
- 保证初始解的质量，加速收敛

## 十一、常见问题与调试

### 11.1 收敛问题

**现象：** 算法不收敛或收敛很慢

**解决方案：**
1. 检查参数设置是否合理（mu和beta不能过大）
2. 增加最大迭代次数
3. 调整学习率（如果实现了自适应学习率）
4. 检查数据预处理是否正确

### 11.2 内存不足

**现象：** 大规模数据集运行时内存溢出

**解决方案：**
1. 减少网络层数或隐藏单元数
2. 使用稀疏矩阵存储
3. 分批处理数据

### 11.3 聚类结果不稳定

**现象：** 多次运行结果差异较大

**解决方案：**
1. 增加运行次数（如20次或更多）
2. 固定随机种子
3. 调整K-means的replicates参数

### 11.4 性能提升建议

基于与GDMFC的对比分析，以下改进可以显著提升性能：

1. **立即实施（预期提升2-3%）**：
   - 添加双重数据预处理
   - 实现自适应视图权重
   - 改进图构建方法

2. **中期优化（预期提升1-2%）**：
   - 添加学习率阻尼
   - 使用HSIC多样性约束
   - 参数网格搜索

3. **长期改进（预期提升0.5-1%）**：
   - 添加协正交约束
   - 优化网络结构
   - 增强数值稳定性

## 十二、扩展与改进建议

### 12.1 可能的改进方向

1. **自适应参数学习**
   - 自动学习mu和beta参数
   - 根据数据特性调整网络结构

2. **更高效的优化算法**
   - 使用ADMM或交替方向乘数法
   - 实现自适应学习率

3. **处理缺失视图**
   - 扩展算法处理不完整的多视图数据

4. **在线学习**
   - 支持增量学习新样本

### 12.2 代码优化建议

1. **并行化**
   - 对不同视图的处理可以并行化
   - 使用parfor循环加速

2. **缓存机制**
   - 缓存中间计算结果
   - 避免重复计算

3. **可视化**
   - 添加收敛曲线可视化
   - 可视化学习到的表示

## 十三、参考文献

1. Huang, H., Zhou, G., Liang, N., Zhao, Q., & Xie, S. (2022). Diverse Deep Matrix Factorization with Hypergraph Regularization for Multi-view Data Representation. IEEE/CAA Journal of Automatica Sinica.

2. Zhao, H., et al. (2017). Multi-View Clustering via Deep Matrix Factorization. AAAI.

3. Trigeorgis, G., et al. (2014). A Deep Semi-NMF Model for Learning Hidden Representations. ICML.

## 十四、快速实施指南：从86%提升到89%+

基于与GDMFC的对比分析，以下是按优先级排序的改进实施步骤：

### 14.1 优先级1：立即实施（预期提升2-3%）

#### 改进1：双重数据预处理

**文件：** `demo_HDDMF.m`

**修改位置：** 数据加载后，NormalizeFea之前

**代码：**
```matlab
% 添加data_guiyi_choos函数（从GDMFC复制或创建）
function data = data_guiyi_choos(data, data_g)
    for v = 1:length(data)
        X = data{v};
        switch data_g
            case 3  % 列L2归一化（推荐）
                norms = sqrt(sum(X.^2, 1));
                norms(norms == 0) = 1;
                X = X * diag(1./norms);
            case 1  % Min-Max归一化
                X = mapminmax(X, 0, 1);
            otherwise
                error('Unknown mode');
        end
        data{v} = X;
    end
end

% 在demo_HDDMF.m中修改
load(dataf);
C = length(unique(label));

% === 添加双重预处理 ===
preprocess_mode = 3;  % 列L2归一化
data = data_guiyi_choos(data, preprocess_mode);

% 保持原有的NormalizeFea
for v = 1:length(data)
    data{v} = NormalizeFea(data{v}, 0);
end
```

#### 改进2：自适应视图权重学习

**文件：** `DiDMF_DE_hyper.m`

**修改位置：** 在计算Hstar之前（约第200行）

**代码：**
```matlab
% 在函数开始处添加gamma参数
dflts  = {0, 0, 1, 1, 100, 1e-5, 1, 1, 0, 0, 0.1, 0.0001, 5.0};  % 添加gamma默认值
[z0, h0, bUpdateH, bUpdateLastH, maxiter, tolfun, verbose, bUpdateZ, cache, gnd, beta, mu, gamma] = ...
    internal.stats.parseArgs(pnames,dflts,varargin{:});

% 初始化视图权重
alpha = ones(numOfView, 1) / numOfView;

% 在每次迭代的末尾（计算Hstar之前）添加：
% ========== 更新视图权重 ==========
R = zeros(numOfView, 1);
for v_ind = 1:numOfView
    R(v_ind) = dnorm_all(v_ind);
end
R = max(R, 1e-10);
R_powered = R.^(1/(1-gamma));
alpha = R_powered / sum(R_powered);

% 修改Hstar的计算（约第200行）
Hstar = zeros(layers(num_of_layers), N);
for v_ind = 1:numOfView
    Hstar = Hstar + alpha(v_ind) * H{v_ind, num_of_layers};
end
% 删除原来的：Hstar = Hstar./numOfView;
```

**调用修改：**
```matlab
% 在demo_HDDMF.m中
[Z, H, dnorm] = DiDMF_DE_hyper(data, layers, 'gnd', gnd, ...
    'beta', beta, 'mu', mu, 'gamma', 5.0);  % 添加gamma参数
```

#### 改进3：改进图构建方法

**文件：** 创建新文件 `solvers/constructGraphLaplacian_improved.m`

**代码：**
```matlab
function L = constructGraphLaplacian_improved(X, k)
    n = size(X, 1);
    D_dist = EuDist2(X, X);
    W = zeros(n, n);
    
    for i = 1:n
        [sorted_dist, idx] = sort(D_dist(i, :));
        neighbors = idx(2:k+1);
        
        % 自适应sigma
        sigma = mean(sorted_dist(2:k+1));
        
        % 热核权重
        for j = neighbors
            W(i, j) = exp(-D_dist(i, j) / (2 * sigma^2));
        end
    end
    
    W = max(W, W');
    D_deg = diag(sum(W, 2));
    L = D_deg - W;
end
```

**修改DiDMF_DE_hyper.m：**
```matlab
% 在第一次迭代时（约第121行）
if start==1
    Weight{v_ind} = constructGraphLaplacian_improved(H_last', param.k);
    Diag_tmp = diag(sum(Weight{v_ind}));
    L_graph{v_ind} = Diag_tmp - Weight{v_ind};
end
```

### 14.2 优先级2：中期优化（预期提升1-2%）

#### 改进4：添加学习率阻尼

**文件：** `DiDMF_DE_hyper.m`

**修改位置：** H更新规则（约第193行）

**代码：**
```matlab
% 在H更新前添加
eta = 0.5;  % 学习率阻尼因子

% 修改更新规则
Hm_a = (Ap + Bn* H{v_ind,i} + beta*(HmLn));
Hm_b = (max(An + Bp* H{v_ind,i} + beta*(HmLp), 1e-10));

% 添加阻尼
update_ratio = sqrt((Hm_a) ./ max(Hm_b + mu*R, 1e-10));
update_ratio = update_ratio.^eta;
update_ratio = min(update_ratio, 2);
update_ratio = max(update_ratio, 0.5);
update_ratio(isnan(update_ratio) | isinf(update_ratio)) = 1;

H{v_ind,i} = H{v_ind,i} .* update_ratio;
```

#### 改进5：使用HSIC多样性约束

**文件：** `DiDMF_DE_hyper.m`

**添加函数：**
```matlab
function hsic = computeHSIC_HDDMF(H1, H2)
    n = size(H1, 1);
    K1 = H1 * H1';
    K2 = H2 * H2';
    H_center = eye(n) - ones(n) / n;
    K1_c = H_center * K1 * H_center;
    K2_c = H_center * K2 * H_center;
    hsic = trace(K1_c * K2_c) / (n^2);
end
```

**修改多样性项计算（约第89行和第211行）：**
```matlab
for www = 1:numOfView
    if (abs(www-v_ind)>0)
        dorm_diver_www(www) = mu * computeHSIC_HDDMF(...
            H{v_ind,num_of_layers}, H{www,num_of_layers});
    end
end
```

### 14.3 优先级3：可选优化（预期提升0.5-1%）

#### 改进6：参数网格搜索

**文件：** 创建新文件 `demo_HDDMF_gridsearch.m`

**代码框架：**
```matlab
% 参数搜索空间
mu_list = [0.0001, 0.001, 0.01];
beta_list = [0.1, 1, 10, 50, 100, 115, 200];
k_list = [5, 7, 10];
gamma_list = [3.0, 4.0, 5.0, 6.0];

best_acc = 0;
best_params = struct();

for mu = mu_list
    for beta = beta_list
        for k = k_list
            for gamma = gamma_list
                % 运行算法
                [Z, H, dnorm] = DiDMF_DE_hyper(data, layers, ...
                    'gnd', gnd, 'beta', beta, 'mu', mu, 'gamma', gamma);
                [acc, nmi, ~] = evalResults_multiview(H, gnd);
                
                if acc > best_acc
                    best_acc = acc;
                    best_params = struct('mu', mu, 'beta', beta, ...
                        'k', k, 'gamma', gamma);
                end
            end
        end
    end
end
```

### 14.4 实施检查清单

- [ ] **步骤1**：添加`data_guiyi_choos.m`函数
- [ ] **步骤2**：修改`demo_HDDMF.m`添加双重预处理
- [ ] **步骤3**：修改`DiDMF_DE_hyper.m`添加gamma参数
- [ ] **步骤4**：实现视图权重学习机制
- [ ] **步骤5**：创建改进的图构建函数
- [ ] **步骤6**：添加学习率阻尼
- [ ] **步骤7**：实现HSIC多样性约束
- [ ] **步骤8**：运行参数网格搜索
- [ ] **步骤9**：验证性能提升

### 14.5 预期性能提升

| 改进项 | 预期提升 | 实施难度 | 优先级 |
|--------|---------|---------|--------|
| 双重数据预处理 | +1.0-1.5% | 低 | P1 |
| 自适应视图权重 | +0.8-1.2% | 中 | P1 |
| 改进图构建 | +0.5-0.8% | 低 | P1 |
| 学习率阻尼 | +0.3-0.5% | 低 | P2 |
| HSIC多样性 | +0.3-0.5% | 中 | P2 |
| 参数调优 | +0.5-1.0% | 高 | P3 |
| **总计** | **+3.4-5.5%** | - | - |

**目标：** 从86%提升到89-91%+

## 十五、总结

HDDMF是一个结合了深度学习、图正则化和多样性约束的多视图聚类方法。代码实现完整，包含了从数据预处理、算法优化到结果评估的完整流程。通过深度矩阵分解学习数据的层次表示，利用超图正则化保持数据的局部结构，并通过多样性约束确保不同视图学习到互补的信息。

**核心优势：**
1. 深度结构能够学习数据的层次特征
2. 超图正则化能够捕获高阶关系
3. 多样性约束保证多视图的互补性
4. 非负约束使得表示具有可解释性

**适用场景：**
- 多视图聚类任务
- 需要保持数据局部结构的场景
- 需要学习层次表示的场景

---

*本文档由AI助手基于代码分析生成，如有疑问请参考源代码或联系作者。*

