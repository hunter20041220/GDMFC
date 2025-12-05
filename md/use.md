# GDMFC 使用说明 / GDMFC Usage Guide

## 目录 / Table of Contents
1. [简介 / Introduction](#简介--introduction)
2. [文件结构 / File Structure](#文件结构--file-structure)
3. [快速开始 / Quick Start](#快速开始--quick-start)
4. [详细说明 / Detailed Instructions](#详细说明--detailed-instructions)
5. [参数调优 / Parameter Tuning](#参数调优--parameter-tuning)
6. [输出结果 / Output Results](#输出结果--output-results)
7. [故障排除 / Troubleshooting](#故障排除--troubleshooting)

---

## 简介 / Introduction

**GDMFC** (Graph-regularized Diversity-aware Deep Matrix Factorization for Multi-view Clustering) 是一种新颖的多视图聚类算法，结合了：
- 深度矩阵分解（Deep Matrix Factorization）
- 图正则化（Graph Regularization）
- HSIC多样性约束（HSIC Diversity Constraint）
- 协正交约束（Co-orthogonal Constraint）
- 自适应视图加权（Adaptive View Weighting）

**GDMFC** is a novel multi-view clustering algorithm that combines:
- Deep Matrix Factorization
- Graph Regularization
- HSIC Diversity Constraint
- Co-orthogonal Constraint
- Adaptive View Weighting

---

## 文件结构 / File Structure

```
GDMFC/
├── demo_GDMFC.m                  # 主演示脚本 / Main demo script
├── GDMFC.m                       # 核心算法实现 / Core algorithm implementation
├── constructGraphLaplacian.m     # 图拉普拉斯构建 / Graph Laplacian construction
├── computeHSIC.m                 # HSIC计算 / HSIC computation
├── EuDist2.m                     # 欧氏距离计算 / Euclidean distance computation
├── use.md                        # 本说明文档 / This usage guide
└── GDMFC_results_WebKB.mat       # 运行结果（运行后生成）/ Results (generated after running)
```

### 文件详细说明 / Detailed File Descriptions

#### 1. `demo_GDMFC.m` - 主演示脚本
**作用 / Purpose:**
- 完整的运行流程示例，展示如何使用GDMFC算法
- 加载WebKB数据集，执行聚类，评估性能

**运行结果 / Output:**
- 控制台输出：ACC、NMI、Purity评估指标
- 可视化图表：目标函数收敛曲线、视图权重
- 保存文件：`GDMFC_results_WebKB.mat`

**如何运行 / How to Run:**
```matlab
cd('E:\research\paper\multiview\code\GDMFC')
demo_GDMFC
```

---

#### 2. `GDMFC.m` - 核心算法
**作用 / Purpose:**
- 实现完整的GDMFC优化算法
- 包含两阶段：逐层预训练 + 联合微调

**输入 / Input:**
```matlab
[H, Z, alpha, obj_values] = GDMFC(X, numCluster, layers, options)
```
- `X`: 多视图数据单元数组，`X{v}` 是第v个视图的 n×d_v 矩阵
- `numCluster`: 聚类数目 k
- `layers`: 隐藏层维度向量，例如 `[100, 50]`
- `options`: 参数结构体（见下文参数说明）

**输出 / Output:**
- `H`: 最终的低维表示 (n × k)
- `Z`: 学习到的分解矩阵（多层、多视图）
- `alpha`: 视图权重向量 (V × 1)
- `obj_values`: 目标函数值的迭代历史

**示例调用 / Example:**
```matlab
options.lambda1 = 0.1;
options.lambda2 = 0.01;
options.beta = 0.1;
options.gamma = 1.5;
[H, Z, alpha, obj_values] = GDMFC(X, 2, [100, 50], options);
```

---

#### 3. `constructGraphLaplacian.m` - 图拉普拉斯矩阵构建
**作用 / Purpose:**
- 使用k近邻方法构建图拉普拉斯矩阵
- 自动计算热核权重

**输入 / Input:**
- `X`: n × d 数据矩阵
- `k`: 近邻数量

**输出 / Output:**
- `L`: n × n 拉普拉斯矩阵 (L = D - W)

**调用示例 / Example:**
```matlab
L = constructGraphLaplacian(X{1}, 5);  % 5-NN graph
```

---

#### 4. `computeHSIC.m` - HSIC多样性计算
**作用 / Purpose:**
- 计算两个表示矩阵之间的Hilbert-Schmidt独立性准则
- 用于度量视图间的多样性

**输入 / Input:**
- `H1`, `H2`: 两个表示矩阵 (n × d)

**输出 / Output:**
- `hsic`: HSIC值（标量）

**说明 / Notes:**
- HSIC越小，表示H1和H2越独立（多样性越高）

---

#### 5. `EuDist2.m` - 欧氏距离矩阵
**作用 / Purpose:**
- 高效计算两个矩阵之间的欧氏距离

**输入 / Input:**
- `X`: n1 × d 矩阵
- `Y`: n2 × d 矩阵

**输出 / Output:**
- `D`: n1 × n2 距离矩阵

---

## 快速开始 / Quick Start

### 步骤1：准备环境 / Step 1: Prepare Environment

1. **确保MATLAB版本 / Ensure MATLAB Version:**
   - 要求：MATLAB R2024b 或更高版本
   - Required: MATLAB R2024b or later

2. **添加依赖路径 / Add Dependencies:**
   ```matlab
   % 在demo_GDMFC.m中已自动添加以下路径
   % The following paths are automatically added in demo_GDMFC.m
   addpath(genpath('../DMF_MVC/misc'));           % 评估函数
   addpath(genpath('../DMF_MVC/approx_seminmf')); % Semi-NMF
   ```

3. **检查数据集 / Check Dataset:**
   - 确保 `E:\research\paper\multiview\dataset\WebKB.mat` 存在
   - Ensure `E:\research\paper\multiview\dataset\WebKB.mat` exists

### 步骤2：运行演示 / Step 2: Run Demo

在MATLAB命令窗口执行 / Execute in MATLAB Command Window:

```matlab
cd('E:\research\paper\multiview\code\GDMFC')
demo_GDMFC
```

### 步骤3：查看结果 / Step 3: View Results

运行完成后，您将看到 / After completion, you will see:

```
========================================
Results on WebKB Dataset:
  ACC    = 0.XXXX (XX.XX%)
  NMI    = 0.XXXX
  Purity = 0.XXXX (XX.XX%)
========================================
```

同时生成两个可视化图表 / Two visualization figures will be generated:
- 左图：目标函数收敛曲线 / Objective function convergence
- 右图：学习到的视图权重 / Learned view weights

---

## 详细说明 / Detailed Instructions

### 使用自定义数据集 / Using Custom Dataset

如果要使用其他数据集（如Cornell、3Sources等），修改`demo_GDMFC.m`：

```matlab
% 修改数据路径 / Modify data path
dataPath = '../../dataset/YOUR_DATASET.mat';  % 替换为你的数据集
load(dataPath);

% 检查数据格式 / Check data format
% 必须包含 / Must contain:
%   X: 单元数组，X{v} 是第v个视图 / Cell array, X{v} is the v-th view
%   y: 真实标签向量 / Ground truth label vector

% 调整层结构 / Adjust layer structure
% 根据数据集大小和聚类数量 / Based on dataset size and cluster number
layers = [200, 100];  % 示例：两层隐藏层 / Example: two hidden layers
```

### 参数说明 / Parameter Descriptions

在`demo_GDMFC.m`的第**38-46行**设置参数 / Set parameters in lines **38-46** of `demo_GDMFC.m`:

```matlab
options = struct();
options.lambda1 = 0.1;      % 图正则化系数
options.lambda2 = 0.01;     % HSIC多样性系数
options.beta = 0.1;         % 协正交约束系数
options.gamma = 1.5;        % 视图权重参数
options.graph_k = 5;        % 图构造邻居数
options.maxIter = 100;      % 最大迭代次数
options.tol = 1e-5;         # 收敛容差
```

---

## 参数调优 / Parameter Tuning

### 关键参数指南 / Key Parameter Guide

| 参数 / Parameter | 推荐范围 / Recommended Range | 说明 / Description |
|-----------------|----------------------------|-------------------|
| `lambda1` (图正则化) | [0.01, 1.0] | 控制局部结构保持。**增大**：更强的流形约束 |
| `lambda2` (HSIC) | [0.001, 0.1] | 控制视图多样性。**增大**：鼓励更独立的视图表示 |
| `beta` (协正交) | [0.01, 1.0] | 控制表示正交性。**增大**：更强的正交约束 |
| `gamma` (权重) | (1.0, 2.0] | 视图权重非线性度。**接近1**：接近平均；**接近2**：更倾向质量高的视图 |
| `graph_k` | [5, 10] | k近邻数量。**小数据集**：k=5；**大数据集**：k=7-10 |
| `layers` | 视数据而定 | 隐藏层维度。建议：`[样本数/10, 样本数/20]` |

### 调优策略 / Tuning Strategy

#### 1. **初始设置（使用默认值）/ Initial Setup (Use Defaults)**
```matlab
options.lambda1 = 0.1;
options.lambda2 = 0.01;
options.beta = 0.1;
options.gamma = 1.5;
```

#### 2. **如果ACC较低 / If ACC is Low:**
- **增大 `lambda1`**（0.1 → 0.3）：加强局部结构保持
- **调整 `layers`**：增加第一层维度，如 `[150, 50]`
- **增大 `graph_k`**（5 → 7）：构建更密集的邻接图

#### 3. **如果视图权重不均衡 / If View Weights are Imbalanced:**
- **减小 `gamma`**（1.5 → 1.2）：使权重分布更均匀
- **增大 `lambda2`**（0.01 → 0.05）：鼓励视图多样性

#### 4. **如果收敛速度慢 / If Convergence is Slow:**
- **减小 `maxIter`**（100 → 50）并检查中间结果
- **增大 `tol`**（1e-5 → 1e-4）：放宽收敛条件

### 网格搜索示例 / Grid Search Example

```matlab
% 定义搜索空间 / Define search space
lambda1_range = [0.01, 0.1, 0.5];
lambda2_range = [0.001, 0.01, 0.05];
beta_range = [0.01, 0.1, 0.5];

best_ACC = 0;
best_params = struct();

% 网格搜索 / Grid search
for lambda1 = lambda1_range
    for lambda2 = lambda2_range
        for beta = beta_range
            options.lambda1 = lambda1;
            options.lambda2 = lambda2;
            options.beta = beta;
            
            % 运行算法 / Run algorithm
            [H, ~, ~, ~] = GDMFC(X, numCluster, layers, options);
            predict_labels = SpectralClustering(H, numCluster);
            
            % 评估 / Evaluate
            res = bestMap(y, predict_labels);
            ACC = length(find(y == res)) / length(y);
            
            % 更新最佳参数 / Update best parameters
            if ACC > best_ACC
                best_ACC = ACC;
                best_params.lambda1 = lambda1;
                best_params.lambda2 = lambda2;
                best_params.beta = beta;
            end
        end
    end
end

fprintf('Best ACC: %.4f\n', best_ACC);
disp(best_params);
```

---

## 输出结果 / Output Results

### 控制台输出 / Console Output

```
========================================
GDMFC Demo on WebKB Dataset
========================================

Step 1: Loading WebKB dataset...
  Number of views: 2
  Number of samples: 1051
  Number of clusters: 2
  View 1 dimension: 1840
  View 2 dimension: 3000

Step 2: Preprocessing data...
  Data normalized (L2 norm).

Step 3: Setting algorithm parameters...
  Layer structure: [100, 50, 2]
  lambda1 (graph reg): 0.100
  lambda2 (HSIC): 0.010
  beta (orthogonal): 0.100
  gamma (view weight): 1.50
  graph_k: 5
  maxIter: 100, tol: 1e-05

Step 4: Running GDMFC algorithm...
----------------------------------------
Constructing graph Laplacian matrices...
Phase 1: Layer-wise pre-training...
  Pre-training layer 1/3...
  Pre-training layer 2/3...
  Pre-training layer 3/3...
Phase 2: Joint fine-tuning with alternating optimization...
  Iter 1: obj = 123456.789012
  Iter 2: obj = 98765.432100, rel_change = 2.000000e-01
  ...
  Iter 25: obj = 12345.678900, rel_change = 8.765432e-06
Converged at iteration 25
GDMFC optimization completed.
----------------------------------------
  Time elapsed: 12.34 seconds
  Final view weights: [0.5234, 0.4766]

Step 5: Performing spectral clustering...
  Spectral clustering completed.

Step 6: Evaluating clustering performance...
========================================
Results on WebKB Dataset:
  ACC    = 0.7543 (75.43%)
  NMI    = 0.3210
  Purity = 0.7891 (78.91%)
========================================
```

### 保存的结果文件 / Saved Results File

运行完成后生成 `GDMFC_results_WebKB.mat`，包含 / After running, `GDMFC_results_WebKB.mat` is generated, containing:

```matlab
results.ACC              % 聚类准确率 / Clustering accuracy
results.NMI              % 归一化互信息 / Normalized mutual information
results.Purity           % 聚类纯度 / Clustering purity
results.alpha            % 学习到的视图权重 / Learned view weights
results.predict_labels   % 预测的聚类标签 / Predicted cluster labels
results.obj_values       % 目标函数迭代历史 / Objective value history
results.elapsed_time     % 运行时间（秒）/ Elapsed time (seconds)
results.options          % 使用的参数设置 / Used parameter settings
results.layers           % 层结构配置 / Layer configuration
```

加载和使用结果 / Load and use results:

```matlab
load('GDMFC_results_WebKB.mat');
fprintf('Saved ACC: %.4f\n', results.ACC);
plot(results.obj_values);  % 绘制收敛曲线 / Plot convergence curve
```

---

## 故障排除 / Troubleshooting

### 问题1：路径错误 / Issue 1: Path Error

**错误信息 / Error Message:**
```
Undefined function or variable 'NormalizeFea'.
```

**解决方法 / Solution:**
确保已添加依赖路径 / Ensure dependencies are added:
```matlab
addpath(genpath('../DMF_MVC/misc'));
addpath(genpath('../DMF_MVC/approx_seminmf'));
```

---

### 问题2：数据格式不匹配 / Issue 2: Data Format Mismatch

**错误信息 / Error Message:**
```
Index exceeds the number of array elements.
```

**解决方法 / Solution:**
检查数据格式 / Check data format:
```matlab
load('YOUR_DATASET.mat');
whos  % 查看变量信息 / View variable info

% 确保 / Ensure:
% 1. X 是 cell 数组 / X is a cell array
% 2. X{v} 是 n × d_v 矩阵 / X{v} is an n × d_v matrix
% 3. y 是 n × 1 向量 / y is an n × 1 vector
```

---

### 问题3：收敛速度慢或不收敛 / Issue 3: Slow or No Convergence

**现象 / Symptom:**
- 迭代100次仍未收敛
- 目标函数值波动

**解决方法 / Solution:**

1. **检查参数设置 / Check parameter settings:**
   ```matlab
   % 可能的问题：参数过大导致数值不稳定
   % Possible issue: Parameters too large causing numerical instability
   options.lambda1 = 0.01;  % 减小正则化系数 / Reduce regularization
   options.lambda2 = 0.001; 
   ```

2. **增加迭代次数 / Increase iterations:**
   ```matlab
   options.maxIter = 200;
   ```

3. **检查数据预处理 / Check data preprocessing:**
   ```matlab
   % 确保数据已归一化 / Ensure data is normalized
   for v = 1:numView
       X{v} = NormalizeFea(X{v}, 0);
   end
   ```

---

### 问题4：内存不足 / Issue 4: Out of Memory

**错误信息 / Error Message:**
```
Out of memory.
```

**解决方法 / Solution:**

1. **减少层维度 / Reduce layer dimensions:**
   ```matlab
   layers = [50, 25];  % 从 [100, 50] 减小 / Reduce from [100, 50]
   ```

2. **减少邻居数 / Reduce neighbor count:**
   ```matlab
   options.graph_k = 3;  % 从 5 减小 / Reduce from 5
   ```

3. **使用稀疏矩阵 / Use sparse matrices:**
   在 `constructGraphLaplacian.m` 中修改 / Modify in `constructGraphLaplacian.m`:
   ```matlab
   W = sparse(n, n);  % 使用稀疏矩阵 / Use sparse matrix
   ```

---

### 问题5：ACC/NMI/Purity都很低 / Issue 5: Low ACC/NMI/Purity

**可能原因 / Possible Causes:**

1. **参数不适合数据集 / Parameters not suitable for dataset**
   - 尝试参数网格搜索（见上文）
   - Try parameter grid search (see above)

2. **层结构不合适 / Inappropriate layer structure**
   ```matlab
   % 尝试不同的层配置 / Try different layer configurations
   layers = [200, 100];  % 更深的网络 / Deeper network
   layers = [50];        % 更浅的网络 / Shallower network
   ```

3. **数据质量问题 / Data quality issue**
   ```matlab
   % 检查数据 / Check data
   for v = 1:numView
       fprintf('View %d: %d samples, %d features\n', v, size(X{v}, 1), size(X{v}, 2));
       fprintf('  Missing values: %d\n', sum(isnan(X{v}(:))));
       fprintf('  Min: %.4f, Max: %.4f\n', min(X{v}(:)), max(X{v}(:)));
   end
   ```

---

## 联系与支持 / Contact and Support

如有问题或建议，请参考以下资源 / For questions or suggestions, refer to:

- **代码库 / Repository:** `E:\research\paper\multiview\code\GDMFC\`
- **参考论文 / Reference Papers:**
  - `optimization/Multi-View_Clustering_via_Deep_Matrix_Factorization-CN.md`
  - `optimization/Diversity-induced_Multi-view_Subspace_Clustering-CN.md`
  - `optimization/Graph_Regularized_Non-negative_Matrix_Factorization_for_Data-CN.md`
- **优化推导 / Optimization Derivation:** `optimization/goal_function.md`

---

## 引用 / Citation

如果使用本代码，请引用 / If you use this code, please cite:

```
@article{GDMFC2024,
  title={Graph-regularized Diversity-aware Deep Matrix Factorization for Multi-view Clustering},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

---

**最后更新 / Last Updated:** 2024  
**版本 / Version:** 1.0  
**作者 / Author:** Generated for GDMFC Research Project
