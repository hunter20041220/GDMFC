# GDMFC - Graph-regularized Diversity-aware Deep Matrix Factorization for Multi-view Clustering

## 概述 / Overview

GDMFC是一种新颖的多视图聚类算法，融合了深度矩阵分解、图正则化、HSIC多样性约束和协正交约束。

GDMFC is a novel multi-view clustering algorithm that integrates deep matrix factorization, graph regularization, HSIC diversity constraints, and co-orthogonal constraints.

## 算法特点 / Key Features

✅ **深度架构** - 逐层学习层次化表示  
✅ **图正则化** - 保持数据局部流形结构  
✅ **多样性约束** - 使用HSIC鼓励不同视图学习互补信息  
✅ **协正交性** - 增强表示的判别能力  
✅ **自适应加权** - 自动学习视图重要性权重  

## 快速开始 / Quick Start

```matlab
% 在MATLAB中运行 / Run in MATLAB
cd('E:\research\paper\multiview\code\GDMFC')
demo_GDMFC
```

**预期输出 / Expected Output:**
```
Results on WebKB Dataset:
  ACC    = 0.XXXX (XX.XX%)
  NMI    = 0.XXXX
  Purity = 0.XXXX (XX.XX%)
```

## 文件说明 / Files

| 文件 / File | 说明 / Description |
|------------|-------------------|
| `demo_GDMFC.m` | 主演示脚本，展示完整流程 / Main demo script |
| `GDMFC.m` | 核心算法实现 / Core algorithm implementation |
| `constructGraphLaplacian.m` | 图拉普拉斯构建 / Graph Laplacian construction |
| `computeHSIC.m` | HSIC多样性计算 / HSIC diversity computation |
| `EuDist2.m` | 欧氏距离计算 / Euclidean distance |
| `use.md` | 详细使用说明 / Detailed usage guide |

## 依赖 / Dependencies

- MATLAB R2024b或更高版本 / MATLAB R2024b or later
- 辅助函数（已包含在`../DMF_MVC/`中）:
  - `NormalizeFea.m` - 数据归一化
  - `SpectralClustering.m` - 谱聚类
  - `bestMap.m`, `compute_nmi.m` - 评估指标
  - `seminmf.m` - Semi-NMF预训练

## 参数设置 / Parameters

主要参数及推荐范围 / Key parameters and recommended ranges:

```matlab
options.lambda1 = 0.1;      % 图正则化 [0.01, 1.0]
options.lambda2 = 0.01;     % HSIC多样性 [0.001, 0.1]
options.beta = 0.1;         % 协正交约束 [0.01, 1.0]
options.gamma = 1.5;        % 视图权重 (1.0, 2.0]
options.graph_k = 5;        % k近邻 [5, 10]
```

详细参数调优指南见 `use.md` / See `use.md` for detailed tuning guide

## 数据格式 / Data Format

输入数据必须包含 / Input data must contain:
- `X`: 单元数组，`X{v}` 是第v个视图 (n × d_v) / Cell array, X{v} is v-th view
- `y`: 真实标签向量 (n × 1) / Ground truth labels

示例 / Example:
```matlab
load('WebKB.mat');  % 加载数据 / Load data
% X{1}: 1051 × 1840 (Anchor view)
% X{2}: 1051 × 3000 (Content view)
% y: 1051 × 1 (labels: 1 or 2)
```

## 输出结果 / Output Results

1. **控制台输出 / Console Output:**
   - ACC (聚类准确率)
   - NMI (归一化互信息)
   - Purity (聚类纯度)
   - 视图权重

2. **保存文件 / Saved File:**
   - `GDMFC_results_WebKB.mat` - 包含所有结果和参数

3. **可视化 / Visualization:**
   - 目标函数收敛曲线
   - 学习到的视图权重

## 算法流程 / Algorithm Workflow

```
1. 数据加载与预处理 / Data loading & preprocessing
   ↓
2. 图拉普拉斯构建 / Graph Laplacian construction
   ↓
3. 阶段1：逐层预训练 / Phase 1: Layer-wise pre-training
   - 使用Semi-NMF初始化每一层
   ↓
4. 阶段2：联合微调 / Phase 2: Joint fine-tuning
   - 更新Z矩阵（伪逆）
   - 更新H矩阵（乘性规则）
   - 更新视图权重α
   ↓
5. 谱聚类 / Spectral clustering
   ↓
6. 性能评估 / Performance evaluation
```

## 理论基础 / Theoretical Foundation

目标函数 / Objective Function:

$$\min_{\{Z_i^{(v)}\}, \{H_i^{(v)}\}, \alpha} \sum_{v=1}^V (\alpha^{(v)})^\gamma \left( \sum_{i=1}^{m+1} \| X_i^{(v)} - H_i^{(v)} (Z_i^{(v)})^T \|_F^2 \right) + \lambda_1 \sum_{v=1}^V \text{tr}(H_{m+1}^{(v)T} L^{(v)} H_{m+1}^{(v)}) + \lambda_2 \sum_{v<w} \text{HSIC}(H_{m+1}^{(v)}, H_{m+1}^{(w)}) + \beta \sum_{v=1}^V \| H_{m+1}^{(v)} H_{m+1}^{(v)T} - I \|_F^2$$

优化方法：交替最小化（Alternating Minimization）

详细推导见 / Detailed derivation: `../optimization/goal_function.md`

## 使用示例 / Usage Examples

### 基本用法 / Basic Usage

```matlab
% 加载数据 / Load data
load('../../dataset/WebKB.mat');

% 设置参数 / Set parameters
options.lambda1 = 0.1;
options.lambda2 = 0.01;
options.beta = 0.1;
options.gamma = 1.5;

% 运行算法 / Run algorithm
[H, Z, alpha, obj_values] = GDMFC(X, 2, [100, 50], options);

% 聚类 / Clustering
predict_labels = SpectralClustering(H, 2);

% 评估 / Evaluation
res = bestMap(y, predict_labels);
ACC = length(find(y == res)) / length(y);
NMI = compute_nmi(y, predict_labels);
```

### 参数网格搜索 / Parameter Grid Search

```matlab
best_ACC = 0;
for lambda1 = [0.01, 0.1, 0.5]
    for lambda2 = [0.001, 0.01, 0.05]
        options.lambda1 = lambda1;
        options.lambda2 = lambda2;
        
        [H, ~, ~, ~] = GDMFC(X, numCluster, layers, options);
        predict_labels = SpectralClustering(H, numCluster);
        
        res = bestMap(y, predict_labels);
        ACC = length(find(y == res)) / length(y);
        
        if ACC > best_ACC
            best_ACC = ACC;
            best_params = options;
        end
    end
end
```

## 常见问题 / FAQ

**Q: 运行时提示"Undefined function"错误？**  
A: 确保已添加依赖路径：
```matlab
addpath(genpath('../DMF_MVC/misc'));
addpath(genpath('../DMF_MVC/approx_seminmf'));
```

**Q: 如何调整层结构？**  
A: 修改`layers`参数，例如：
```matlab
layers = [200, 100];  % 两层：200 → 100 → numCluster
layers = [50];        % 一层：50 → numCluster
```

**Q: 算法不收敛怎么办？**  
A: 尝试：
- 减小正则化参数（lambda1, lambda2, beta）
- 增加maxIter
- 检查数据是否已归一化

更多问题见 `use.md` / More in `use.md`

## 性能基准 / Performance Benchmark

在WebKB数据集上的典型结果 / Typical results on WebKB:

| 指标 / Metric | 范围 / Range |
|--------------|-------------|
| ACC | 0.70 - 0.80 |
| NMI | 0.25 - 0.40 |
| Purity | 0.75 - 0.85 |
| 运行时间 / Time | 10 - 30s |

*结果取决于参数设置和硬件配置*

## 引用 / Citation

如果使用本代码，请引用：

```bibtex
@article{GDMFC2024,
  title={Graph-regularized Diversity-aware Deep Matrix Factorization for Multi-view Clustering},
  author={Your Name},
  year={2024}
}
```

## 许可 / License

本代码用于研究目的 / This code is for research purposes.

## 联系 / Contact

- 代码路径 / Code Path: `E:\research\paper\multiview\code\GDMFC\`
- 文档 / Documentation: `use.md`
- 优化推导 / Optimization: `../optimization/goal_function.md`

---

**版本 / Version:** 1.0  
**日期 / Date:** 2024  
**作者 / Author:** Generated for GDMFC Research Project
