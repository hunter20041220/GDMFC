# GDMFC: Graph-regularized Diversity-aware Deep Matrix Factorization for Multi-view Clustering

> 图正则化多样性感知深度矩阵分解的多视图聚类算法

**Version:** 1.0  
**Language:** MATLAB R2024b+  
**Author:** Research Team  
**Date:** 2024

---

## 📋 目录

1. [项目概述](#项目概述)
2. [目录结构](#目录结构)
3. [核心文件说明](#核心文件说明)
4. [快速开始](#快速开始)
5. [详细使用指南](#详细使用指南)
6. [数据集说明](#数据集说明)
7. [参数调优](#参数调优)
8. [常见问题](#常见问题)

---

## 项目概述

GDMFC 是一个用于多视图聚类的深度学习算法实现。该算法结合了以下特性：

- **多视图融合**：同时处理多个不同的特征视图，学习权重进行最优融合
- **深度学习**：使用多层神经网络进行特征学习和表示
- **图正则化**：利用数据点之间的局部几何结构
- **多样性约束**：通过 HSIC（Hilbert-Schmidt Independence Criterion）保证视图间的多样性
- **参数优化**：使用梯度下降法优化目标函数

### 主要应用

- 人脸图像聚类（ORL 数据集）
- 文档聚类（Washington WebKB 数据集）
- 其他多视图数据的聚类任务

---

## 目录结构

```
GDMFC/
├── core/                          # 核心算法实现
│   ├── GDMFC.m                    # 主算法实现
│   └── GDMFC_improved.m           # 改进版本（实验中）
│
├── utils/                         # 工具函数库
│   ├── SpectralClustering.m       # 谱聚类（CPU 版）
│   ├── SpectralClustering_GPU.m   # 谱聚类（GPU 加速版）
│   ├── NormalizeFea.m             # 特征归一化 (L2 norm)
│   ├── data_guiyi_choos.m         # 多模式数据预处理/归一化 (5种模式)
│   ├── bestMap.m                  # 寻找最优标签映射
│   ├── MutualInfo.m               # 计算互信息 (NMI)
│   ├── compute_purity.m           # 计算聚类纯度
│   ├── computeHSIC.m              # HSIC 多样性计算
│   ├── EuDist2.m                  # 欧式距离计算
│   ├── litekmeans.m               # 轻量级 K-means
│   ├── hungarian.m                # 匈牙利算法（用于 bestMap）
│   ├── RandIndex.m                # Rand Index 计算
│   └── generate_checksum.m        # 文件校验和生成
│
├── demos/                         # 演示脚本
│   ├── demo_GDMFC_orl.m           # ORL 数据集演示（标准版）
│   ├── demo_GDMFC_Washington.m    # Washington WebKB 数据集演示
│   └── demo_GDMFC_improve_ORL.m   # 改进版本演示
│
├── scripts/                       # 辅助脚本
│   ├── systematic_search.m        # 系统化参数搜索
│   ├── best_param.m               # 参数优化搜索（v1）
│   ├── best_param_v2.m            # 参数优化搜索（v2）
│   ├── search_best_beta_orl.m     # Beta 参数最优搜索
│   ├── search_best_params_orl.m   # 多参数最优搜索
│   ├── run_orl_with_config.m      # 用配置文件运行 ORL
│   └── test_preprocess_orl.m      # 预处理测试
│
├── data/                          # 数据相关文件（通常为空或缓存）
│   └── orl_images_cache.mat       # ORL 图像缓存文件
│
├── results/                       # 实验结果存储目录
│   ├── GDMFC_results_ORL.mat      # ORL 标准运行结果
│   ├── GDMFC_results_Washington.mat  # Washington 数据集结果
│   └── *.csv, *.mat               # 各类参数搜索结果
│
├── docs/                          # 文档目录
│   └── 目标函数与优化.md          # 优化算法文档
│
├── GDMFC.m                        # 快捷指向 core/GDMFC.m
├── README.md                      # 本文件
├── QUICK_START.md                 # 快速开始指南
├── use.md                         # 使用说明
└── best_param_README.md           # 参数优化说明
```

---

## 核心文件说明

### 1. **算法核心** (`core/GDMFC.m`)

**功能**：GDMFC 算法的主实现

**函数签名**：
```matlab
[H, Z, alpha, obj_values] = GDMFC(X, numCluster, layers, options)
```

**输入参数**：
- `X`：多视图数据的 cell 数组，`X{v}` 为第 v 个视图的样本×特征矩阵
- `numCluster`：聚类数目（类别总数）
- `layers`：隐层维度向量，例如 `[400, 150, 40]` 表示两层隐藏层
- `options`：参数结构体，包括：
  - `lambda1`：HSIC 多样性系数（默认 1e-5）
  - `lambda2`：协正交约束系数（默认 1e-3）
  - `beta`：图正则化系数（默认 115）
  - `gamma`：视图权重参数（默认 5.0，必须 > 1）
  - `graph_k`：图构造的邻居数（默认 7）
  - `maxIter`：最大迭代次数（默认 100）
  - `tol`：收敛容差（默认 1e-5）

**输出**：
- `H`：最终低维表示矩阵（n × numCluster）
- `Z`：学到的变换矩阵
- `alpha`：视图权重向量
- `obj_values`：目标函数值序列

### 2. **演示脚本** (`demos/demo_GDMFC_orl.m`)

**功能**：完整的 ORL 数据集聚类演示

**执行步骤**：
1. 加载 ORL 人脸数据集（40 个人，10 张图/人，112×92 像素）
2. 构造多视图特征：
   - View 1：降采样像素（56×46）
   - View 2：分块统计特征（均值、标准差、最小值、最大值）
3. 数据预处理和归一化
4. 运行 GDMFC 算法
5. 谱聚类
6. 评估性能（ACC、NMI、Purity）
7. 可视化结果（收敛曲线、视图权重、混淆矩阵）
8. 保存结果到 `GDMFC_results_ORL.mat`

### 3. **Washington 数据集演示** (`demo_GDMFC_Washington.m`)

**功能**：Washington WebKB 数据集的演示（230 份文档，5 类）

**特点**：
- 加载 4 个视图：content（1703 词）、inbound links、outbound links、cites
- 使用 Matrix Market 稀疏格式读取
- 参数针对小规模数据优化

### 4. **数据预处理** (`utils/data_guiyi_choos.m`)

**功能**：提供 5 种数据预处理/归一化模式

**支持的模式**：
- `case 1`：MinMax 归一化（按行）
- `case 2`：MinMax 归一化（按列，转置后处理）
- `case 3`：L2 列向归一化（**推荐用于本算法**）
- `case 4`：按列求和归一化
- `case 5`：全局归一化

**使用示例**：
```matlab
X = data_guiyi_choos(X, 3);  % 使用 case 3（L2 列向）
```

### 5. **参数搜索脚本** (`scripts/systematic_search.m`)

**功能**：系统化参数搜索和优化

**搜索策略**：
- Step 1：基于 TOP30 beta 种子的归一化方法（1-5）搜索
- Step 2：Layers 结构搜索（从 50 递增到 400）
- Step 3：Gamma 参数搜索
- Step 4：Lambda1 参数搜索
- Step 5：Lambda2 参数搜索
- Step 6：K 值搜索

**输出**：
- `systematic_search_results.csv`：所有搜索结果表
- `systematic_search_results.mat`：搜索结果数据

---

## 快速开始

### 前置要求

- **MATLAB R2024b** 或更高版本
- **数据集**：ORL 或 Washington（已下载到 `../../dataset/`）
- **工具箱**（可选）：
  - Deep Learning Toolbox（用于 GPU 加速）
  - Image Processing Toolbox（用于图像处理）

### 最简单的运行方式

#### 方式 1：运行 ORL 演示（推荐首选）

在 MATLAB 命令行中：

```matlab
% 切换到 GDMFC 目录
cd 'E:\research\paper\multiview\code\GDMFC'

% 运行演示
demo_GDMFC_orl
```

**预期输出**：
```
========================================
GDMFC Demo on ORL Face Dataset
========================================

Step 1: Loading ORL face dataset...
  Loading 400 images from 40 subjects...
  Loaded 400 images (size: 112×92)
  Number of classes: 40

Step 2: Constructing multi-view features...
  View 1 (Downsampled pixels): 2576 dimensions
  View 2 (Block statistics): 400 dimensions
  ...

Results on ORL Face Dataset:
  ACC    = 0.8150 (81.50%)
  NMI    = 0.9051
  Purity = 0.8500
  ...
```

#### 方式 2：运行 Washington 演示

```matlab
demo_GDMFC_Washington
```

#### 方式 3：用自定义配置运行

```matlab
% 1. 准备数据
X = {X1, X2, X3, X4};  % 多视图特征
numCluster = 5;         % 聚类数
labels = ...;           % 真实标签

% 2. 设置参数
layers = [100, 50, 20];
options = struct();
options.lambda1 = 1e-4;
options.lambda2 = 1e-3;
options.beta = 0.1;
options.gamma = 2.0;
options.graph_k = 5;
options.maxIter = 100;

% 3. 运行算法
[H, Z, alpha, obj] = GDMFC(X, numCluster, layers, options);

% 4. 聚类和评估
S = H * H';
S = (S + S') / 2;
S = max(S, 0);
pred = SpectralClustering(S, numCluster);
res = bestMap(labels, pred);
ACC = mean(labels == res);
NMI = MutualInfo(labels, pred);
Purity = compute_purity(labels, pred);

fprintf('ACC=%.2f%%, NMI=%.4f, Purity=%.2f%%\n', ACC*100, NMI, Purity*100);
```

---

## 详细使用指南

### 1. 数据准备

#### ORL 数据集格式

```
E:\research\paper\multiview\dataset\orl\
├── s1/
│   ├── 1.pgm
│   ├── 2.pgm
│   └── ...
├── s2/
└── ...
└── s40/
```

- 40 个文件夹（s1 ~ s40），每个对应一个人
- 每个文件夹中 10 张 PGM 格式图像（112×92 像素，8 位灰度）
- 总共 400 张图像

#### Washington 数据集格式

```
E:\research\paper\multiview\dataset\Washington\
├── washington_content.mtx       # View 1: 文档-词
├── washington_inbound.mtx       # View 2: 入链
├── washington_outbound.mtx      # View 3: 出链
├── washington_cites.mtx         # View 4: 引用
├── washington_act.txt           # 标签（每行一个数字 1-5）
├── labels.txt                   # 类别名称
└── readme.txt                   # 数据集说明
```

- 230 份文档
- 5 个类别
- 4 个视图，矩阵市场格式（稀疏）

### 2. 多视图特征提取

#### 自定义提取特征

```matlab
% 示例：从 2 个视图提取特征

% View 1: 原始特征（例如像素）
X1 = load_feature_view1();  % 400 x 2576

% View 2: 不同特征类型（例如纹理）
X2 = load_feature_view2();  % 400 x 400

% 组织为 cell 数组
X = {X1, X2};

% 可选：先做每个视图的预处理
X = data_guiyi_choos(X, 3);  % case 3: L2 列向

% 再做全局 L2 归一化
for v = 1:length(X)
    X{v} = NormalizeFea(X{v}, 0);
end
```

#### 归一化模式选择

```matlab
% 测试不同的预处理模式
for case_id = 1:5
    X_test = data_guiyi_choos(X, case_id);
    for v = 1:length(X_test)
        X_test{v} = NormalizeFea(X_test{v}, 0);
    end
    
    % 运行 GDMFC
    [H, ~, ~, ~] = GDMFC(X_test, numCluster, layers, options);
    
    % 评估
    pred = SpectralClustering(H*H', numCluster);
    ACC = mean(labels == bestMap(labels, pred));
    fprintf('Case %d: ACC=%.2f%%\n', case_id, ACC*100);
end
```

### 3. 参数设置和调优

#### 推荐的参数范围

对于 **ORL 数据集**（40 类）：
```matlab
options.lambda1 = 1e-5;      % HSIC 系数
options.lambda2 = 1e-3;      % 协正交系数
options.beta = 115;          % 图正则化系数
options.gamma = 5.0;         % 视图权重参数
options.graph_k = 7;         % 图邻居数
layers = [300, 200, 100, 50];  % 4 层网络
```

对于 **Washington 数据集**（5 类）：
```matlab
options.lambda1 = 1e-4;
options.lambda2 = 1e-3;
options.beta = 0.1;
options.gamma = 2.0;
options.graph_k = 5;
layers = [100, 50, 20];  % 3 层网络
```

#### 参数的含义和调优

| 参数 | 范围 | 含义 | 调优指南 |
|------|------|------|---------|
| `lambda1` | 1e-6 ~ 1e-3 | HSIC 多样性强度 | 保持视图多样性，过小失去多样性约束，过大过度约束 |
| `lambda2` | 1e-4 ~ 1e-1 | 协正交约束强度 | 控制特征学习的稳定性，过大可能欠拟合 |
| `beta` | 0.01 ~ 1000 | 图正则化强度 | 高值强调局部几何，低值让数据驱动 |
| `gamma` | 1.5 ~ 10.0 | 视图权重非线性度 | 必须 > 1，越高视图分化越明显 |
| `graph_k` | 3 ~ 15 | 邻近图的邻居数 | 影响图的连接性，小数据集用小值 |
| `layers` | 可变 | 网络深度和宽度 | 更深的网络学习更复杂的表示，但可能过拟合 |

### 4. 自动参数搜索

#### 使用系统化搜索脚本

```matlab
% 运行系统化搜索（需要 TOP30 beta 列表和 orl_preprocessed.mat）
systematic_search

% 输出文件：
% - systematic_search_results.csv：所有试验结果
% - systematic_search_results.mat：搜索数据
```

#### 加载和分析搜索结果

```matlab
% 加载搜索结果
data = readtable('systematic_search_results.csv');

% 排序找到最佳参数
[~, best_idx] = max(data.ACC);
best_params = data(best_idx, :);

fprintf('Best configuration:\n');
fprintf('  Norm: %d, Beta: %d, Layers: %s\n', ...
    best_params.norm, best_params.beta, best_params.Layers);
fprintf('  ACC: %.2f%%, NMI: %.4f, Purity: %.2f%%\n', ...
    best_params.ACC, best_params.NMI, best_params.Purity);
```

---

## 数据集说明

### ORL Face Dataset

**来源**：Olivetti Research Laboratory (ORL)  
**样本数**：400（40 人 × 10 张）  
**类别数**：40  
**图像大小**：112 × 92 像素（8 位灰度）  
**特点**：
- 真实人脸图像
- 同一个人的图像具有光照、表情、姿态变化
- 多视图特征：像素 + 纹理统计

**数据准备**（已包含在项目中）：
- 路径：`E:\research\paper\multiview\dataset\orl\`
- 自动缓存：首次运行会生成 `orl_images_cache.mat`

### Washington WebKB Dataset

**来源**：University of Maryland  
**样本数**：230  
**类别数**：5（student, project, course, staff, faculty）  
**视图**：4 个
- content：1703 个词的文档-词矩阵
- inbound：入链矩阵
- outbound：出链矩阵
- cites：引用矩阵

**特点**：
- 稀疏矩阵（Matrix Market 格式）
- 链接结构信息
- 文本内容信息

---

## 参数调优

### 从零开始的调优流程

#### 步骤 1：确定基准参数

根据数据集大小和类别数选择初始参数：

```matlab
% 对于 N 样本、K 类、V 视图的数据集
num_samples = size(X{1}, 1);
num_clusters = K;
num_views = length(X);

% 初始化参数
options = struct();
options.lambda1 = 1e-5;          % 从小开始
options.lambda2 = 1e-3;          % 中等强度
options.beta = 0.1;              % 根据数据集大小调整
options.gamma = 2.0;             % 初始 2.0
options.graph_k = min(7, num_samples/10);  % 邻居数
options.maxIter = 100;
options.tol = 1e-5;

% 设置层结构
% 总维度大约为输入维度的一半，逐层递减
total_input_dim = sum(cellfun(@(x) size(x,2), X));
layers = round([total_input_dim/2, total_input_dim/4, num_clusters]);
```

#### 步骤 2：单参数扫描

逐个参数在小范围内扫描：

```matlab
% 扫描 beta 参数
beta_range = [0.01, 0.1, 1, 10, 100];
ACC_results = zeros(size(beta_range));

for i = 1:length(beta_range)
    options.beta = beta_range(i);
    [H, ~, ~, ~] = GDMFC(X, num_clusters, layers, options);
    pred = SpectralClustering(H*H', num_clusters);
    res = bestMap(labels, pred);
    ACC_results(i) = mean(labels == res);
    fprintf('beta=%g: ACC=%.2f%%\n', beta_range(i), ACC_results(i)*100);
end

% 找到最佳 beta
[~, best_idx] = max(ACC_results);
best_beta = beta_range(best_idx);
```

#### 步骤 3：网格搜索（可选）

在最优参数周围进行细致网格搜索：

```matlab
% 二维网格搜索 (gamma, beta)
gamma_range = [1.5, 2.0, 3.0, 5.0];
beta_range = [100, 115, 130, 150];

ACC_grid = zeros(length(gamma_range), length(beta_range));

for i = 1:length(gamma_range)
    for j = 1:length(beta_range)
        options.gamma = gamma_range(i);
        options.beta = beta_range(j);
        [H, ~, ~, ~] = GDMFC(X, num_clusters, layers, options);
        pred = SpectralClustering(H*H', num_clusters);
        res = bestMap(labels, pred);
        ACC_grid(i, j) = mean(labels == res);
    end
end

% 可视化和找到最佳点
imagesc(ACC_grid);
colorbar;
```

---

## 常见问题

### Q1：运行 demo 时报错 "函数或变量 'mmread' 无法识别"

**解决方案**：
Washington 演示使用自定义的 `read_matrix_market` 函数，不需要 `mmread`。如果仍有问题，检查 demo 文件末尾是否包含该函数定义。

### Q2：如何处理自己的数据集？

**步骤**：
1. 将多视图特征组织为 cell 数组：`X = {X1, X2, ...}`
2. 每个视图 `Xi` 应为 `(样本数) × (特征维度)` 的矩阵
3. 准备真实标签 `labels`（长度等于样本数）
4. 选择合适的层结构和参数
5. 调用 `[H, Z, alpha, obj] = GDMFC(X, numCluster, layers, options);`
6. 用谱聚类得到预测标签：`pred = SpectralClustering(H*H', numCluster);`

### Q3：如何改进聚类性能？

**可尝试的方向**：
1. **特征工程**：
   - 提取更有判别力的视图特征
   - 使用 PCA、LBP、HOG、CNN 等特征
   - 增加视图的多样性

2. **参数调优**：
   - 系统扫描 `beta`、`gamma`、`lambda1`、`lambda2`
   - 调整网络层数和宽度
   - 改变图的邻居数 `graph_k`

3. **预处理**：
   - 尝试不同的归一化模式（case 1-5）
   - 数据清洗和异常检测
   - 特征缩放和标准化

4. **算法改进**：
   - 增加最大迭代次数
   - 降低收敛容差
   - 尝试改进版本 `GDMFC_improved.m`

### Q4：如何保存和加载训练结果？

```matlab
% 保存
results = struct();
results.H = H;
results.alpha = alpha;
results.parameters = options;
results.ACC = ACC;
results.NMI = NMI;
save('my_result.mat', 'results');

% 加载
load('my_result.mat');
H_loaded = results.H;
```

### Q5：多少个样本和类别时算法效果好？

**建议**：
- **样本数**：至少 100+ （每类至少 5-10 个）
- **类别数**：2-50 个
- **视图数**：2-10 个视图
- **特征维度**：100-10000（太高时做 PCA 降维）

---

## 文献和参考

- **算法优化**：见 `docs/目标函数与优化.md`
- **参数调优参考**：见 `best_param_README.md`
- **使用示例**：见 `use.md` 和各 demo 文件
- **快速入门**：见 `QUICK_START.md`

---

## 许可和引用

如使用本代码进行研究或发表论文，请引用以下形式：

```bibtex
@software{gdmfc2024,
  title = {GDMFC: Graph-regularized Diversity-aware Deep Matrix Factorization for Multi-view Clustering},
  author = {Research Team},
  year = {2024},
  url = {https://github.com/hunter20041220/GDMFC}
}
```

---

## 支持和反馈

有问题或建议？请：
1. 检查本 README 的"常见问题"部分
2. 查看各 demo 文件中的注释和说明
3. 参考项目内的其他文档

---

**最后更新**：2024-12-09  
**维护者**：Research Team
