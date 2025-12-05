# GDMFC Improved Demo 说明文档

## 概述

`demo_GDMFC_improve_ORL.m` 是一个改进版的GDMFC演示脚本，在**保持GDMFC核心算法不变**的前提下，借鉴了HDDMF的以下优点：

1. **数据预处理方法**：使用HDDMF的NormalizeFea列归一化方法
2. **图构建方法**：使用HDDMF的PKN（概率k近邻）方法替代GDMFC的热核方法
3. **数据集**：使用ORL40_3_400.mat数据集

## 文件说明

### 主要文件

1. **demo_GDMFC_improve_ORL.m** - 主演示脚本
   - 加载ORL40_3_400.mat数据集
   - 应用HDDMF风格的数据预处理
   - 调用改进的GDMFC算法
   - 运行10次取平均统计结果

2. **GDMFC_improved.m** - 改进的GDMFC算法实现
   - 核心算法：完全保持GDMFC不变
   - 图构建：可选择PKN（HDDMF）或热核（GDMFC）方法
   - 其他优化：保持GDMFC的所有优化技巧

## 使用方法

### 前置条件

1. 确保以下目录存在：
   - `../HDDMF/` - HDDMF代码目录
   - `../HDDMF/data/ORL40_3_400.mat` - 数据集文件

2. 确保以下函数可用：
   - `NormalizeFea` - 来自HDDMF/misc
   - `constructW_PKN` - 来自HDDMF/solvers
   - `seminmf` - 来自HDDMF/approx_seminmf
   - `computeHSIC` - 来自GDMFC
   - `SpectralClustering` - 来自HDDMF/misc或GDMFC

### 运行步骤

1. 打开MATLAB
2. 切换到GDMFC目录
3. 运行：
   ```matlab
   demo_GDMFC_improve_ORL
   ```

### 参数配置

在`demo_GDMFC_improve_ORL.m`中可以调整以下参数：

```matlab
% 网络结构
layers = [100, 50];  % 或 [200, 100, 50] 等

% 算法参数
options.lambda1 = 1e-5;      % HSIC多样性系数
options.lambda2 = 1e-3;      % 协正交约束系数
options.beta = 0.1;          % 图正则化系数
options.gamma = 5.0;         % 视图权重参数
options.graph_k = 5;         % k近邻数
options.maxIter = 100;       % 最大迭代次数
options.tol = 1e-5;          % 收敛容差

% 图构建方法选择
options.use_PKN = true;      % 使用PKN（HDDMF风格）
options.use_heat_kernel = false;  % 使用热核（GDMFC风格）
```

## 主要改进点

### 1. 数据预处理（HDDMF风格）

```matlab
% HDDMF预处理：直接列归一化
for v = 1:numView
    X{v} = NormalizeFea(data{v}', 0);  % 列归一化
end
```

**特点**：
- 简单直接
- 与HDDMF保持一致
- 保证特征尺度统一

### 2. 图构建方法（HDDMF PKN）

```matlab
% 使用PKN方法构建图
L{v} = constructGraphLaplacian_PKN(X{v}, graph_k);
```

**PKN方法特点**：
- 概率k近邻，距离一致性更好
- 权重公式：`W(i,j) = (d_{k+1} - d_j) / (k*d_{k+1} - sum(d_1 to d_k))`
- 自动对称化

### 3. 核心算法（GDMFC，保持不变）

- 逐层预训练
- 联合微调
- 自适应视图权重学习
- HSIC多样性约束
- 协正交约束
- 学习率阻尼

## 输出结果

运行后会生成：

1. **控制台输出**：
   - 每次迭代的目标函数值
   - 10次运行的平均结果和标准差
   - ACC, NMI, Purity指标

2. **结果文件**：
   - `GDMFC_improved_results_ORL40_3_400.mat` - 包含所有结果数据

3. **可视化**：
   - 目标函数收敛曲线
   - 视图权重分布
   - 性能指标对比

## 预期性能

结合HDDMF和GDMFC的优点，预期性能：
- **ACC**: 89%+ (相比HDDMF的86%有提升)
- **NMI**: 90%+
- **Purity**: 90%+

## 注意事项

1. **路径依赖**：确保HDDMF目录路径正确
2. **数据格式**：ORL40_3_400.mat应包含`data`（cell数组）和`label`（向量）
3. **内存需求**：根据数据集大小调整，ORL40_3_400约400样本，内存需求较小
4. **运行时间**：10次运行可能需要较长时间，可以调整`num_runs`参数

## 故障排除

### 问题1：找不到NormalizeFea函数
**解决**：确保`../HDDMF/misc/`路径已添加

### 问题2：找不到constructW_PKN函数
**解决**：确保`../HDDMF/solvers/`路径已添加

### 问题3：找不到seminmf函数
**解决**：确保`../HDDMF/approx_seminmf/`路径已添加

### 问题4：数据集文件不存在
**解决**：检查`../HDDMF/data/ORL40_3_400.mat`是否存在

## 版本信息

- **创建日期**: 2024
- **基于**: GDMFC核心算法 + HDDMF改进方法
- **数据集**: ORL40_3_400.mat

## 联系方式

如有问题，请参考：
- GDMFC原始代码
- HDDMF原始代码
- 本文档

