# GDMFC 重组说明文档

## 项目重组完成

本项目已完成模块化重组，所有依赖文件已复制到项目内部，所有路径引用已改为相对路径，可直接上传到GitHub。

## 新的目录结构

```
GDMFC/
├── core/                    # 核心算法
│   ├── GDMFC.m             # 主算法
│   └── GDMFC_improved.m    # 改进版算法（使用HDDMF的PKN图构建）
│
├── utils/                   # 工具函数
│   ├── approx_seminmf/     # 半非负矩阵分解（从HDDMF复制）
│   ├── computeHSIC.m       # HSIC计算
│   ├── data_guiyi_choos.m  # 数据预处理
│   ├── EuDist2.m           # 欧氏距离计算
│   ├── NormalizeFea.m      # 特征归一化（从HDDMF复制）
│   ├── SpectralClustering.m # 谱聚类（从HDDMF复制）
│   ├── bestMap.m           # 标签映射（从HDDMF复制）
│   ├── MutualInfo.m        # 互信息计算（从HDDMF复制）
│   └── ...                 # 其他工具函数
│
├── solvers/                 # 求解器
│   ├── constructGraphLaplacian.m  # 图拉普拉斯构建（热核方法）
│   └── constructW_PKN.m           # PKN图构建（从HDDMF复制）
│
├── demos/                   # 演示脚本
│   ├── demo_GDMFC.m
│   ├── demo_GDMFC_orl.m
│   ├── demo_GDMFC_Washington.m
│   └── demo_GDMFC_improve_ORL.m   # 改进版（使用HDDMF方法）
│
├── scripts/                 # 参数搜索脚本
│   ├── best_param.m
│   ├── best_param_v2.m
│   ├── search_best_beta_orl.m
│   └── ...
│
├── data/                    # 数据文件
│   └── ORL40_3_400.mat     # 示例数据集（从HDDMF复制）
│
├── results/                 # 结果文件
│   └── *.mat, *.csv        # 实验结果
│
├── setup_paths.m           # 路径初始化脚本
└── README.md               # 原始README
```

## 使用方法

### 方法1：使用setup_paths.m（推荐）

```matlab
% 在运行任何demo之前
setup_paths;

% 然后运行demo
cd demos
demo_GDMFC_improve_ORL
```

### 方法2：手动添加路径

```matlab
% 获取根目录
root_dir = fileparts(mfilename('fullpath'));

% 添加所有子目录
addpath(genpath(fullfile(root_dir, 'core')));
addpath(genpath(fullfile(root_dir, 'utils')));
addpath(genpath(fullfile(root_dir, 'solvers')));
addpath(genpath(fullfile(root_dir, 'demos')));
```

## 主要改进

### 1. 模块化结构
- **core/**: 核心算法，保持GDMFC算法不变
- **utils/**: 所有工具函数，包括从HDDMF复制的依赖
- **solvers/**: 图构建方法（支持热核和PKN两种）
- **demos/**: 所有演示脚本
- **scripts/**: 参数搜索和实验脚本
- **data/**: 数据文件
- **results/**: 结果文件

### 2. 相对路径
- 所有路径引用已改为相对路径
- 使用`mfilename('fullpath')`获取脚本位置
- 使用`fullfile()`构建跨平台路径
- 可直接上传到GitHub，无需修改路径

### 3. 依赖管理
- 所有依赖已复制到项目内部
- 不再依赖外部项目（如DMF_MVC、HDDMF）
- 项目完全自包含

## 从HDDMF借鉴的改进

在`demo_GDMFC_improve_ORL.m`中，我们结合了HDDMF的优点：

1. **数据预处理**：使用HDDMF的NormalizeFea列归一化
2. **图构建**：使用HDDMF的PKN（概率k近邻）方法
3. **核心算法**：保持GDMFC不变

## 文件说明

### 核心算法
- `core/GDMFC.m`: 原始GDMFC算法
- `core/GDMFC_improved.m`: 改进版，支持PKN图构建

### 工具函数
- `utils/approx_seminmf/`: 半非负矩阵分解（从HDDMF复制）
- `utils/NormalizeFea.m`: 特征归一化（从HDDMF复制）
- `utils/SpectralClustering.m`: 谱聚类（从HDDMF复制）
- `utils/computeHSIC.m`: HSIC计算（GDMFC原有）

### 求解器
- `solvers/constructGraphLaplacian.m`: 热核图构建（GDMFC原有）
- `solvers/constructW_PKN.m`: PKN图构建（从HDDMF复制）

## 快速开始

```matlab
% 1. 初始化路径
setup_paths;

% 2. 运行改进版demo（使用HDDMF方法）
cd demos
demo_GDMFC_improve_ORL

% 或运行原始demo
demo_GDMFC_orl
```

## 注意事项

1. **路径初始化**：运行任何脚本前，确保已调用`setup_paths`或手动添加路径
2. **数据文件**：确保`data/`目录中有所需的数据集
3. **结果保存**：结果会自动保存到`results/`目录

## GitHub上传准备

✅ 所有依赖已复制到项目内部  
✅ 所有路径已改为相对路径  
✅ 项目结构模块化  
✅ 可直接上传到GitHub  

## 联系方式

如有问题，请参考：
- 原始README.md
- 代码注释
- HDDMF项目（用于参考）

