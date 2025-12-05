# GDMFC 快速参考 / Quick Reference

## 🚀 一行命令运行 / Run with One Command

```matlab
cd('E:\research\paper\multiview\code\GDMFC'); demo_GDMFC
```

---

## 📁 核心文件 / Core Files

| 文件名 | 作用 | 何时使用 |
|-------|-----|---------|
| **demo_GDMFC.m** | 🎯 主演示脚本 | **直接运行此文件** |
| GDMFC.m | 核心算法 | 被demo调用 |
| constructGraphLaplacian.m | 图构建 | 被GDMFC调用 |
| computeHSIC.m | 多样性计算 | 被GDMFC调用 |
| EuDist2.m | 距离计算 | 被图构建调用 |
| use.md | 详细说明 | 需要深入了解时 |
| README.md | 项目概述 | 快速了解项目 |

---

## 🎯 运行流程 / Workflow

```
1️⃣ 打开MATLAB → 2️⃣ cd到GDMFC文件夹 → 3️⃣ 运行demo_GDMFC → 4️⃣ 查看结果
```

---

## 📊 输出结果 / Output

### 控制台输出
```
Results on WebKB Dataset:
  ACC    = 0.XXXX (XX.XX%)  ← 聚类准确率
  NMI    = 0.XXXX           ← 归一化互信息
  Purity = 0.XXXX (XX.XX%)  ← 聚类纯度
```

### 生成文件
- `GDMFC_results_WebKB.mat` - 所有结果数据

### 可视化图表
- 左图：目标函数收敛曲线
- 右图：视图权重条形图

---

## ⚙️ 快速修改参数 / Quick Parameter Modification

在 `demo_GDMFC.m` 的 **第38-46行** 修改：

```matlab
% 常用调整 / Common Adjustments:

options.lambda1 = 0.1;    % ↑增大 = 更强图约束 / ↓减小 = 更弱约束
options.lambda2 = 0.01;   # ↑增大 = 更强多样性 / ↓减小 = 更少多样性
options.beta = 0.1;       % ↑增大 = 更强正交性 / ↓减小 = 更弱正交性
options.gamma = 1.5;      % ↑接近2 = 偏向好视图 / ↓接近1 = 平均权重

layers = [100, 50];       % 改为 [200, 100] = 更深网络
                          % 改为 [50] = 更浅网络
```

---

## 🔄 使用其他数据集 / Use Other Datasets

在 `demo_GDMFC.m` 的 **第21行** 修改：

```matlab
% 原始 / Original:
dataPath = '../../dataset/WebKB.mat';

% 改为 / Change to:
dataPath = '../../dataset/3Sources.mat';  % 或其他数据集
```

**可用数据集 / Available Datasets:**
- WebKB.mat (2类, 2视图)
- 3Sources.mat (6类, 3视图)
- BBCSport.mat (5类, 2视图)
- Handwritten.mat (10类, 6视图)
- 100Leaves.mat (100类, 3视图)

---

## 🐛 常见错误 / Common Errors

### Error: "Undefined function 'NormalizeFea'"
**原因 / Cause:** 缺少依赖函数  
**解决 / Fix:** demo已自动添加路径，确保运行demo_GDMFC而非直接运行GDMFC

### Error: "Index exceeds array elements"
**原因 / Cause:** 数据格式不对  
**解决 / Fix:** 确保数据有X（cell数组）和y（标签向量）

### Warning: "Matrix is singular"
**原因 / Cause:** 参数设置导致数值问题  
**解决 / Fix:** 减小lambda1, lambda2, beta到0.01-0.1范围

---

## 📈 预期性能 / Expected Performance

### WebKB数据集 (1051样本, 2类, 2视图)
- **ACC:** 70% - 80%
- **NMI:** 0.25 - 0.40
- **Purity:** 75% - 85%
- **运行时间:** 10-30秒

### 如果结果太低 / If Results Too Low:
1. 尝试增大 `lambda1` 到 0.3
2. 调整层结构 `layers = [150, 50]`
3. 增加图邻居数 `options.graph_k = 7`

---

## 🔍 结果解读 / Result Interpretation

| 指标 | 含义 | 好的结果 |
|-----|------|---------|
| **ACC** | 有多少样本被正确聚类 | > 0.70 |
| **NMI** | 聚类与真实标签的互信息 | > 0.30 |
| **Purity** | 每个聚类的主导类占比 | > 0.75 |
| **View Weights** | 哪个视图更重要 | 接近均匀 = 两视图都重要<br>不均匀 = 某视图质量更高 |

---

## 📞 获取帮助 / Get Help

1. **快速问题** → 查看本文件
2. **参数调优** → 阅读 `use.md` 的"参数调优"章节
3. **算法原理** → 阅读 `../optimization/goal_function.md`
4. **代码细节** → 查看各.m文件内的详细注释

---

## 🎓 学习路径 / Learning Path

### 初学者 / Beginner
1. 直接运行 `demo_GDMFC.m`
2. 修改参数观察结果变化
3. 尝试不同数据集

### 进阶 / Advanced
1. 阅读 `GDMFC.m` 代码理解算法细节
2. 阅读 `goal_function.md` 理解数学推导
3. 修改算法实现自定义功能

---

## ✅ 检查清单 / Checklist

运行前确认 / Before running:
- [ ] MATLAB版本 ≥ R2024b
- [ ] 在正确目录: `E:\research\paper\multiview\code\GDMFC\`
- [ ] WebKB.mat存在: `../../dataset/WebKB.mat`
- [ ] 依赖函数可访问: `../DMF_MVC/misc/` 和 `../DMF_MVC/approx_seminmf/`

---

## 💡 高级技巧 / Advanced Tips

### 加速收敛
```matlab
options.maxIter = 50;      % 减少迭代次数
options.tol = 1e-4;        % 放宽收敛条件
```

### 网格搜索最佳参数
```matlab
% 见 use.md 中的"参数网格搜索示例"
```

### 保存中间结果
在GDMFC.m的迭代循环中添加：
```matlab
if mod(iter, 10) == 0
    save(sprintf('checkpoint_iter%d.mat', iter), 'H', 'Z', 'alpha');
end
```

---

**快速入门完成！现在运行 demo_GDMFC 开始实验吧！** 🎉

**Quick start complete! Now run demo_GDMFC to start experimenting!** 🎉
