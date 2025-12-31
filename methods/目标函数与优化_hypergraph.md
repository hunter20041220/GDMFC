# 目标函数与优化 - 超图正则化版本

## 1. 目标函数 (Objective Function with Hypergraph Regularization)

在原有GDMFC的目标函数基础上,我们引入超图正则化(Hypergraph Regularization)来捕获数据的高阶关系(High-order Relationships)。超图能够同时连接多个顶点,相比传统的成对图(Pairwise Graph),能更好地保持数据的局部流形结构(Local Manifold Structure)。

### 1.1 完整目标函数

$$
\begin{aligned}
\min_{Z_i^{(v)},\, H_m^{(v)},\, \alpha^{(v)}} 
\quad 
\mathcal{L} 
&= \sum_{v=1}^{V} (\alpha^{(v)})^\gamma 
\left(
    \left\| X^{(v)} - Z_1^{(v)} Z_2^{(v)} \dots Z_m^{(v)} H_m^{(v)} \right\|_F^2
    + \beta\, \mathrm{tr}\!\left(H_m^{(v)} L_h^{(v)} (H_m^{(v)})^T \right)
\right) \\
&\quad - \lambda_1 \sum_{v \ne w}^V \mathrm{HSIC}(H_m^{(v)},\, H_m^{(w)})
+ \lambda_2 \sum_{v=1}^V \left\| H_m^{(v)} (H_m^{(v)})^T - I \right\|_F^2
\end{aligned}
$$

$$
\text{s.t.} \quad
H_m^{(v)} \ge 0,\quad 
\sum_{v=1}^V \alpha^{(v)} = 1,\quad
\alpha^{(v)} \ge 0.
$$

**与原目标函数的主要区别**: 
- 将图拉普拉斯矩阵 $L^{(v)}$ 替换为**超图拉普拉斯矩阵** $L_h^{(v)}$
- 超图能够捕获样本之间的高阶关系,而非仅仅成对关系

### 1.2 超图正则化 (Hypergraph Regularization)

#### 1.2.1 超图的定义

超图 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{W})$ 由以下组成:
- $\mathcal{V} = \{v_1, v_2, \ldots, v_n\}$: 顶点集合,对应数据样本
- $\mathcal{E} = \{e_1, e_2, \ldots, e_m\}$: 超边集合,每条超边可连接多个顶点
- $\mathbf{W}$: 超边权重,对角矩阵,元素 $w(e_j)$ 表示超边 $e_j$ 的权重

#### 1.2.2 关联矩阵 (Incidence Matrix)

定义关联矩阵 $\mathbf{R} \in \mathbb{R}^{n \times m}$:

$$
r(v_i, e_j) = 
\begin{cases}
1, & \text{if } v_i \in e_j \\
0, & \text{otherwise}
\end{cases}
$$

#### 1.2.3 度矩阵 (Degree Matrices)

**顶点度** (Vertex Degree):
$$
d(v_i) = \sum_{e_j \in \mathcal{E}} w(e_j) r(v_i, e_j)
$$

**超边度** (Hyperedge Degree):
$$
\delta(e_j) = \sum_{v_i \in \mathcal{V}} r(v_i, e_j)
$$

定义对角矩阵:
- $\mathbf{D}_v$: 顶点度对角矩阵,对角元素为 $d(v_1), d(v_2), \ldots, d(v_n)$
- $\mathbf{D}_e$: 超边度对角矩阵,对角元素为 $\delta(e_1), \delta(e_2), \ldots, \delta(e_m)$

#### 1.2.4 超图拉普拉斯矩阵

**非归一化超图拉普拉斯**:
$$
L_h = \mathbf{D}_v - \mathbf{R} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{R}^T
$$

**归一化超图拉普拉斯** (本文采用):
$$
\mathcal{L}_h = \mathbf{D}_v^{-1/2} L_h \mathbf{D}_v^{-1/2} = I - \mathbf{D}_v^{-1/2} \mathbf{R} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{R}^T \mathbf{D}_v^{-1/2}
$$

为简化记号,定义:
$$
\boxed{L_h^{(v)} = \mathbf{D}_v - \mathbf{R} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{R}^T}
$$

#### 1.2.5 超图正则化项的意义

超图正则化项:
$$
\mathrm{tr}\!\left(H_m^{(v)} L_h^{(v)} (H_m^{(v)})^T \right) = \frac{1}{2} \sum_{e \in \mathcal{E}} \frac{w(e)}{\delta(e)} \sum_{v_i, v_j \in e} \left\| h_i^{(v)} - h_j^{(v)} \right\|^2
$$

其中 $h_i^{(v)}$ 是 $H_m^{(v)}$ 的第 $i$ 列。

**物理意义**: 
- 使得在同一超边内的样本在表示空间中更加接近
- 保持了原始数据的高阶局部几何结构
- 相比传统图正则化,超图可以同时约束多个样本(而非仅成对约束)

## 2. 超图构造策略 (Hypergraph Construction)

### 2.1 k-近邻超图构造

对于第 $v$ 个视图的数据 $X^{(v)}$,我们采用**k-近邻超图**构造方法:

**步骤**:
1. 对每个样本 $x_i^{(v)}$,找到其 $k$ 个最近邻 $\mathcal{N}_k(i)$
2. 为每个样本 $i$ 创建一条超边 $e_i$,该超边连接样本 $i$ 及其 $k$ 个近邻:
   $$
   e_i = \{i\} \cup \mathcal{N}_k(i)
   $$
3. 构建关联矩阵 $\mathbf{R}$:
   $$
   r(j, e_i) = 
   \begin{cases}
   1, & \text{if } j \in e_i \\
   0, & \text{otherwise}
   \end{cases}
   $$
4. 计算超边权重(采用高斯核):
   $$
   w(e_i) = \frac{1}{|e_i|} \sum_{j \in e_i} \exp\left(-\frac{\|x_i^{(v)} - x_j^{(v)}\|^2}{2\sigma^2}\right)
   $$
   其中 $\sigma$ 为带宽参数,可设为数据的平均距离

### 2.2 与传统图的对比

| 特性 | 传统k-NN图 | k-NN超图 |
|------|-----------|---------|
| 边的连接 | 成对连接 (样本i与样本j) | 多路连接 (样本i与其k个近邻) |
| 关系阶数 | 二阶 (Pairwise) | 高阶 (High-order) |
| 正则化约束 | $\sum_{i,j} w_{ij} \|h_i - h_j\|^2$ | $\sum_{e} \frac{w(e)}{\delta(e)} \sum_{i,j \in e} \|h_i - h_j\|^2$ |
| 结构信息 | 局部成对相似性 | 局部邻域结构 |

**优势**: 超图能够更好地保持数据的内在流形结构,特别是在高维数据中,成对关系往往不足以描述复杂的几何结构。

## 3. 优化算法 (Optimization Algorithm)

采用**交替最小化策略**(Alternating Minimization),逐块优化各变量。

### 3.1 预训练阶段 (Pre-training Phase)

借鉴深度矩阵分解的逐层预训练思想,为后续的全局优化提供良好初始化。

**第一层** ($i=1$): 对每个视图 $v$,求解
$$
\min_{Z_1^{(v)}, H_1^{(v)}} \left\| X^{(v)} - Z_1^{(v)} H_1^{(v)} \right\|_F^2, \quad \text{s.t. } H_1^{(v)} \geq 0
$$

**第 $i$ 层** ($i = 2, \ldots, m$): 固定前 $i-1$ 层,求解
$$
\min_{Z_i^{(v)}, H_i^{(v)}} \left\| H_{i-1}^{(v)} - Z_i^{(v)} H_i^{(v)} \right\|_F^2, \quad \text{s.t. } H_i^{(v)} \geq 0
$$

**求解方法**: 使用交替最小二乘法(Alternating Least Squares)或乘法更新规则。

### 3.2 微调阶段 (Fine-tuning Phase)

#### 3.2.1 更新 $Z_i^{(v)}$ (中间层基矩阵)

固定 $H_m^{(v)}$ 和其他 $Z_j^{(v)}$ ($j \neq i$),优化 $Z_i^{(v)}$。

定义辅助矩阵:
$$
\Phi_i^{(v)} = Z_1^{(v)} Z_2^{(v)} \cdots Z_{i-1}^{(v)}, \quad 
\Psi_i^{(v)} = Z_{i+1}^{(v)} \cdots Z_m^{(v)} H_m^{(v)}
$$

子问题简化为:
$$
\min_{Z_i^{(v)}} \left\| X^{(v)} - \Phi_i^{(v)} Z_i^{(v)} \Psi_i^{(v)} \right\|_F^2
$$

**闭式解**:
$$
\boxed{Z_i^{(v)} = (\Phi_i^{(v)})^\dagger X^{(v)} (\Psi_i^{(v)})^\dagger}
$$

其中 $\dagger$ 表示 Moore-Penrose 伪逆。

**实现技巧**:
- 当矩阵条件数较好时,可用: $Z_i^{(v)} = [(\Phi_i^{(v)})^T \Phi_i^{(v)}]^{-1} (\Phi_i^{(v)})^T X^{(v)} [(\Psi_i^{(v)})^T]^\dagger$
- 当矩阵奇异或病态时,使用SVD计算伪逆

#### 3.2.2 更新 $H_m^{(v)}$ (顶层表示矩阵)

这是最关键也是最复杂的步骤。固定所有 $Z_i^{(v)}$ 和 $\alpha^{(v)}$,优化 $H_m^{(v)}$。

##### (A) 子问题形式

$$
\begin{aligned}
\min_{H_m^{(v)} \geq 0} \quad
\mathcal{L}_{H_m} 
&= \sum_{v=1}^{V} (\alpha^{(v)})^\gamma \left( \left\| X^{(v)} - \Phi_m^{(v)} H_m^{(v)} \right\|_F^2 + \beta\, \text{tr}\!\left(H_m^{(v)} L_h^{(v)} (H_m^{(v)})^T \right) \right) \\
&\quad - \lambda_1 \sum_{v \ne w}^V \text{HSIC}(H_m^{(v)},\, H_m^{(w)}) 
+ \lambda_2 \sum_{v=1}^V \left\| H_m^{(v)} (H_m^{(v)})^T - I \right\|_F^2
\end{aligned}
$$

其中 $\Phi_m^{(v)} = Z_1^{(v)} Z_2^{(v)} \cdots Z_m^{(v)}$。

##### (B) 各项梯度推导

**1. 重构误差项**:
$$
\frac{\partial}{\partial H_m^{(v)}} \left\| X^{(v)} - \Phi_m^{(v)} H_m^{(v)} \right\|_F^2 
= 2 (\Phi_m^{(v)})^T \Phi_m^{(v)} H_m^{(v)} - 2 (\Phi_m^{(v)})^T X^{(v)}
$$

**2. 超图正则化项** (关键推导):

$$
\begin{aligned}
&\frac{\partial}{\partial H_m^{(v)}} \text{tr}\!\left(H_m^{(v)} L_h^{(v)} (H_m^{(v)})^T \right) \\
&= \frac{\partial}{\partial H_m^{(v)}} \text{tr}\!\left(H_m^{(v)} [\mathbf{D}_v - \mathbf{R} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{R}^T] (H_m^{(v)})^T \right) \\
&= 2 H_m^{(v)} L_h^{(v)} \\
&= 2 H_m^{(v)} (\mathbf{D}_v - \mathbf{R} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{R}^T)
\end{aligned}
$$

为便于乘法更新,定义:
- $D_v^{(v)}$: 第 $v$ 个视图的超图顶点度矩阵
- $S^{(v)} = \mathbf{R} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{R}^T$: 超图亲和矩阵

则:
$$
\frac{\partial}{\partial H_m^{(v)}} \text{tr}\!\left(H_m^{(v)} L_h^{(v)} (H_m^{(v)})^T \right) 
= 2 H_m^{(v)} D_v^{(v)} - 2 H_m^{(v)} S^{(v)}
$$

**3. HSIC多样性项**:

使用内积核 $K^{(v)} = (H_m^{(v)})^T H_m^{(v)}$,中心化矩阵 $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$:

$$
\text{HSIC}(H_m^{(v)}, H_m^{(w)}) = \text{tr}(H K^{(w)} H K^{(v)})
$$

$$
\frac{\partial}{\partial H_m^{(v)}} \sum_{w \neq v} \text{HSIC}(H_m^{(v)}, H_m^{(w)}) 
= 2 H_m^{(v)} \sum_{w \neq v} H K^{(w)} H 
= 2 H_m^{(v)} K_{-v}
$$

其中:
$$
\boxed{K_{-v} = \sum_{w \neq v} H (H_m^{(w)})^T H_m^{(w)} H}
$$

**4. 正交约束项**:

$$
\begin{aligned}
&\frac{\partial}{\partial H_m^{(v)}} \left\| H_m^{(v)} (H_m^{(v)})^T - I \right\|_F^2 \\
&= \frac{\partial}{\partial H_m^{(v)}} \left[ \text{tr}(H_m^{(v)} (H_m^{(v)})^T H_m^{(v)} (H_m^{(v)})^T) - 2\text{tr}(H_m^{(v)} (H_m^{(v)})^T) + n \right] \\
&= 4 H_m^{(v)} (H_m^{(v)})^T H_m^{(v)} - 4 H_m^{(v)}
\end{aligned}
$$

##### (C) 完整梯度

$$
\begin{aligned}
\nabla_{H_m^{(v)}} \mathcal{L} 
&= 2(\alpha^{(v)})^\gamma \left[ (\Phi_m^{(v)})^T \Phi_m^{(v)} H_m^{(v)} - (\Phi_m^{(v)})^T X^{(v)} 
+ \beta H_m^{(v)} D_v^{(v)} - \beta H_m^{(v)} S^{(v)} \right] \\
&\quad - 2\lambda_1 H_m^{(v)} K_{-v} 
+ 4\lambda_2 (H_m^{(v)} (H_m^{(v)})^T H_m^{(v)} - H_m^{(v)})
\end{aligned}
$$

##### (D) 乘法更新规则

为满足非负约束 $H_m^{(v)} \geq 0$,将梯度分解为正部和负部。

**负部**(分子,驱动增长的项):
$$
\begin{aligned}
\nabla^- &= 2(\alpha^{(v)})^\gamma \left[ (\Phi_m^{(v)})^T X^{(v)} + \beta H_m^{(v)} S^{(v)} \right] \\
&\quad + 2\lambda_1 H_m^{(v)} K_{-v} + 4\lambda_2 H_m^{(v)}
\end{aligned}
$$

**正部**(分母,驱动衰减的项):
$$
\begin{aligned}
\nabla^+ &= 2(\alpha^{(v)})^\gamma \left[ (\Phi_m^{(v)})^T \Phi_m^{(v)} H_m^{(v)} + \beta H_m^{(v)} D_v^{(v)} \right] \\
&\quad + 4\lambda_2 H_m^{(v)} (H_m^{(v)})^T H_m^{(v)}
\end{aligned}
$$

**最终更新规则**:
$$
\boxed{
H_m^{(v)} \leftarrow H_m^{(v)} \odot \sqrt{\frac{\nabla^- + \epsilon}{\nabla^+ + \epsilon}}
}
$$

展开形式:
$$
\boxed{
H_m^{(v)} \leftarrow H_m^{(v)} \odot \sqrt{
\frac{
(\alpha^{(v)})^\gamma [(\Phi_m^{(v)})^T X^{(v)} + \beta H_m^{(v)} S^{(v)}] 
+ \lambda_1 H_m^{(v)} K_{-v} + 2\lambda_2 H_m^{(v)}
}{
(\alpha^{(v)})^\gamma [(\Phi_m^{(v)})^T \Phi_m^{(v)} H_m^{(v)} + \beta H_m^{(v)} D_v^{(v)}] 
+ 2\lambda_2 H_m^{(v)} (H_m^{(v)})^T H_m^{(v)} + \epsilon
}
}
}
$$

其中:
- $\odot$ 表示逐元素乘法(Hadamard product)
- $\epsilon$ 是小的正数(如 $10^{-10}$)以避免除零
- $S^{(v)} = \mathbf{R}^{(v)} \mathbf{W}^{(v)} (\mathbf{D}_e^{(v)})^{-1} (\mathbf{R}^{(v)})^T$ 是超图亲和矩阵

#### 3.2.3 更新 $\alpha^{(v)}$ (视图权重)

固定 $Z_i^{(v)}$ 和 $H_m^{(v)}$,优化 $\alpha^{(v)}$。

定义第 $v$ 个视图的总损失:
$$
\mathcal{R}^{(v)} = \left\| X^{(v)} - Z_1^{(v)} \cdots Z_m^{(v)} H_m^{(v)} \right\|_F^2 
+ \beta\, \text{tr}\!\left(H_m^{(v)} L_h^{(v)} (H_m^{(v)})^T \right)
$$

**注意**: 这里使用超图拉普拉斯 $L_h^{(v)}$。

子问题:
$$
\min_{\alpha^{(v)}} \sum_{v=1}^{V} (\alpha^{(v)})^\gamma \mathcal{R}^{(v)}, 
\quad \text{s.t. } \sum_{v=1}^{V} \alpha^{(v)} = 1, \alpha^{(v)} \geq 0
$$

构造拉格朗日函数并求解KKT条件,得到**更新规则**:

$$
\boxed{
\alpha^{(v)} = \frac{(\mathcal{R}^{(v)})^{\frac{1}{1-\gamma}}}{\sum_{w=1}^{V} (\mathcal{R}^{(w)})^{\frac{1}{1-\gamma}}}
}
$$

**参数 $\gamma$ 的作用**:
- $\gamma \to 1^+$: 权重集中于损失最小的视图(硬选择)
- $\gamma \to \infty$: 所有视图等权重 $\alpha^{(v)} \to 1/V$
- 推荐: $\gamma \in (1, 2]$ 以平衡多视图贡献

## 4. 算法流程 (Algorithm Procedure)

### 4.1 完整算法伪代码

```
算法: GDMFC with Hypergraph Regularization (GDMFC-H)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: 
  - X = {X^(1), X^(2), ..., X^(V)}: 多视图数据矩阵
  - k: 聚类数目
  - m: 层数
  - layers: 各层维度 [p_1, p_2, ..., p_m]
  - β: 超图正则化系数
  - λ_1: HSIC多样性系数  
  - λ_2: 正交约束系数
  - γ: 视图权重参数 (γ > 1)
  - k_hyper: 超图k-近邻参数
  - maxIter: 最大迭代次数
  - tol: 收敛容差

输出:
  - H_m^(v): 各视图的低维表示
  - Z_i^(v): 各层的基矩阵
  - α^(v): 视图权重

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1: 超图构造
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1: for v = 1 to V do
2:     构建第v个视图的超图 G^(v) = (V, E^(v), W^(v)):
3:         for i = 1 to n do
4:             找到样本i的k_hyper个最近邻 N_k(i)
5:             创建超边 e_i = {i} ∪ N_k(i)
6:             计算超边权重 w(e_i) = (1/|e_i|) Σ_{j∈e_i} exp(-||x_i - x_j||²/(2σ²))
7:         end for
8:     构建关联矩阵 R^(v)
9:     计算顶点度矩阵 D_v^(v) 和超边度矩阵 D_e^(v)
10:    计算超图拉普拉斯 L_h^(v) = D_v^(v) - R^(v) W^(v) (D_e^(v))^(-1) (R^(v))^T
11:    计算超图亲和矩阵 S^(v) = R^(v) W^(v) (D_e^(v))^(-1) (R^(v))^T
12: end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 2: 逐层预训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
13: 初始化 α^(v) = 1/V for all v
14: for i = 1 to m do
15:     for v = 1 to V do
16:         if i == 1 then
17:             使用Semi-NMF求解: X^(v) ≈ Z_1^(v) H_1^(v)
18:         else
19:             使用Semi-NMF求解: H_{i-1}^(v) ≈ Z_i^(v) H_i^(v)
20:         end if
21:     end for
22: end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3: 全局微调(交替最小化)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
23: 计算初始目标函数值 L_old
24: for iter = 1 to maxIter do
25:     
26:     ─────────────────────────────────────
27:     Step 1: 更新所有中间层基矩阵 Z_i^(v)
28:     ─────────────────────────────────────
29:     for i = 1 to m do
30:         for v = 1 to V do
31:             计算 Φ_i^(v) = Z_1^(v) ... Z_{i-1}^(v)
32:             计算 Ψ_i^(v) = Z_{i+1}^(v) ... Z_m^(v) H_m^(v)
33:             更新 Z_i^(v) = (Φ_i^(v))^† X^(v) (Ψ_i^(v))^†
34:         end for
35:     end for
36:     
37:     ─────────────────────────────────────
38:     Step 2: 更新顶层表示矩阵 H_m^(v)
39:     ─────────────────────────────────────
40:     for v = 1 to V do
41:         计算 Φ_m^(v) = Z_1^(v) ... Z_m^(v)
42:         计算中心化矩阵 H = I - (1/n)11^T
43:         计算 K_{-v} = Σ_{w≠v} H (H_m^(w))^T H_m^(w) H
44:         
45:         // 计算分子(负梯度部分)
46:         numerator = (α^(v))^γ [(Φ_m^(v))^T X^(v) + β H_m^(v) S^(v)]
47:                     + λ_1 H_m^(v) K_{-v} + 2λ_2 H_m^(v)
48:         
49:         // 计算分母(正梯度部分)
50:         denominator = (α^(v))^γ [(Φ_m^(v))^T Φ_m^(v) H_m^(v) + β H_m^(v) D_v^(v)]
51:                       + 2λ_2 H_m^(v) (H_m^(v))^T H_m^(v) + ε
52:         
53:         // 乘法更新
54:         H_m^(v) = H_m^(v) ⊙ sqrt(numerator / denominator)
55:     end for
56:     
57:     ─────────────────────────────────────
58:     Step 3: 更新视图权重 α^(v)
59:     ─────────────────────────────────────
60:     for v = 1 to V do
61:         计算 R^(v) = ||X^(v) - Φ_m^(v) H_m^(v)||_F² 
62:                     + β tr(H_m^(v) L_h^(v) (H_m^(v))^T)
63:     end for
64:     for v = 1 to V do
65:         α^(v) = (R^(v))^(1/(1-γ)) / Σ_w (R^(w))^(1/(1-γ))
66:     end for
67:     
68:     ─────────────────────────────────────
69:     Step 4: 收敛性检查
70:     ─────────────────────────────────────
71:     计算当前目标函数值 L_new
72:     if |L_old - L_new| / max(1, |L_new|) < tol then
73:         break  // 收敛,退出循环
74:     end if
75:     L_old = L_new
76:     
77: end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 4: 融合表示与聚类
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
78: 计算融合表示 H* = Σ_v α^(v) H_m^(v)
79: 对H*^T应用k-means聚类或谱聚类得到聚类标签
80: return H_m^(v), Z_i^(v), α^(v), 聚类标签
```

### 4.2 关键实现要点

#### 4.2.1 超图构造的高效实现

```matlab
% MATLAB伪代码
function [L_h, S, D_v, D_e, R, W] = constructHypergraph(X, k_hyper, sigma)
    % X: n x d 数据矩阵
    % k_hyper: 超图k近邻数
    % sigma: 高斯核带宽
    
    n = size(X, 1);
    
    % Step 1: 找k近邻(使用KNN搜索)
    [idx, dist] = knnsearch(X, X, 'K', k_hyper+1);  % +1因为包含自己
    idx = idx(:, 2:end);  % 去掉自己
    dist = dist(:, 2:end);
    
    % Step 2: 构建关联矩阵R (n x n, 每行对应一条超边)
    R = sparse(n, n);
    for i = 1:n
        R(i, i) = 1;  % 包含自己
        R(i, idx(i,:)) = 1;  % 包含近邻
    end
    
    % Step 3: 计算超边权重W
    W = sparse(n, n);
    for i = 1:n
        neighbors = [i, idx(i,:)];
        weight = 0;
        for j = neighbors
            weight = weight + exp(-dist(i, find(idx(i,:)==j, 1))^2 / (2*sigma^2));
        end
        W(i, i) = weight / length(neighbors);
    end
    
    % Step 4: 计算度矩阵
    D_e = diag(sum(R, 1));  % 超边度
    D_v = diag(R * diag(W) * ones(n, 1));  % 顶点度
    
    % Step 5: 计算超图拉普拉斯和亲和矩阵
    S = R * W * inv(D_e) * R';
    L_h = D_v - S;
end
```

#### 4.2.2 数值稳定性技巧

**1. 避免除零**:
- 在分母中添加 $\epsilon = 10^{-10}$
- 在计算伪逆时使用截断SVD: `pinv(A, tol)`

**2. 矩阵条件数检查**:
```matlab
if cond(A) > 1e10
    Z = pinv(A, 1e-6);  % 使用更大的截断阈值
else
    Z = A \ B;  % 直接求解
end
```

**3. 超图度矩阵的处理**:
```matlab
D_e_inv = diag(1 ./ (diag(D_e) + eps));  % 避免除零
```

**4. 乘法更新的数值稳定**:
```matlab
% 避免sqrt内出现负数(由于浮点误差)
ratio = max(numerator ./ denominator, eps);
H = H .* sqrt(ratio);
% 避免H过小
H = max(H, eps);
```

## 5. 收敛性分析 (Convergence Analysis)

### 5.1 理论保证

**定理 1** (基矩阵更新的最优性):
固定 $H_m^{(v)}$ 和其他基矩阵,更新规则 $Z_i^{(v)} = (\Phi_i^{(v)})^\dagger X^{(v)} (\Psi_i^{(v)})^\dagger$ 给出子问题的全局最优解。

**证明**: 该子问题是关于 $Z_i^{(v)}$ 的无约束二次优化问题,令梯度为零即得闭式解。

**定理 2** (表示矩阵更新的单调性):
在乘法更新规则下,目标函数关于 $H_m^{(v)}$ 的部分单调递减。

**证明**: 采用辅助函数方法(Auxiliary Function Method),构造辅助函数:
$$
Z(H, H') \geq \mathcal{L}(H), \quad Z(H, H) = \mathcal{L}(H)
$$

通过Jensen不等式和对数不等式 $z \geq 1 + \log z$ 证明乘法更新规则保证目标函数下降。

**定理 3** (视图权重更新的最优性):
权重更新规则 $\alpha^{(v)} = \frac{(\mathcal{R}^{(v)})^{1/(1-\gamma)}}{\sum_w (\mathcal{R}^{(w)})^{1/(1-\gamma)}}$ 满足KKT条件,是带约束优化问题的全局最优解。

**证明**: 构造拉格朗日函数,求导并利用约束条件求解。

### 5.2 收敛准则

**相对变化准则**:
$$
\frac{|\mathcal{L}_{t} - \mathcal{L}_{t-1}|}{\max(1, |\mathcal{L}_{t}|)} < \text{tol}
$$

推荐: $\text{tol} = 10^{-5}$ 或 $10^{-6}$

**实践经验**:
- 通常在100-200次迭代内收敛
- 预训练可显著加速收敛
- 良好的参数初始化很重要

## 6. 计算复杂度 (Computational Complexity)

### 6.1 各阶段复杂度分析

假设:
- $n$: 样本数
- $V$: 视图数
- $m$: 层数
- $p$: 各层隐藏维度(假设相同)
- $d$: 原始特征维度(假设各视图相同)
- $k$: 聚类数
- $T_{\text{pre}}$: 预训练迭代次数
- $T_{\text{fine}}$: 微调迭代次数

**Phase 1: 超图构造**
- k-NN搜索: $O(Vn^2 d)$ (暴力搜索) 或 $O(Vn \log n \cdot d)$ (KD树)
- 构建关联矩阵: $O(Vn k_{\text{hyper}})$
- 计算拉普拉斯: $O(Vn^2)$

**Phase 2: 预训练**
- 每层Semi-NMF: $O(Vm \cdot T_{\text{pre}} (ndp + np^2))$

**Phase 3: 微调(每次迭代)**
- 更新所有 $Z_i^{(v)}$: $O(Vm(ndp + np^2 + d^2p))$
- 更新所有 $H_m^{(v)}$: $O(V(ndp + np^2 + Vn^2 k))$
  - 其中 $Vn^2 k$ 来自HSIC项的 $K_{-v}$ 计算
  - 超图正则化项: $O(Vn^2 k)$ (稀疏矩阵乘法可优化)
- 更新 $\alpha^{(v)}$: $O(V(ndp + np^2))$

**总复杂度**:
$$
O\left( Vn^2d + VmT_{\text{pre}}(ndp + np^2) + T_{\text{fine}} \cdot Vm(ndp + np^2 + Vn^2k) \right)
$$

### 6.2 优化策略

**1. 稀疏化**:
- 超图亲和矩阵 $S^{(v)}$ 通常是稀疏的(只有k-近邻)
- 使用稀疏矩阵运算: `S_sparse = sparse(S)`

**2. 并行化**:
- 不同视图的超图构造可并行: `parfor v = 1:V`
- $Z_i^{(v)}$ 的更新可并行(视图独立)

**3. 近似方法**:
- Nyström近似: 用 $n' \ll n$ 个样本近似超图
- 随机SVD: 加速伪逆计算

**4. 预计算与缓存**:
- $\Phi_m^{(v)}, (\Phi_m^{(v)})^T \Phi_m^{(v)}$ 在Step 2中预计算
- $K_{-v}$ 只需更新一次(对所有视图)

## 7. 超参数设置指南 (Hyperparameter Tuning Guide)

### 7.1 参数敏感性分析

| 参数 | 推荐范围 | 作用 | 调优建议 |
|------|---------|------|---------|
| $\beta$ | $[0.01, 10]$ | 控制超图正则化强度 | 数据流形明显时增大 |
| $\lambda_1$ | $[0.001, 1]$ | 控制多视图多样性 | 视图差异大时增大 |
| $\lambda_2$ | $[0.001, 1]$ | 控制正交性约束 | 避免表示退化 |
| $\gamma$ | $(1, 2]$ | 控制视图权重平衡 | 1.5是常用值 |
| $k_{\text{hyper}}$ | $[5, 15]$ | 超图k近邻数 | 通常设为类别数 |
| $\sigma$ | 自适应 | 高斯核带宽 | 设为平均k近邻距离 |
| layers | 问题相关 | 网络深度与宽度 | 深度2-3,宽度逐渐减小 |

### 7.2 自适应参数设置

**1. 高斯核带宽 $\sigma$**:
```matlab
% 自适应设置为第k个近邻的平均距离
[~, dist] = knnsearch(X, X, 'K', k_hyper+1);
sigma = mean(dist(:, end));  % 最远近邻的平均距离
```

**2. 正则化参数 $\beta$**:
```matlab
% 基于重构误差与正则化项的比例
recon_error = norm(X - Z*H, 'fro')^2;
reg_term = trace(H * L_h * H');
beta = recon_error / (reg_term + eps);
```

### 7.3 网格搜索策略

```matlab
% 使用对数网格搜索
beta_list = [0.001, 0.01, 0.1, 1, 10];
lambda1_list = [0.001, 0.01, 0.1, 1];
lambda2_list = [0.001, 0.01, 0.1, 1];

best_acc = 0;
for beta = beta_list
    for lambda1 = lambda1_list
        for lambda2 = lambda2_list
            % 运行算法
            [H, Z, alpha] = GDMFC_Hypergraph(X, k, layers, ...
                beta, lambda1, lambda2, gamma);
            % 评估聚类性能
            acc = evaluate_clustering(H, true_labels);
            if acc > best_acc
                best_params = [beta, lambda1, lambda2];
                best_acc = acc;
            end
        end
    end
end
```

## 8. 与原始GDMFC的对比 (Comparison with Original GDMFC)

### 8.1 理论对比

| 方面 | 原始GDMFC (图正则化) | GDMFC-H (超图正则化) |
|------|---------------------|---------------------|
| **正则化项** | $\text{tr}(H L H^T)$, $L = D - W$ | $\text{tr}(H L_h H^T)$, $L_h = D_v - RWD_e^{-1}R^T$ |
| **关系建模** | 成对关系(边连接2个点) | 高阶关系(超边连接k+1个点) |
| **流形保持** | 局部成对相似性 | 局部邻域结构 |
| **权重矩阵** | $W_{ij}$ (n×n, 成对) | $W(e_i)$ + $R$ (关联) |
| **适用场景** | 低维流形,成对关系充分 | 高维流形,复杂邻域结构 |
| **计算复杂度** | $O(n^2 k)$ | $O(n k_{\text{hyper}}^2)$ (稀疏) |
| **参数** | $W, D$ | $R, W, D_v, D_e, S$ |

### 8.2 优势分析

**超图正则化的优势**:

1. **更强的几何保持能力**:
   - 传统图: 只约束成对样本 $(i,j)$ 接近
   - 超图: 约束邻域内所有样本 $\{i, j_1, j_2, \ldots, j_k\}$ 接近
   - 更好地保持局部流形的内在结构

2. **对噪声的鲁棒性**:
   - 传统图: 噪声边直接影响特定样本对
   - 超图: 超边内多个样本的平均效应,噪声被稀释

3. **高维数据的优势**:
   - 高维空间中,成对距离往往失去区分性(维度灾难)
   - 超图通过邻域结构保持更丰富的拓扑信息

4. **理论支持**:
   - 谱超图理论(Spectral Hypergraph Theory)保证了良好的聚类性质
   - 超图割(Hypergraph Cut)优化具有更强的理论保证

### 8.3 实验预期

在以下情况下,GDMFC-H预期优于GDMFC:
- 数据分布在复杂的非线性流形上
- 样本间存在明显的邻域结构(如社区、簇)
- 高维特征空间(如图像、文本)
- 噪声较多的数据

## 9. 实现建议 (Implementation Recommendations)

### 9.1 MATLAB实现框架

```matlab
function [H, Z, alpha, obj_history] = GDMFC_Hypergraph(X, k, layers, options)
%% GDMFC with Hypergraph Regularization
% 
% Input:
%   X - 多视图数据 cell array {X^(1), ..., X^(V)}
%   k - 聚类数
%   layers - 隐藏层维度 [p1, p2, ..., pm]
%   options - 参数结构体
%
% Output:
%   H - 最终表示 {H_m^(1), ..., H_m^(V)}
%   Z - 基矩阵 {Z_i^(v)}
%   alpha - 视图权重
%   obj_history - 目标函数历史

    %% 参数设置
    V = length(X);
    n = size(X{1}, 1);
    m = length(layers);
    
    % 默认参数
    if ~isfield(options, 'beta'), options.beta = 1; end
    if ~isfield(options, 'lambda1'), options.lambda1 = 0.1; end
    if ~isfield(options, 'lambda2'), options.lambda2 = 0.1; end
    if ~isfield(options, 'gamma'), options.gamma = 1.5; end
    if ~isfield(options, 'k_hyper'), options.k_hyper = k; end
    if ~isfield(options, 'maxIter'), options.maxIter = 100; end
    if ~isfield(options, 'tol'), options.tol = 1e-5; end
    
    %% Phase 1: 构建超图
    fprintf('Building hypergraphs...\n');
    [L_h, S, D_v] = deal(cell(V, 1));
    for v = 1:V
        [L_h{v}, S{v}, D_v{v}] = constructHypergraph(...
            X{v}, options.k_hyper, options.sigma);
    end
    
    %% Phase 2: 预训练
    fprintf('Pre-training...\n');
    [Z, H] = pretrain_layers(X, layers, k);
    
    %% Phase 3: 微调
    fprintf('Fine-tuning...\n');
    alpha = ones(V, 1) / V;
    obj_history = zeros(options.maxIter, 1);
    
    for iter = 1:options.maxIter
        % Update Z_i^(v)
        for i = 1:m
            for v = 1:V
                Z{v}{i} = update_basis(X{v}, Z{v}, H{v}{end}, i);
            end
        end
        
        % Update H_m^(v)
        for v = 1:V
            H{v}{end} = update_representation(...
                X{v}, Z{v}, H, v, L_h{v}, S{v}, D_v{v}, ...
                alpha(v), options);
        end
        
        % Update alpha^(v)
        alpha = update_weights(X, Z, H, L_h, options.gamma, options.beta);
        
        % Compute objective
        obj_history(iter) = compute_objective(...
            X, Z, H, alpha, L_h, options);
        
        % Check convergence
        if iter > 1 && abs(obj_history(iter) - obj_history(iter-1)) < ...
                options.tol * max(1, abs(obj_history(iter)))
            fprintf('Converged at iteration %d\n', iter);
            obj_history = obj_history(1:iter);
            break;
        end
        
        if mod(iter, 10) == 0
            fprintf('Iter %d: obj = %.6f\n', iter, obj_history(iter));
        end
    end
    
end
```

### 9.2 关键子函数

详细实现将在 `GDMFC_Hypergraph.m` 中提供:
- `constructHypergraph()`: 超图构造
- `pretrain_layers()`: 逐层预训练
- `update_basis()`: 基矩阵更新
- `update_representation()`: 表示矩阵更新(带超图正则化)
- `update_weights()`: 视图权重更新
- `compute_objective()`: 目标函数计算

## 10. 总结与展望 (Summary and Future Work)

### 10.1 本文贡献

1. **理论扩展**: 将GDMFC的图正则化扩展为超图正则化,更好地捕获高阶关系
2. **完整推导**: 提供了超图正则化下的完整优化推导和乘法更新规则
3. **算法设计**: 给出了高效的实现算法和数值稳定技巧
4. **参数指导**: 提供了详细的超参数设置和调优建议

### 10.2 理论保证

- 超图拉普拉斯的半正定性保证了正则化的有效性
- 交替最小化保证了目标函数的单调下降
- 乘法更新规则保证了非负约束的满足

### 10.3 未来改进方向

1. **自适应超图**: 在优化过程中动态更新超图结构
2. **注意力机制**: 为超边内不同样本分配不同权重
3. **深度超图网络**: 结合深度学习,构建端到端的超图神经网络
4. **大规模优化**: 开发基于采样或分块的大规模超图优化方法
5. **理论分析**: 提供收敛速率和聚类误差界的理论分析

### 10.4 与HDDMF论文的关系

本文的超图正则化思想直接借鉴自参考论文HDDMF,但做了以下调整:
- **目标函数**: 保持原GDMFC的目标(HSIC多样性+正交约束+视图权重)
- **超图部分**: 采用HDDMF的超图构造和正则化方法
- **优化策略**: 结合两者优点,保留视图加权的自适应机制

这种结合预期能够:
- 利用超图的高阶关系建模能力
- 保持视图自适应加权的优势  
- 通过HSIC多样性和正交约束增强表示能力

---

## 参考文献 (References)

1. **GDMFC原始论文**: Graph-regularized Diversity-aware Multi-view Factorization for Clustering
2. **HDDMF论文**: Diverse Deep Matrix Factorization with Hypergraph Regularization for Multi-View Data Representation
3. **超图理论**: Zhou, D., et al. "Learning with Hypergraphs: Clustering, Classification, and Embedding." NIPS 2006.
4. **谱超图理论**: Feng, Y., et al. "Hypergraph Neural Networks." AAAI 2019.
5. **HSIC多样性**: Benton, A., et al. "Deep Generalized Canonical Correlation Analysis." ICLR 2017.

---

**文档版本**: v1.0  
**创建日期**: 2024-12-31  
**作者**: GDMFC研究团队
