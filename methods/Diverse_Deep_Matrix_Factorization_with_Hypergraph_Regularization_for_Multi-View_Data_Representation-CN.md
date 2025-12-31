# Diverse_Deep_Matrix_Factorization_with_Hypergraph_Regularization for_Multi-View Data Representation

### Abstract

深度矩阵分解 (Deep Matrix Factorization, DMF) 已被证实为一种能够捕获多视图数据表示 (Multi-view Data Representation, MDR) 中复杂层次信息 (Hierarchical Information) 的强有力工具。然而，现有的多视图深度矩阵分解方法主要挖掘多视图数据的一致性 (Consistency)，却忽视了不同视图间的多样性 (Diversity) 以及数据的高阶关系 (High-order Relationships)，从而导致了宝贵的互补信息 (Complementary Information) 的丢失。在本文中，我们设计了一种用于多视图数据表示的超图正则化多样性深度矩阵分解 (Hypergraph Regularized Diverse Deep Matrix Factorization, HDDMF) 模型，旨在多层分解框架 (Multi-layer Factorization Framework) 内联合利用多视图的多样性与高阶流形 (High-order Manifold)。我们设计了一个新颖的多样性增强项 (Diversity Enhancement Term)，以利用数据不同视图间的结构互补性 (Structural Complementarity)。同时，利用超图正则化 (Hypergraph Regularization) 来保持每个视图中数据的高阶几何结构 (High-order Geometry Structure)。我们开发了一种高效的迭代优化算法 (Iterative Optimization Algorithm) 来求解所提出的模型，并提供了理论上的收敛性分析 (Convergence Analysis)。在五个真实世界数据集 (Real-world Data Sets) 上的实验结果表明，所提出的方法显著优于目前最先进的 (State-of-the-art) 多视图学习方法 (Multi-view Learning Approaches)。

### 1. Introduction

![image-20251230184751628](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230184751628.png)

现实世界的数据通常可以从多个视图（Multiple Views）进行描述，或者收集自不同的来源。例如，同一张图像可以通过其颜色、纹理和边缘特征来表示；同一条新闻可能由不同的机构报道。这些由不同数据视图描述的异构特征被称为多视图数据（Multi-view Data）[1]。多视图数据学习已成为机器学习（Machine Learning）领域的研究热点之一，因为它能为许多现实应用提供丰富且深刻的数据信息，例如生物信息学（Bioinformatics）[2]、人脸识别（Face Recognition）[3]和文档挖掘（Document Mining）[4]。

在过去十年中，学术界提出了大量的多视图数据表示（Multi-view Data Representation, MDR）方法。在处理多视图数据时，与仅将多种类型的特征在通过一个大矩阵进行简单拼接（Concatenate）的传统单视图方法不同，多视图方法旨在将丰富的信息和多路交互（Multi-way Interactions）系统地嵌入到学习过程中[5], [6]。经典的数据表示方法包括自表示（Self-representation）[7]、谱聚类（Spectral Clustering）[8]、非负矩阵分解（Non-negative Matrix Factorization, NMF）、稀疏编码（Sparse Coding）[9], [10]以及张量分解（Tensor Factorization）[11], [12]。例如，Gao等人[13]提出了一种具有公共指示矩阵（Indicator Matrix）的自表示方法，以保证不同视图间聚类结构的一致性（Consistency）。文献[14]引入了张量正则化自表示（Tensor Regularized Self-representation）[15], [16]，以确保低冗余（Low Redundancy）并探索多视图潜在的高阶相关性（High-order Correlations）。文献[8]的工作开发了一种协同正则化谱聚类方法，以获得一致的聚类结果。在文献[17]中，Xu等人通过在指示矩阵上添加张量核范数最小化（Tensor Nuclear Norm Minimization）[18]来控制不同视图间的一致性。然而，自表示方法和谱聚类方法都需要构建对称的亲和矩阵（Affinity Matrix），这使得它们难以在大规模数据集（Large-scale Data Sets）上应用。

近年来，基于NMF的方法在MDR领域引起了广泛关注。由于该方法能够利用低维的基于部分（Parts-based）的表示矩阵，它可以进一步提高聚类任务的准确性（Accuracy）和可扩展性（Scalability）[19], [20]。为了保持聚类解的可比性，Liu等人[21]提出了一种MultiNMF方法，该方法构建了一个一致性项（Consensus Term）来学习跨不同视图的公共矩阵。文献[22]提出了一种部分共享的NMF方法，以同时考虑多视图数据的特性（一致性和互补性）。Yang等人[23]设计了一种均匀分布的多视图NMF模型，通过联合学习潜在的一致性矩阵来减少不同视图间的分布散度（Distribution Divergences）。尽管上述基于NMF的方法在特定条件下通常能取得令人满意的聚类性能[24]，但它们大多采用单层形式（One-layer Formulation），这无法捕获原始数据中包含的复杂的层次信息（Hierarchical Information）和隐式的低层隐藏属性（Implicit Low-level Hidden Attributes）。

受深度学习（Deep Learning）方法进展的启发[25]，Trigeorgis等人[26]提出了一种新颖的深度矩阵分解（Deep Matrix Factorization, DMF）模型，用于学习隐藏表示，使其能够根据输入数据的未知属性进行聚类解释。与传统的基于NMF的方法相比，DMF具有更强的数据表示能力[27]，因而受到研究人员的青睐并迅速扩展到各种场景中：包括社区检测（Community Detection）[28]、遥感（Remote Sensing）[29]等。紧随其后，Zhao等人[30]通过直接固定多视图间的公共单侧因子，将单视图DMF扩展为多视图版本（MDMF）。文献[31]提出了一种无参数的MDMF，以简化模型结构并降低复杂度。文献[32]提出了一种部分共享的深度矩阵分解模型，利用部分标签信息来兼顾一致性信息和视图特有特征（View-specific Features）。此外，多层分解技术也被应用于提高其他传统浅层分解模型的表示能力。基于概念分解（Concept Factorization）的方法[33]被开发用于捕获全面的多视图信息。文献[34]提出了一种新颖的深度多视图概念学习方法，以半监督方式（Semi-supervised Way）对一致性和互补性信息进行建模。在文献[35]中，作者设计了一种新颖的鲁棒自加权深度k-means多视图模型，该模型直接给出划分结果。最近，Huang等人[36]提出了一种类深度自编码器（Deep Autoencoder-like）的NMF方法，旨在寻找一种同时考虑互补和一致信息的紧凑多视图表示。

动机（Motivation）： 值得注意的是，这些基于DMF的MDR方法仅强调多视图间的一致性（Consensus），而忽略了多样性属性（Diversity Attribute），导致每个非显著视图中影响性能的相互互补信息（Complementary Information）丢失。我们考虑引入多样性约束，以确保每个视图的表示尽可能包含独特的信息，从而发现跨不同视图的结构互补性（Structural Complementarity）。一些工作也指出了多样性在多视图学习中的重要性[37], [38]。另一方面，现有方法通常无法保持局部流形结构（Local Manifold Structure）或仅考虑成对连接（Pairwise Connectivity）（例如 MDMF [30]）。在现实世界的应用中，数据点之间的关系往往比简单的成对关系更为复杂。如果将这种复杂关系简单地压缩为成对关系，必然会导致学习任务中有价值信息的丢失。一些研究人员也展示了高阶几何正则化（即超图正则化，Hypergraph Regularization）在数据表示中的优势[39], [40]。

在本文中，为了解决上述问题，我们提出了一种基于超图正则化多样性深度矩阵分解的多视图数据表示方法（HDDMF）。如图1所示，每个视图的数据矩阵（上标 $v$ 表示第 $v$ 个视图）被分解为 $m$ 个基矩阵（Basis Matrices，下标 $i$ 表示第 $i$ 层）以及一个表示矩阵（Representation Matrix，记为 $H_m^v$）。多样性增强约束被施加在最终的低维表示 $H_m^v$ 上。如图1红框所示，如果两个样本在第1个视图的子空间（Subspace）中相似，HDDMF强制它们在第 $V$ 个视图的子空间中表现出互补性。这种方法确保了多视图间的多样性信息能够被捕获，从而实现更全面的学习。通过引入超图嵌入正则化（Hypergraph Embedding Regularization），HDDMF保留了嵌入在高维特征空间中的高阶几何结构，以显式地对视图特有特征进行建模。超图正则化和多样性约束可以起到良好的互补平衡作用；超图正则项可以防止因过度的多样性约束而导致内部几何流形（Geometric Manifold）的丢失，并有助于区分不同视图的学习表示，实现更全面的学习。

HDDMF的主要贡献总结如下：

1. 在多视图数据存在多样性信息的假设下，建立了一个基于多样性增强深度矩阵分解的多视图表示学习模型，以探索视图间（Inter-views）和视图内（Intra-views）存在的结构互补性。
2. 引入超图正则化以保持内在几何结构（Intrinsic Geometrical Structure），这能够捕获视图特有数据局部性的高阶关系（High-order Relation），并增强模型的表示能力。
3. 我们开发了一种高效的算法来优化HDDMF，并证明了该算法能够单调地降低HDDMF的目标函数（Objective Function）并收敛到一个驻点（Stationary Point）。

本文的其余部分安排如下。在第二节中，我们简要介绍了NMF和DMF的一些预备知识（Preliminaries）。在第三节中，我们形式化地描述了提出的HDDMF模型。第四节A部分提出了一种求解该问题的高效算法，讨论了收敛性证明并分析了时间复杂度（Time Complexity）。在第五节中，我们报告了在五个真实世界数据集上的广泛实验结果。最后，第六节总结了全文。表I为了方便读者，归纳了本文的一般符号（Notations）。

![image-20251230184946007](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230184946007.png)

### 2. Preliminaries

非负矩阵分解 (Non-negative Matrix Factorization, NMF) [41], [42] 旨在分析非负数据 (Non-negative Data)。在数学上，给定一个数据矩阵 $X$，NMF 旨在将其近似分解为两个非负矩阵，即基矩阵 (Basis Matrix) $Z$ 和低秩表示矩阵 (Low-rank Representation Matrix) $H$

$$X \approx ZH$$

Cui 等人 [31] 通过放宽 NMF 对输入数据的非负约束，将 NMF 扩展为半非负矩阵分解 (Semi-NMF)，从而使模型能够处理混合符号数据 (Mix-sign Data)。Semi-NMF 可以被视为 k-means 的软版本 (Soft Version)，其中 $Z$ 表示聚类质心 (Cluster Centroids)，$H$ 表示每个数据点的聚类指示器 (Cluster Indicators)。另一方面，现实世界的数据集总是由复杂的和多层次的特征 (Multi-level Features) 组成。我们学到的浅层表示 (Shallow Representation) 可能包含复杂的结构和层次信息 (Hierarchical Information)。例如，人脸图像还包含关于姿态、表情、衣着和其他属性的信息，这些有助于识别所描绘的人物。为了提取更具表现力的表示 (Expressive Representation)，Trigeorgis 等人 [26] 通过将数据矩阵 $X$ 分解为多个因子，将 Semi-NMF 扩展为深度矩阵分解 (Deep Matrix Factorization, DMF)，以学习高层表示 (High-level Representation)

$$X \approx Z_1 Z_2 \cdots Z_m H_m \tag{2}$$

其中 $m$ 是层数，基矩阵 $Z_1 \in \mathbb{R}^{q \times p_1}, \dots, Z_m \in \mathbb{R}^{p_{m-1} \times p_m}$，表示矩阵 $H_m \in \mathbb{R}_{+}^{p_m \times n}$。事实上，公式 (2) 中的近似对应于 $X$ 的连续分解 (Successive Factorizations)

$$\begin{aligned} X &\approx Z_1 H_1 \\ H_1 &\approx Z_2 H_2 \\ &\vdots \\ H_{m-1} &\approx Z_m H_m \end{aligned} \tag{3}$$

因此，基于 Frobenius 范数 (Frobenius Norm)，DMF 的损失函数 (Loss Function) 可以写为

$$\min_{Z_1, \dots, H_m} \| X - Z_1 Z_2 \cdots Z_m H_m \|_F^2 \quad \text{s.t. } H_m \geq 0 \tag{4}$$

其中 $\|\cdot\|_F$ 是 Frobenius 范数。DMF 可以弥补浅层 NMF 方法的不足，因为其多层分解 (Multi-layer Decomposition) 能够捕获数据的层次结构 (Hierarchical Structure)，从而提高低维数据表示 (Low-dimensional Data Representation) 和聚类 (Clustering) 的性能。

为了应对多视图数据 (Multi-view Data) 的挑战，可以直接设计一种通用的多视图版本的深度矩阵分解。让我们用 $X = \{X^1, X^2, \dots, X^V\}$ 表示具有 $V$ 个视图的输入数据矩阵，其目标函数可以表示为

$$\min_{Z_1^v, \dots, H_m^v} \sum_{v=1}^{V} \| X^v - Z_1^v Z_2^v \cdots Z_m^v H_m^v \|_F^2 \quad \text{s.t. } H_m^v \geq 0 \tag{5}$$

每个矩阵 $Z_i^v$ 可以解释为第 $i$ 层的基（也称为权重）矩阵 (Basis/Weight Matrix)，每个 $H_m^v$ 可以表示为第 $m$ 层的表示矩阵。此后，最终的多视图表示通常构建为 $H_m^v$ 的平均值 [38]

$$H^* = \frac{\sum_{v=1}^{V} H_m^v}{V} \tag{6}$$

由于上述方法仅考虑每个视图数据的特定属性 (Specific Attribute)，无法衡量多视图数据的多样性属性 (Diversity Attribute)，我们将该方法称为非多样性深度矩阵分解 (Non-diverse Deep Matrix Factorization, NdDMF)。

### 3. Hypergraph Regularized Diversity-Enhanced Deep Matrix Factorization

在本节中，我们将提出一种新的深度矩阵分解（Deep Matrix Factorization）方法，该方法能够保持高阶内在几何结构（High-order Intrinsic Geometrical Structure），并同时利用多视图多样性信息（Multi-view Diversity Information）以构建一个完整的最终表示矩阵（Final Representation Matrix）。我们首先详细阐述两个主要组件：1）用于发现数据间高阶关系（High-order Relationships）的超图函数（Hypergraph Function）；2）旨在增强多视图多样性表示能力（Multi-view Diversity Representation Ability）的学习机制。随后，我们将给出最终的目标函数（Objective Function）及其算法求解方案（Algorithmic Solution）。关于算法收敛性（Convergence）的证明以及时间复杂度（Time Complexity）的分析包含在最后的小节中。

##### 3.1. Hypergraph Regularization

我们构建一个超图 (Hypergraph) $G = (V, E, W)$ 以编码数据空间中的高阶关系 (High-order Relationships)。$V$ 表示顶点的有限集合，$E$ 是 $V$ 的超边 (Hyperedge) $e$ 的族，且满足 $\bigcup_{e \in E} = V$。$W$ 由 $w(e)$ 组成，其被定义为加权函数 (Weighting Function) 以度量超边的权重 [43]。大小为 $V \times E$ 的关联矩阵 (Incident Matrix) $R$ 用于定义顶点与超边之间的关系，若 $v_i \in e_i$（注：此处原文应指 $v_i \in e_j$），则其元素 $r(v_i, e_i)$ 为 1，否则为 0。因此，每个顶点 $d(v_i)$ 的度 (Degree) 和超边 $d(e_j)$ 的度可以通过下式计算

$$d(v_i) = \sum_{e_j \in E} w(e_j)r(v_i, e_j)$$

$$d(e_j) = \sum_{v_i \in V} r(v_i, e_j). \tag{7}$$

与文献 [44] 类似，非归一化超图矩阵 (Unnormalized Hypergraph Matrix) 定义如下：

$$L_h = D_V - R W D_E^{-1} R^T \tag{8}$$

其中 $D_V$ 和 $D_E$ 是对角矩阵 (Diagonal Matrices)，分别对应于 $d(v_i)$ 和 $d(e_j)$。因此，超图正则化项 (Hypergraph Regularization Term) 可以形式化为

$$\begin{aligned} &\frac{1}{2} \sum_{e \in E} \sum_{\{i,j\} \in e} \frac{w(e)}{d(e)} \|H_i - H_j\|^2 \\ &= \frac{1}{2} \sum_{e \in E} \sum_{\{v_i, v_j\} \in V} \frac{w(e)r(v_i,e)r(v_j,e)}{d(e)} \|H_i - H_j\|^2 \\ &= \sum_{e \in E} \sum_{v_i \in V} w(e)r(v_i,e) H_i^T H_i \\ &\quad - \sum_{e \in E} \sum_{\{v_i, v_j\} \in V} \frac{w(e)r(v_i,e)r(v_j,e)}{d(e)} H_i^T H_j \\ &= \text{tr}(H D_V H^T) - \text{tr}(H R W D_E^{-1} R^T H^T) \\ &= \text{tr}(H L_h H^T) \end{aligned} \tag{9}$$

其中 $H$ 表示表示矩阵 (Representation Matrix)，$H_i$ 表示第 $i$ 个数据表示向量。超图是图 (Graph) 的一种推广 (Generalization)，其中一条超边可以连接任意数量的顶点，而以前的普通图边仅能表示顶点对。因此，构建超图而非普通图能够保持样本间的高阶关系 (High-order Relationships)。

##### 3.2. Diversity Measurement

为了保证两个视图之间的多样性 (Diversity)，主要思想是控制两个视图中数据表示的正交性 (Orthogonality)。如图 2(a) 所示，我们将第 $v$ 个视图的指示矩阵 (Indicator Matrix) 记为 $Q^v$。为了量化两个视图（$v$ 和 $w$）之间的多样性，我们可以最小化以下函数 [45], [46]：

$$\|Q_i^v \odot Q_i^w\|_1 = \sum_{j=1}^n Q_{ji}^v Q_{ji}^w \tag{10}$$

其中算子 $\odot$ 表示逐元素乘积 (Element-wise Product)。设 $h_i^v$ 表示如图 2(b) 所示的数据样本 $i$ 的潜在特征 (Latent Features) $H^v$， $h_i^w$ 表示潜在特征 $H^w$，一种朴素方法 (Naive Method) 是直接将潜在特征 $H^v$ 作为指示矩阵 $Q^v$ [38]，并设计如下的多样性度量项 (Diversity Measure Term) $DI(\cdot)$：

$$DI(H^v, H^w) = \sum_{j=1}^p \sum_{i=1}^n (H^v)_{ji} (H^w)_{ji} \tag{11}$$

对于多视图表示学习 (Multi-view Representation Learning)，直接约束来自不同视图的同一样本表示向量的正交性，其解释性较弱 (Weak Interpretability)。因为不同的视图代表不同的异构特征 (Heterogeneous Features)，很难在表示列向量的位置上实现一一对应 (One-to-one Correspondence)。此外，上述 $DI$ 项无法度量潜在特征内部的关系。为了解决上述问题，我们首先将 $Q^v$ 定义为新表示矩阵 $H^v$ 的第 $j$ 行和第 $i$ 列的内积 (Inner Product)，即 $Q^v = H^{vT} H^v$。沿着这一思路，如图 2(c) 所示，我们设计了如下的多样性增强项 (Diversity Enhancement Term) $DE(\cdot)$：

$$DE(H^v, H^w) = \sum_{j=1}^n \sum_{i=1}^n (H^{vT} H^v)_{ji} (H^{wT} H^w)_{ji} \tag{12}$$

基于迹运算 (Trace Operation) 的性质，我们可以将 (12) 重新表述为一个简单的二次项 (Quadratic Term)：

$$DE(H^v, H^w) = \text{tr}(H^{vT} H^v H^{wT} H^w) \tag{13}$$

$DE$ 项确保了来自不同视图的表示矩阵内积的正交性。$Q$ 矩阵的每个列向量表示该样本与其他样本之间的相似性 (Similarity)，这与不同视图的位置相对应，并且具有很强的解释性 (Interpretability)。特别地，如果两个学习到的特征点 $h_i$ 和 $h_j$ 在第 $v$ 个视图中非常相似（即 $Q_{ij}^v \approx 1$），我们期望它们在第 $w$ 个视图中学习到互补特征 (Complementary Features)（即 $Q_{ij}^w \approx 0$）。总之，$DE$ 项本质上是在挖掘来自不同视图的样本对之间的多样性，并且能够探索存在于视图间 (Inter-views) 和视图内 (Intra-views) 的结构互补关系 (Structural Complementary Relationship)。

最后，结合 (5)，(9) 和 (13)，我们可以制定我们的超图正则化多样性增强深度矩阵分解 (Hypergraph Regularized Diversity-enhanced Deep Matrix Factorization) 的目标函数 (Objective Function) $O$ 如下：

$$O = \sum_{v=1}^V \left( \underbrace{\|X^v - Z_1^v \cdots Z_m^v H_m^v\|_F^2}_{\text{深度误差 (deep error)}} + \underbrace{\beta \text{tr}(H_m^v L_h^v H_m^{vT})}_{\text{超图正则化 (hypergraph regularization)}} + \mu \underbrace{\sum_{w=1; v \neq w}^V DE(H_m^v, H_m^w)}_{\text{多样性增强 (diversity enhancement)}} \right)$$

$$\text{s.t. } H_m^v \ge 0.$$

![image-20251230190001760](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230190001760.png)

### 4. Optimization Algorithm

##### 4.1. The HDDMF Algorithm

为了加速我们模型的学习过程（Learning Process），我们引入了一种通过逐层分解数据矩阵的预训练策略（Pre-training Tactic）。这种预训练策略的有效性在之前关于深度神经网络（Deep Neural Networks）的研究 [47] 中已得到证实。对于每个视图 $v$，我们首先近似分解数据矩阵 $X^v \approx Z_1^v H_1^v$，其中 $Z_1^v \in \mathbb{R}^{q^v \times p_1}$ 且 $H_1^v \in \mathbb{R}_{+}^{p_1 \times n}$。然后，表示矩阵（Representation Matrix）$H_1^v$ 被进一步近似分解为 $H_1^v \approx Z_2^v H_2^v$，其中 $Z_2^v \in \mathbb{R}^{p_1 \times p_2}$ 且 $H_2^v \in \mathbb{R}_{+}^{p_2 \times n}$。需要注意的是，预训练过程完全通过最小化半非负矩阵分解（Semi-NMF）[48] 来进行，我们持续这样做直到所有层都被预训练。之后，我们通过交替最小化（Alternating Minimization）公式 (14) 中提出的代价函数（Cost Function）来微调（Fine-tune）所有变量。

1. **基矩阵（Basis Matrix）$Z_i^v$ 的更新规则**：首先，我们在第 $i$ 层固定第 $v$ 个视图中的其他变量并最小化 $Z_i^v$。该子问题（Subproblem）可以简化为

$$O(Z_i^v) = \| X^v - Z_1^v \cdots Z_m^v H_m^v \|_F^2. \tag{15}$$

$O$ 关于 $Z_i^v$ 的导数计算如下：

$$\frac{\partial O(Z_i^v)}{\partial Z_i^v} = \Phi_{i-1}^{v T} \Phi_{i-1}^v Z_i^v \tilde{H}_i^v \tilde{H}_i^{v T} - \Phi_{i-1}^{v T} X^v \tilde{H}_i^{v T} \tag{16}$$

其中 $\Phi_{i}^v = \prod_{j=1}^i Z_j^v$ 表示前 $i$ 个基矩阵的乘积，$\tilde{H}_i^v$ 表示第 $i$ 层特征矩阵的重构（Reconstruction），即 $\tilde{H}_i^v = Z_{i+1}^v \cdots Z_m^v H_m^v$。通过令 $\frac{\partial O(Z_i^v)}{\partial Z_i^v} = 0$，更新规则可给出为

$$Z_i^v = \Phi_{i-1}^{v \dagger} X^v \tilde{H}_i^{v \dagger} \tag{17}$$

其中 $\dagger$ 表示摩尔-彭若斯伪逆（Moore-Penrose Pseudo-inverse）。

1. **特征矩阵（Feature Matrix）$H_m^v$ 的更新规则**：关于 $H_m^v$ 的优化 (14) 等价于最小化以下内容：

$$\begin{aligned} O(H_m^v) &= \| X^v - Z_1^v \cdots Z_m^v H_m^v \|_F^2 + \beta \text{tr}(H_m^v L_h^v H_m^{v T}) \\ &\quad + \mu \sum_{w=1; v \neq w}^V \text{tr}(H_m^{v T} H_m^v H_m^{w T} H_m^w) \end{aligned} \tag{18}$$

$$\text{s.t. } H_m^v \ge 0.$$

对于约束 $H_m^v \ge 0$，我们引入拉格朗日乘子（Lagrangian Multipliers）$\varphi^v$ 和拉格朗日函数（Lagrange Function）$\mathcal{L}$ 如下

$$\begin{aligned} \mathcal{L}(H_m^v) &= \text{tr}\left( X^{v T} X^v - 2 H_m^{v T} \Phi_m^{v T} X^v + H_m^{v T} \Phi_m^{v T} \Phi_m^v H_m^v \right) \\ &\quad + \beta \text{tr}(H_m^v L_h^v H_m^{v T}) + \mu \sum_{v \neq w}^V \text{tr}(H_m^{v T} H_m^v H_m^{w T} H_m^w) \\ &\quad - \text{tr}(\varphi^v H_m^v). \end{aligned} \tag{19}$$

$O$ 关于 $H_m^v$ 的导数计算如下：

$$\begin{aligned} \frac{\partial \mathcal{L}(H_m^v)}{\partial H_m^v} &= -2 \Phi_m^{v T} X^v + 2 \Phi_m^{v T} \Phi_m^v H_m^v + 2 \beta H_m^v L^v \\ &\quad + 2 \mu \sum_{v \neq w}^V H_m^v H_m^{w T} H_m^w - \varphi^v. \end{aligned} \tag{20}$$

利用与 [48] 类似的证明，并令偏导数 $\frac{\partial \mathcal{L}(H_m^v)}{\partial H_m^v} = 0$ 且 $\varphi_{kl}^v (H_m^v)_{kl} = 0$，我们可以制定更新规则如下：

$$H_m^v = H_m^v \odot \sqrt{\frac{[\Phi_m^{v T} \Phi_m^v H_m^v]^- + [\Phi_m^{v T} X^v]^+ + \beta [H_m^v L^v]^-}{[\Phi_m^{v T} \Phi_m^v H_m^v]^+ + [\Phi_m^{v T} X^v]^- + \beta [H_m^v L^v]^+ + \mu \Psi}} \tag{21}$$

其中 $\Psi = \sum_{w=1, v \neq w}^V H_m^v H_m^{w T} H_m^w$，$[Q]^-$ 表示将所有正元素替换为 0 的矩阵，$[Q]^+$ 表示将负元素替换为 0 的矩阵

$$\forall k, l. \quad [Q]_{kl}^+ = \frac{|Q_{kl}| + Q_{kl}}{2}, \quad [Q]_{kl}^- = \frac{|Q_{kl}| - Q_{kl}}{2}. \tag{22}$$

我们在算法 1 中总结了 HDDMF 的整个优化过程（Optimization Procedure）。我们使用收敛规则（Convergence Rule）$O_{k} - O_{k-1} \le \xi \max(1, O_k)$（其中 $\xi = 10^{-4}$）来在当前更新与前一次更新之间的目标值（Objective Value）$(O_k)$ 足够小时终止迭代过程（Iteration Process）。

![image-20251230194117801](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230194117801.png)

##### 4.2. Convergence of the Algorithm

在本节中，我们证明更新规则 (17) 和 (21) 的收敛性 (Convergence)。

*定理 1：* 固定 $H_m^v$，针对 $Z_i^v$ 的更新规则 (17) 给出了最小化目标函数 (Objective Function) 的最优解 (Optimal Solution)。

*证明：* 为了证明定理 1，我们固定其余的因子并计算 $\frac{\partial O(Z_i^v)}{\partial Z_i^v} = 0$。$Z_i^v$ 的解可以通过以下方式获得

$$Z_i^v = (\Phi_{i-1}^{v T} \Phi_{i-1}^v)^{-1} (\Phi_{i-1}^{v T} X^v \tilde{H}_i^{v T}) (\tilde{H}_i^v \tilde{H}_i^{v T})^{-1}$$

$$Z_i^v = \Phi_{i-1}^{v \dagger} X^v \tilde{H}_i^{v \dagger} \tag{23}$$

其中 $\Phi_i^v = Z_1^v \cdots Z_i^v$ 且 $\tilde{H}_i^v = Z_{i+1}^v \cdots Z_m^v H_m^v$。 $\blacksquare$

*定理 2：* 固定 $Z_i^v$，目标函数在针对任意层 $i$ 和任意视图 $v$ 的 $Z_i^v$ 更新规则下单调递减 (Monotonically Decreases)（即，它是非递增的 (Nonincreasing)）。

*证明：* 为了证明定理 2，我们固定所有的基矩阵 (Basis Matrices) $Z_i^v$ 并在非负约束 (Non-negative Constraint) 下求解 $H_m^v$。我们从两个方面分析收敛性：1) 为了证明方程更新规则解的最终因子是正确的，我们展示了其在收敛时满足 Karush-Kuhn-Tucker (KKT) 条件 (Condition)。2) 我们展示了 (21) 中给出的更新规则的迭代是收敛的。

*命题 1：* $H_m^v$ 更新规则的极限解 (Limiting Solution) 满足 KKT 条件。

*证明：* 拉格朗日函数 (Lagrangian Function) $\mathcal{L}$ 的梯度 (Gradient) 在 (20) 中给出。通过设 $H_m^v$ 的梯度为 0 并利用互补松弛条件 (Complementary Slackness Condition)，我们可以得到

$$\left( -2\Phi_m^{v T} X^v + 2\Phi_m^{v T} \Phi_m^v H_m^v + 2\beta H_m^v L^v + 2\mu \Psi \right)_{kl} (H_m^v)_{kl} = \varphi_{kl}^v (H_m^v)_{kl} = 0. \tag{24}$$

这是一个不动点方程 (Fixed-point Equation)，这意味着解必须满足收敛要求。此外，我们发现更新规则 (21) 的极限解满足该不动点方程。在收敛时，$H_m^{v(\infty)} = H_m^{v(t+1)} = H_m^{v(t)} = H_m^v$，即

$$H_m^v = H_m^v \odot \sqrt{\frac{[\Phi_m^{v T} \Phi_m^v H_m^v]^- + [\Phi_m^{v T} X^v]^+ + \beta [H_m^v L^v]^-}{[\Phi_m^{v T} \Phi_m^v H_m^v]^+ + [\Phi_m^{v T} X^v]^- + \beta [H_m^v L^v]^+ + \mu \Psi}}. \tag{25}$$

注意

$$\Phi_m^{v T} X^v = [\Phi_m^{v T} X^v]^+ - [\Phi_m^{v T} X^v]^-$$

$$\Phi_m^{v T} \Phi_m^v H_m^v = [\Phi_m^{v T} \Phi_m^v H_m^v]^+ - [\Phi_m^{v T} \Phi_m^v H_m^v]^-$$

$$H_m^v L^v = [H_m^v L^v]^+ - [H_m^v L^v]^-. \tag{26}$$

因此，(25) 简化为

$$\left( -2\Phi_m^{v T} X^v + 2\Phi_m^{v T} \Phi_m^v H_m^v + 2\beta H_m^v L^v + 2\mu \Psi \right)_{kl} (H_m^v)_{kl}^2 = 0. \tag{27}$$

方程 (27) 等价于 (24)。尽管 (27) 和 (24) 中的第一个因子是相同的，对于第二个因子，如果 $(H_m^v)_{kl} = 0$ 则 $(H_m^v)_{kl}^2 = 0$，反之亦然。因此，如果 (24) 成立，(27) 也成立，反之亦然。 $\blacksquare$

*命题 2：* (18) 的残差 (Residual) 在 $H_m^v$ 的更新规则 (21) 下单调递减。

*证明：* 为了证明命题 2，我们将 $O(H_m^v)$ (18) 重写为

$$\begin{aligned} O(G) &= \text{tr}(-2G^T B^+ + 2G^T B^- + G A^+ G^T - G A^- G^T \\ &\quad + \beta G L^+ G^T - \beta G L^- G^T + \mu \Psi) \end{aligned} \tag{28}$$

其中 $A = \Phi_m^{v T} \Phi_m^v$，$B = \Phi_m^{v T} X^v$ 且 $G = H_m^v$。

作为 [49] 中的辅助函数方法 (Auxiliary Function Method)，我们将 $O(G)$ 的辅助函数记为 $Z(G, G')$，对于任意 $G$ 和 $G'$，如果它满足

$$Z(G, G') \ge O(G), \quad Z(G, G) = O(G). \tag{29}$$

如果 $Z$ 是辅助函数，则 $O$ 在更新 $G = \arg\min_G Z(G, G')$ 下是一个非递增函数 (Nonincreasing Function)。因此，我们有 $O(G) = Z(G, G) \ge Z(G', G) \ge O(G')$。因此，如 [48] 所述，我们需要构造一个满足要求的合适的辅助函数 $Z$。

$$\begin{aligned} Z(G, G') &= -2 \sum_{kl} B_{kl}^+ G_{kl}' \left( 1 + \log \frac{G_{kl}}{G_{kl}'} \right) + 2 \sum_{kl} B_{kl}^- \frac{G_{kl}^2 + G_{kl}'^2}{G_{kl}'} \\ &\quad + \sum_{kl} \frac{(G' A^+)_{kl} G_{kl}^2}{G_{kl}'} - \sum_{kll} A_{l\ell}^- G_{kl}' G_{k\ell}' \left( 1 + \log \frac{G_{kl} G_{k\ell}}{G_{kl}' G_{k\ell}'} \right) \\ &\quad + \beta \sum_{kl} \frac{(G' L^+)_{kl} G_{kl}^2}{G_{kl}'} - \beta \sum_{kll} L_{l\ell}^- G_{kl}' G_{k\ell}' \left( 1 + \log \frac{G_{kl} G_{k\ell}}{G_{kl}' G_{k\ell}'} \right) \\ &\quad + \mu \Psi. \end{aligned} \tag{30}$$

如附录 (Appendix) 所证，$Z$ 是 $O$ 的这样一个函数并且满足必要条件。此外，$Z(G, G')$ 是关于 $G$ 的凸函数 (Convex Function)，其全局最小值 (Global Minimum) 为

$$\begin{aligned} G_{kl} &= \arg\min_G Z(G, G') \\ &= G_{kl}' \odot \sqrt{\frac{[B]_{kl}^+ + [AG']_{kl}^- + \beta [G'L]_{kl}^-}{[B]_{kl}^- + [AG']_{kl}^+ + \beta [G'L]_{kl}^+ + \mu \Psi}}. \end{aligned} \tag{31}$$

##### 4.3. Time Complexity Analysis

为了直观地进行分析，我们假设所有层的大小 $p$ 均相同。我们将 $q^v$ 记为第 $v$ 个视图的原始特征维度 (Original Feature Dimension)，其中 $V$ 是视图的数量，$m$ 是层数。值得注意的是，HDDMF 的优化 (Optimization) 由两个阶段组成，即预训练 (Pre-training) 和微调 (Fine-tuning)。在预训练阶段，半非负矩阵分解 (Semi-NMF) 过程是耗时 (Time Consuming) 的部分。其复杂度 (Complexity) 为 $T_{\text{pre}} = O(\sum_{v}^V (m t_p (q^v n p + n p^2 + p (q^v)^2 + p n^2)))$，其中 $t_p$ 是该预训练阶段的迭代次数 (Number of Iterations)。在微调阶段，超图矩阵 (Hypergraph Matrix) $\{L_h^v\}_v^V$ 的构建以及 $\{H_m^v\}_v^V$ 的优化过程 (Optimization Procedures) 是两个计算密集型 (Computation-intensive) 步骤。就超图而言，对于第 $v$ 个视图，它需要 $O(pn \log(n))$ 的时间复杂度。因此，对于微调阶段，其复杂度为 $T_{\text{fine}} = O(\sum_{v}^V (m t_f (pn \log(n) + q^v n p + p (q^v)^2 + p n^2)))$，其中 $t_f$ 是微调中的迭代次数。

### 5. Experimental Results and Analysis

##### 5.1. Experimental Setup

所使用的数据集和评估指标描述如下：

​	1）数据集 (Data Sets)：Prokaryotic 数据集包含 551 个原核生物样本 (Prokaryotic samples)，具有三个视图：文本特征 (Textual features) 和两类基因组表示 (Genomic representations)。

Caltech101-7 [50] 是 Caltech101 的一个子集 (Subset)，它包含来自 7 个广泛使用类别的 1474 张图像，即 Windsor-Chair、Motorbikes、Dolla-Bill、Snoopy、Garfield、Stop-Sign 和 Face。遵循 [51] 的方法，我们提取了 6 种特征，即 Gabor、小波矩 (Wavelet moments)、CENTRIST、HOG、GIST 和局部二值模式 (LBP) 特征。它们的维度 (Dimensions) 分别为 48、40、254、1984、512 和 928。

ORL 由来自 40 个不同个体 (Individuals) 的 400 张人脸图像组成。为了构建多视图数据集 (Multi-view data sets)，与 [40] 类似，我们提取了三种类型的特征，包括维度为 4096 的强度 (Intensity)、维度为 3304 的 LBP 和维度为 6750 的 Gabor 特征。

Extended YaleB [52] 包含 10 个对象 (Subjects) 在 65 种不同姿态/光照条件 (Poses/Illumination conditions) 下的 650 张图像。该数据集包含三种类型的特征（强度、LBP、Gabor）。

STL10 [53] 是一个图像数据集，包含 10 个类别，即鸟、飞机、猫、卡车、汽车、狗、猴子、船、鹿和马。然后我们从每个类别中采样 1300 个样本（注：此处原文表述如此，请核对是否为总数或每类采样数），并将强度 (Intensity)、HOG 特征和 LBP 特征构建为三个视图。

我们在表 II 中总结了数据集的重要统计细节 (Statistical details)。

​	2）评估指标 (Evaluation Measures)：为了评估我们方法的性能 (Performance)，采用了以下指标：准确率 (Accuracy, ACC)、归一化互信息 (Normalized Mutual Information, NMI)、纯度 (Purity)、调整兰德指数 (Adjusted Rank Index, AR)（注：学术界通常为 Adjusted Rand Index，原文此处写作 Rank，但语境应指 Rand）、F-分数 (F-score)、精确率 (Precision) 和召回率 (Recall)。由于每个指标惩罚聚类中的不同属性 (Properties)，我们报告所有指标的结果以进行综合评估 (Comprehensive evaluation)。

![image-20251230194423455](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230194423455.png)

##### 5.2. Compared Methods

接下来，我们将提出的方法与以下最先进的 (State-of-the-art) 聚类方法进行比较：

​	1）多视图 NMF (Multi-View NMF, MultiNMF) [21]：这是一种经典的多视图 NMF 方法，它构建了一个联合一致性矩阵 (Joint Consensus Matrix) 学习过程，并获得有意义且可比较的聚类结果。根据原论文的建议，我们将唯一的参数 $\lambda$ 设置为 0.01。

​	2）局部保持多样性 NMF (Locality Preserved Diverse NMF, LP-DiNMF) [38]：这是一种浅层 NMF (Shallow NMF) 方法，它同时保持跨多视图的局部几何结构 (Local Geometry Structure) 和多样性 (Diversity)。按照建议，我们在 $\{0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000\}$ 范围内搜索参数。

​	3）具有协同正交约束的 NMF (NMF With Co-Orthogonal Constraints, NMFCC) [54]：基于 LP-DiNMF，NMFCC 额外对学习到的基矩阵 (Basis Matrices) 和表示矩阵 (Representation Matrices) 施加了正交约束。根据作者的建议，我们将所有数据集的参数 $\{\lambda, \delta\}$ 经验性地设置为 $\{70, 1\}$。

​	4）2CMV [55]：一种最近提出的基于分解的多视图模型，利用 CMF 和 NMF，能够利用多视图数据的一致性 (Consensus) 和互补信息 (Complementary Information)。我们按照论文推荐设置参数 $\gamma$ 和 $\beta$。

​	5）多视图深度矩阵分解 (Multi-View Deep Matrix Factorization, MDMF) [30]：该方法通过固定多视图间共享的单侧最终表示 (Shared One-sided Final Representation)，将深度半非负矩阵分解 (Deep Semi-NMF) 从单视图扩展到多视图。两个超参数 $\lambda$ 和 $\beta$ 按照作者建议分别设置为 0.5 和 0.1。

​	6）自加权多视图 DMF (Self-Weighted Multi-View DMF, SMDMF) [31]：SMDMF 是 MDMF 的无超参数 (Hyper-parameters-free) 版本，能够自动获得每个视图的适当权重。

​	7）部分共享 DMF (Partially Shared DMF, PSDMF) [32]：最近提出的一种部分共享结构，能够发现不同视图间的视图特有 (View-specific) 特征和公共特征。为了进行无监督场景 (Unsupervised Scene) 的实验，参数 $\mu$ 设置为 0.1，$\beta$ 设置为 0。

​	8）非多样性 DMF (Non-Diverse DMF, NdDMF)：NdDMF 的实现方式是对每个视图使用深度半非负矩阵分解 (Deep Semi-NMF) [26]（如公式 (5) 所示），然后通过谱聚类 (Spectral Clustering) 对最终表示的组合进行聚类。我们还进行了一个超图正则化版本 (Hypergraph-regularized Version)（称为 HNdDMF）以研究流形正则化的有效性。

​	9）我们的方法 (Our Methods)：我们执行了两个版本的超图正则化多样性诱导 (Diversity-induced) DMF。第一个版本结合了如 (11) 所述的通用多样性约束 (Common Diversity Constraint)（称为 HDDMF-DI）。第二个版本是 HDDMF，它包含了如 (13) 所述的多样性增强技术 (Diversity Enhancement Technique)。然后我们在学习到的最终表示上应用谱聚类以获得聚类结果。

需要注意的是，LP-DiNMF、NMFCC、MDMF、HNdDMF、HDDMF-DI 和 HDDMF 都需要使用 k-近邻 (k-NN) 构建拉普拉斯图矩阵 (Laplacian Graph Matrix)，其中参数 k 设置为数据类别的数量，如 [38] 中建议的那样。我们的源代码可在 https://github.com/libertyhhn/DiverseDMF 获取。

##### 5.3. Parametric Sensitivity

​	1）层数的影响 (Influence of the Number of Layers)：为了研究模型深度 (Model Depth) 对聚类结果的影响，我们应用了深度从 1 层到 4 层变化的 HDDMF 方法，层大小分别设置为 [50]、[50 100]、[50 100 150] 和 [50 100 150 200]。不同层数的聚类结果如图 3 所示。尽管不同数据集的表现各异，但无论是在 ACC 还是 NMI 方面，我们都可以观察到多层模型 (Multi-layer Model) 的聚类性能在所有五个数据集上均优于单层模型。这证实了多层模型能够探索隐式的层次信息 (Implicit Hierarchical Information)，这有利于聚类。随着层数的增加，模型的聚类效果可能会下降（例如 Prokaryotic），因为模型已处于过拟合 (Over-fitting) 状态。因此，我们为每个数据集采用合适的层结构来进行后续实验。具体而言，我们为 Prokaryotic、Extended YaleB 和 STL10 数据集配置 2 层结构，为 ORL 配置 3 层结构，为 Caltech101-7 配置 4 层结构。

![image-20251230195652111](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230195652111.png)

​	2）流形正则化和多样性约束的影响 (Influence of the Manifold Regularization and Diversity Constraint)：为了分析 (14) 中模块的影响，我们关注两个重要参数 $\beta$ 和 $\mu$。$\beta$ 控制超图正则化 (Hypergraph Regularization) 在学习到的最终表示矩阵中的贡献。$\mu$ 衡量不同视图间表示的多样性程度 (Degree of Diversity)。根据网格搜索策略 (Grid Search Strategy)，它们均从相同的范围 $[0.0001, 0.001, 0.01, 0.1, 1]$ 中选取。在图 4 中，我们分别展示了在 Prokaryotic、Caltech101-7、ORL、Extended YaleB 和 STL10 数据集上，关于 ACC 和 NMI 的参数调优实验结果。从图中我们可以观察到，当 $\beta$ 设置为相对较大的值且 $\mu$ 设置为相对较小的值时，HDDMF 在大多数情况下可以取得最佳结果。

![image-20251230195710187](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230195710187.png)

##### 5.4. Performance Comparison

![image-20251230195735861](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230195735861.png)

![image-20251230195749474](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230195749474.png)

![image-20251230195958871](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230195958871.png)

为了在所有竞争者之间进行公平比较 (Fair Comparison)，我们直接使用作者提供的源代码进行实验验证，并根据原始论文的建议搜索最佳参数。所有程序均在 MATLAB (R2018b) 环境下运行，服务器配置为 Intel(R) Xeon(R) E5-2640 @2.40 GHz CPU 和 128 GB RAM，操作系统为 Linux。对于每种方法，我们使用十次初始化运行并记录结果的均值 (Mean) 和标准差 (Standard Variance)，参考 [30] 中的实验设置。

五个多视图数据集上的聚类性能比较报告在表 III-VII 中。对于所有这些指标，较高的值表示更好的聚类性能，最高值以粗体显示（注：根据你的要求，此处翻译中未加粗）。注意，Multi-NMF、LP-DiNMF、NMFCC 和 2CMV 都要求只能处理非负数据 (Non-negative Data)。因此，它们无法处理具有负像素的数据集（例如 Prokaryotic 和 Caltech101-7）。且上述非负方法在 Prokaryotic 和 Caltech101-7 上的结果不可用。从这些表格中，我们可以得出以下观察结果：

​	1）总体而言，提出的 HDDMF 在所有数据集上均取得了最佳结果，除了 Prokaryotic 数据集上的 NMI 和 AR 指标。以 STL10 数据集为例；相对于第二好的方法 MDMF，我们的方法在七个指标上分别提升了约 1.6%、2.3%、2.43%、1.78%、1.33%、1.35% 和 2.73%。这主要是因为我们要提出的方法在一个统一模型 (Unified Model) 中使用了三个方面：a) 来自不同视图的表示的结构互补性 (Structural Complementarity)；b) 样本间的高阶关系 (High-order Relationship)；c) 用于发现层次信息的深度表示 (Deep Representation)。

​	2）基于深度矩阵分解的 MDR 方法（MDMF、SMDMF、PSDMF 和提出的 HDDMF(-DI)）在大多数情况下显示出比基于单层矩阵分解的 MDR 方法（MultiNMF、LP-DiNMF、NMFCC 和 2CMV）更好的结果。原因可能是通过深度分解，模型可以消除一些不利因素 (Adverse Factors) 并保留最终表示中的身份信息 (Identity Information)。

​	3）可以看出，具有多样性约束的模型的聚类性能明显优于没有多样性约束的模型。以 ORL 数据集为例，与非多样性方法 HNdDMF 相比，具有多样性约束的方法（HDDMF-DI 和 HDDMF）在所有指标上的性能提升超过 4%。这表明多样性诱导技术 (Diversity-induced Techniques) 能够发现多视图间的相互互补信息，更有利于聚类。

​	4）很明显，HDDMF 在所有数据集上均优于 HDDMF-DI。以 Caltech101-7 数据为例；在 ACC 和 NMI 方面，HDDMF 的领先幅度 (Leading Margins) 分别约为 3% 和 7%。这表明，同时利用样本间和视图间的多样性属性有助于进一步提高模型的表示能力，以实现精确学习。

​	5）在 Extended YaleB 数据上，HNdDMF 取得了比 NdDMF 更好的性能。原因是高阶流形正则化 (High-order Manifold Regularization) 可以在模型学习到的子空间中保留原始数据的局部几何结构。

##### 5.5. The Convergence of the Algorithm

HDDMF 目标函数通过提出的迭代优化方法 (Iterative Optimization Method)（算法 1）求解。我们在第 IV-B 节从理论上证明了其收敛性质 (Convergence Property)。为了通过实验展示 HDDMF 的收敛性，我们记录了每次迭代中的目标值 (14)。ORL、Caltech101-7、Extended YaleB 和 STL10 数据集上的收敛曲线如图 5 所示。我们可以观察到目标函数值急剧下降，并在约 500 次迭代后逐渐达到收敛状态。

![image-20251230195623266](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230195623266.png)

##### 5.6. Visualizations for Embedding Results

Caltech101-7、Extended YaleB 和 ORL 数据集中的嵌入结果可视化分别在图 6-8 中展示。在这里，学习到的嵌入表示矩阵使用 t-SNE [56] 投影到二维子空间。注意，我们在 Caltech101-7 中仅比较基于 DMF 的方法，因为存在负像素。然后我们将来自不同视图的特征直接拼接到原始数据空间。可以观察到，与原始数据空间、两种基于 NMF 的方法以及其他两种基于 DMF 的方法相比，我们提出的 HDDMF 具有更清晰的聚类结构 (Cluster Structure)。

![image-20251230200019409](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230200019409.png)

![image-20251230200032721](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230200032721.png)

![image-20251230200046791](C:\Users\25355\AppData\Roaming\Typora\typora-user-images\image-20251230200046791.png)

### 5. Conclusion

在本文中，我们提出了一种新颖的深度多视图数据表示 (Deep Multi-view Data Representation) 模型 HDDMF，通过最小化正交量化项 (Orthogonality Quantification Term) 来捕获潜在表示结构的多样性 (Diversity)，并集成超图正则化 (Hypergraph Regularization) 以保持嵌入在高维潜在空间 (High-dimensional Latent Space) 中的局部流形结构 (Local Manifold Structure)。为了求解我们方法的优化问题 (Optimization Problem)，我们设计了一种新算法，并提供了收敛性质 (Convergence Property) 的理论分析，证明了所提出的算法能够收敛到目标函数 (Objective Function) 的驻点 (Stationary Point)。在五个真实数据库 (Real Databases) 上的广泛实验结果表明，所提出的方法在聚类挑战 (Clustering Challenge) 上优于最先进的 (State-of-the-art) 多视图学习方法。考虑到多视图数据中通常隐藏着非线性关系 (Nonlinear Relationships) [57]，在未来的工作中 (Future Works)，我们将把 HDDMF 扩展为非线性版本 (Nonlinear Version)，以发现非线性信息并提高模型的学习能力 (Learning Ability)。

### Appendix

为了证明 $Z(G, G')$ 是 $O(G)$ 的辅助函数 (Auxiliary Function)，我们首先引入文献 [58] 中的引理 1 (Lemma 1)：

*引理 1：* 对于任意非负矩阵 (Non-negative Matrices) $Q \in \mathbb{R}^{n \times n}$，$P \in \mathbb{R}^{k \times k}$，$S \in \mathbb{R}^{n \times k}$，$S' \in \mathbb{R}^{n \times k}$，其中 $Q$ 和 $P$ 是对称的 (Symmetric)，以下不等式成立：

$$\sum_{i=1}^n \sum_{p=1}^k \frac{(Q S' P)_{ip} S_{ip}^2}{S'_{ip}} \ge \text{tr}(S^T Q S P). \tag{32}$$

根据 $O(G)$（如 (28) 所示）和 $Z(G, G')$（如 (30) 所示），利用上述引理 1，我们可以得到以下不等式：

$$\text{tr}(G A^+ G^T) \le \sum_{kl} \frac{(G' A^+)_{kl} G_{kl}^2}{G'_{kl}} \tag{33}$$

$$\text{tr}(G L^+ G^T) \le \sum_{kl} \frac{(G' L^+)_{kl} G_{kl}^2}{G'_{kl}}. \tag{34}$$

利用不等式 $a \le \frac{a^2+b^2}{2b}, \forall a, b > 0$，我们还有

$$\text{tr}(G^T B^-) = \sum_{kl} B_{kl}^- G_{kl} \le \sum_{kl} B_{kl}^- \frac{G_{kl}^2 + G'{}_{kl}^2}{2G'_{kl}}. \tag{35}$$

为了获得其余三项的下界 (Lower Bounds)，我们利用不等式 $z \ge 1 + \log z, \forall z > 0$，并得到

$$\text{tr}(G^T B^+) \ge \sum_{kl} B_{kl}^+ G'_{kl} \left( 1 + \log \frac{G_{kl}}{G'_{kl}} \right) \tag{36}$$

$$\text{tr}(G A^- G^T) \ge \sum_{kll} A_{l\ell}^- G'_{kl} G'_{k\ell} \left( 1 + \log \frac{G_{kl} G_{k\ell}}{G'_{kl} G'_{k\ell}} \right) \tag{37}$$

$$\text{tr}(G L^- G^T) \ge \sum_{kll} L_{l\ell}^- G'_{kl} G'_{k\ell} \left( 1 + \log \frac{G_{kl} G_{k\ell}}{G'_{kl} G'_{k\ell}} \right). \tag{38}$$

因此，由 (33) 至 (38)，我们有 $Z(G, G') \ge O(G)$，并且 $Z(G, G) = O(G)$。

接下来，我们对 $G$ 求 $Z(G, G')$（如 (30) 所示）的一阶导数 (First Derivative)，以找到 $Z(G, G')$ 的最小值，得到

$$\begin{aligned} \frac{\partial Z(G, G')}{\partial G_{kl}} &= -2 B_{kl}^+ \frac{G'_{kl}}{G_{kl}} + 2 B_{kl}^- \frac{G_{kl}}{G'_{kl}} \\ &\quad + 2 \frac{(G' A^+)_{kl} G_{kl}}{G'_{kl}} - 2 (A^- G')_{kl} \frac{G'_{kl}}{G_{kl}} \\ &\quad + 2 \beta \frac{(G' L^+)_{kl} G_{kl}}{G'_{kl}} - 2 \beta (G' L^-)_{kl} \frac{G'_{kl}}{G_{kl}}. \end{aligned} \tag{39}$$

对 $G$ 求 $Z(G, G')$ 的二阶导数 (Second Derivative)，我们有

$$\begin{aligned} \frac{\partial^2 Z(G, G')}{\partial G_{kl} \partial G_{j\ell}} &= \delta_{kl} \delta_{j\ell} \left( 2 B_{kl}^+ \frac{G'_{kl}}{G_{kl}^2} + 2 \frac{B_{kl}^-}{G'_{kl}} \right. \\ &\quad \left. + 2 \frac{(G' A^+)_{kl}}{G'_{kl}} + 2 (A^- G')_{kl} \frac{G'_{kl}}{G_{kl}^2} \right. \\ &\quad \left. + 2 \beta \frac{(G' L^+)_{kl}}{G'_{kl}} + 2 \beta (G' L^-)_{kl} \frac{G'_{kl}}{G_{kl}^2} \right). \end{aligned} \tag{40}$$

因此，由 (40) 可知，海森矩阵 (Hessian Matrix) 是一个具有正元素的对角矩阵。因此，$Z$ 是关于 $O$ 的凸函数 (Convex Function)，其全局最小值可以通过在 (39) 中令 $\frac{\partial Z(G, G')}{\partial G_{kl}} = 0$ 获得。经过变换 (Transformation) 后，我们可以得到 (25)。