# Mall-Customer-Clustering
Retail customer segmentation using K-Means with Z-score scaling and custom feature weighting (Gender=0.2). Includes ablation studies to address dimension dominance in high-dimensional clustering.

-----

# 零售商场客户行为的多维聚类分析与特征权重优化
本项目基于 Python 开发，旨在通过改进后的 **K-Means** 算法对商场会员数据进行深度挖掘。通过执行消融实验，本项目论证了标准化处理与特征加权在消除“维度霸凌”及提升聚类商业解释力方面的关键作用。

## 🚀 快速开始

### 1\. 环境安装

本项目依赖 `requirements.txt` 中指定的库版本。建议在虚拟环境中运行：

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2\. 运行脚本

确保 `Mall_Customers.csv` 与脚本处于同一目录下，执行：

```bash
python mall_clustering.py
```

-----

## 💻 跨平台兼容性说明 (重要)

为了确保实验报告在不同操作系统下的展示效果一致，脚本集成了以下兼容性逻辑：

### 1\. 字体显示：解决中文乱码

由于 Windows 和 Mac 系统自带的中文字体库不同，脚本会自动识别系统并加载对应字体，确保图表标签（如“年收入”）正常显示：

  * **Mac (Darwin)**: 优先调用 `Arial Unicode MS`。
  * **Windows**: 优先调用 `SimHei` (黑体)。

> **注意**：若在 Linux 或其他环境运行，脚本将回退至系统默认无衬线字体。

### 2\. 路径处理：避免文件未找到

脚本采用动态路径获取方案（`os.path`），无论您在终端的哪个路径下启动程序，脚本都会自动锁定 `.py` 文件所在的目录来寻找 `Mall_Customers.csv`。

-----

## 🔬 核心研究逻辑

本项目通过对原始 K-Means 算法的预处理流程进行优化，提升了模型对多维特征的感知力：

1.  **量纲对齐 (Scaling)**：
    使用 `StandardScaler` 执行 Z-score 标准化。若不执行此步，年收入（量级为 100）将彻底掩盖年龄（量级为 10）对距离计算的影响。

2.  **特征权重微调 (Feature Weighting)**：
    针对性别编码后的二元特征，引入 **0.2 衰减因子**。

      * **目的**：压缩性别维度的空间跳跃感，防止算法仅因性别不同就将行为相似的客户强行拆分。
      * **结果**：确保了聚类边界由“消费行为”驱动，而非简单的性别标签。

3.  **模型评估**：
    利用 **肘部法则 (Elbow Method)** 确定最佳聚类数 $K=5$。

-----

## 📊 实验结论：五大核心客群画像

| 客群标签 | 平均年龄 | 消费特征 | 业务定义 |
| :--- | :--- | :--- | :--- |
| **精英主力军** | 32.7 | 高收入、高消费 | 核心利润贡献者 |
| **潜力青年派** | 25.3 | 低收入、高消费 | 潮流风向标，高增长潜力 |
| **务实银发层** | 55.6 | 中等收入、消费稳定 | 忠诚度高，关注品质与售后 |
| **都市平价族** | 35.0 | 中等收入、女性主导 | 高频到店，价格敏感型 |
| **理性高净值** | 41.1 | 极高收入、低评分 | 潜在的高端定制客户 |

-----

根据你最新的文件命名（`mall_customer_segmentation.py`）以及新增的中间过程输出需求，我为你更新了 `README.md` 的**文件清单**章节。

你可以直接将以下内容替换到 `README.md` 的对应位置：

---

## 📂 文件清单 (File List)

为了确保实验的严谨性与复现性，本项目包含以下核心文件：

### 1. 核心执行文件
* **`mall_clustering.py`**：经过优化的主程序脚本，包含程数据处理、模型训练及消融实验可视化。
* **`mall_customer_segmentation.ipynb`**：包含研究过程的全部代码（流程数据处理、模型训练及消融实验可视化等），以Jupyter Notebook 形式记录。
* **`Mall_Customers.csv`**：原始数据集，需与脚本置于同一目录下。
* **`requirements.txt`**：环境依赖清單，記錄了 Python 3.12.6 环境下的核心庫版本。

### 2. 关键中间过程输出 (Outputs)
* **`mall_customers_clustered_csvdata`**：档案下的包含代码运行的关键中间过程。
* **`scaled_features_matrix.csv`**：经过 Z-score 标准化及 0.2 性别权重微调后的特征矩阵。
* **`mall_customers_with_labels.csv`**：最终生成的带聚类标签的全量数据集。
* **`cluster_centroids.csv`**：还原至原始物理量纲后的五大客群质心坐标。

### 3. 文档资料
* **`README.md`**：项目技术说明、环境配置及运行指南。

---

**作者**：SEONGWUNC
**完成日期**：2026年3月25日

-----
