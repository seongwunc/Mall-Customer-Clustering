import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# 1. 实验初始化与环境配置
# ==========================================
# 设置中文字体（Mac 系统通用，Windows 用户建议改为 'SimHei'）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False 

path = 'Mall_Customers.csv'

try:
    df = pd.read_csv(path)
    print(f"✅ 数据加载成功！规模为: {df.shape}")
except FileNotFoundError:
    print("❌ 错误：未找到 Mall_Customers.csv 文件，请检查路径。")
    # 如果文件不存在，后续代码将无法运行，此处可根据需要添加 exit()

# ==========================================
# 2. 数据预处理与特征工程
# ==========================================
print(f"缺失值总数: {df.isnull().sum().sum()}")
print(f"重复样本总数: {df.duplicated().sum()}")

# [图 1] 基础特征分布检查
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(df['Age'], bins=10, kde=True, color='skyblue', ax=axes[0])
axes[0].set_title('客户年龄分布')
sns.histplot(df['Annual Income (k$)'], bins=10, kde=True, color='salmon', ax=axes[1])
axes[1].set_title('年收入分布 (k$)')
sns.histplot(df['Spending Score (1-100)'], bins=10, kde=True, color='olive', ax=axes[2])
axes[2].set_title('消费评分分布 (1-100)')
plt.tight_layout()
plt.show()

# [图 2] 性别基准分布
gender_pct = df['Gender'].value_counts(normalize=True) * 100
global_female_pct = gender_pct.get('Female', 0)
plt.figure(figsize=(8, 5))
plt.pie(gender_pct, labels=gender_pct.index, autopct='%1.1f%%', 
        startangle=140, colors=['#ff9999','#66b3ff'], explode=(0.05, 0))
plt.title(f'原始数据集：性别基准分布\n(女性基准线: {global_female_pct:.1f}%)')
plt.show()

# 特征工程执行
df_final = pd.get_dummies(df, columns=['Gender'], prefix='Gender')
X = df_final.drop(['CustomerID'], axis=1) 

# [核心步骤] Z-score 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# [核心步骤] 引入性别权重因子 (0.2)
gender_weight = 0.2
X_scaled[:, 3:] = X_scaled[:, 3:] * gender_weight
print(f"✅ 特征工程完成，已将性别特征权重调整为: {gender_weight}")

# [图 3] 特征相关性热力图
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('商场客户特征相关性热力图')
plt.show()

# ==========================================
# 3. 模型构建与评估
# ==========================================
# [图 4] 肘部法则确定 K 值
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='random', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', color='purple', linestyle='--')
plt.title('肘部法则图 (Elbow Method)')
plt.xlabel('聚类数量 K')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# 最终模型训练 (K=5)
k_best = 5 
kmeans_final = KMeans(n_clusters=k_best, init='k-means++', n_init=10, random_state=42)
df_final['Cluster'] = kmeans_final.fit_predict(X_scaled)
wcss_imp = kmeans_final.inertia_

# ==========================================
# 4. 聚类结果可视化
# ==========================================
# [图 5] 三维空间全景图
centers_scaled = kmeans_final.cluster_centers_
centers_orig = scaler.inverse_transform(centers_scaled) # 还原质心

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_final['Annual Income (k$)'], 
                     df_final['Spending Score (1-100)'], 
                     df_final['Age'], 
                     c=df_final['Cluster'], 
                     cmap='viridis', s=60, alpha=0.7, edgecolors='w')

ax.scatter(centers_orig[:, 1], centers_orig[:, 2], centers_orig[:, 0], 
           s=350, c='red', marker='X', edgecolors='black', label='聚类质心', zorder=15)

ax.set_xlabel('年收入 (k$)')
ax.set_ylabel('消费评分 (1-100)')
ax.set_zlabel('年龄 (Age)')
ax.set_title(f'K={k_best} 三维空间客群细分全景图')
ax.legend(loc='lower right')
ax.view_init(elev=20, azim=45)
plt.show()

# [图 6] 二维分布与性别占比分析
plt.figure(figsize=(15, 6))
if '性别' not in df_final.columns:
    df_final['性别'] = df['Gender']

# 子图 1：收入/评分分布
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_final, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', palette='viridis', s=50, alpha=0.6)
plt.scatter(centers_orig[:, 1], centers_orig[:, 2], 
            s=300, c='red', marker='X', edgecolors='black', label='聚类质心')
plt.title('年收入与消费评分散点图')

# 子图 2：性别比例堆叠图
plt.subplot(1, 2, 2)
cluster_gender_counts = df_final.groupby(['Cluster', '性别']).size().unstack(fill_value=0)
cluster_gender_pct = cluster_gender_counts.div(cluster_gender_counts.sum(axis=1), axis=0) * 100
cluster_gender_pct.plot(kind='bar', stacked=True, color=['#ff9999','#66b3ff'], ax=plt.gca())

# 添加百分比标注
for i, (idx, row) in enumerate(cluster_gender_pct.iterrows()):
    cum_h = 0
    for col in cluster_gender_pct.columns:
        h = row[col]
        if h > 5: # 占比太小则不显示文字
            plt.text(i, cum_h + h/2, f'{h:.1f}%', ha='center', va='center', color='white', fontweight='bold')
        cum_h += h

plt.title('各聚类簇内部性别比例分析 (%)')
plt.xlabel('Cluster')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ==========================================
# 5. 业务标签映射与特性总结
# ==========================================
# [修复点] 定义并计算 analysis 变量
analysis = df_final.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Female']].mean()

cluster_labels = {
    4: "精英主力军",
    2: "潜力青年派",
    1: "都市平价族",
    0: "务实银发层",
    3: "理性高净值"
}

readable_analysis = analysis.copy()
readable_analysis['业务标签'] = readable_analysis.index.map(cluster_labels)
readable_analysis = readable_analysis[['业务标签', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Female']]
readable_analysis.columns = ['客群标签', '平均年龄', '年收入', '消费评分', '女性占比']

print("\n--- 5.1 聚类客群业务画像分析 ---")
print(readable_analysis.sort_values(by='消费评分', ascending=False))

# ==========================================
# 6. 消融实验对比 (Baseline vs Improved)
# ==========================================
# 计算基线模型 (未缩放、未加权)
X_base = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
kmeans_base = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
df['Cluster_Base'] = kmeans_base.fit_predict(X_base)
wcss_base = kmeans_base.inertia_

# 确保对比列名一致
df_final['Cluster_Improved'] = df_final['Cluster']

# [图 7] 最终消融对比图 (2D 投影)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# --- 左：基线模型 ---
base_centers = df.groupby('Cluster_Base')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster_Base', palette='tab10', s=120, ax=ax1, alpha=0.6, edgecolor='w')
ax1.scatter(base_centers['Annual Income (k$)'], base_centers['Spending Score (1-100)'], 
            s=350, c='red', marker='X', edgecolors='black', label='基线质心')
ax1.set_title('基线对比：原始数据直投 (维度霸凌显现)', fontsize=15)
ax1.grid(True, linestyle='--', alpha=0.4)

# --- 右：改进模型 ---
sns.scatterplot(data=df_final, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster_Improved', style='Cluster_Improved', palette='viridis', s=100, alpha=0.6, ax=ax2)
ax2.scatter(centers_orig[:, 1], centers_orig[:, 2], 
            s=300, c='red', marker='X', edgecolors='black', label='聚类质心')
ax2.set_title('改进对比：标准化 + 权重微调 (多维特征平衡)', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

# [打印对比报告]
print("\n" + "="*60)
print(f"{'指标维度':<15} | {'基线模型 (Baseline)':<20} | {'改进模型 (Improved)':<20}")
print("-" * 60)
print(f"{'WCSS 误差值':<15} | {wcss_base:<20.2f} | {wcss_imp:<20.2f}")
print(f"{'特征处理':<15} | {'原始数值 (Raw)':<20} | {'标准化 + 权重微调':<20}")
print(f"{'分类逻辑':<15} | {'收入轴垂直分割':<20} | {'多维空间协同判定':<20}")
print("-" * 60)
print("改进模型各簇性别分布验证：")
gender_comparison = df_final.groupby('Cluster_Improved')['性别'].value_counts(normalize=True).unstack() * 100
print(gender_comparison.round(1).astype(str) + '%')
print("="*60)