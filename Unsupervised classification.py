import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from data_processing import load_jieba, preprocess_text, load_and_preprocess_data, clean_data  

# 获取每个聚类的Top关键词
def _get_top_keywords_for_clusters(X, vectorizer, kmeans, n_terms, excluded_keywords):
    labels = kmeans.labels_
    df_keywords = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    df_keywords['Cluster'] = labels
    cluster_keywords = {}
    for i in range(kmeans.n_clusters):
        # 计算每个聚类中词语的平均TF-IDF值
        words = df_keywords[df_keywords.Cluster == i].mean(axis=0).sort_values(ascending=False)
        # 移除停用词
        words_filtered = words.drop(labels=excluded_keywords, errors='ignore')
        cluster_keywords[i] = words_filtered.head(n_terms).index.tolist()
    return cluster_keywords

# K均值聚类分析
def perform_kmeans_clustering(df, n_clusters, excluded_keywords, n_top_keywords):
    # 使用 TF-IDF 向量化文本数据
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_content'])

    # 执行 KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') #设置随机种子，保证结果可复现
    kmeans.fit(X)

    # 将聚类标签添加到 DataFrame
    df['Cluster'] = kmeans.labels_

    # 获取每个聚类的 Top 关键词
    cluster_keywords = _get_top_keywords_for_clusters(X, vectorizer, kmeans, n_top_keywords, excluded_keywords)
    return df, cluster_keywords, vectorizer, kmeans, X

# 评估最佳聚类数（肘部法则和轮廓系数）
def evaluate_optimal_k(X, k_range):
    distortions = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)  # 惯性（inertia）：样本到最近的聚类中心的平方距离之和
        if k > 1:
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        else:
            silhouette_scores.append(0)  # k=1 时轮廓系数无意义

    # 绘制图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    sns.lineplot(x=k_range, y=distortions, marker='o', ax=ax1)
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Distortion (Inertia)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.set_xticks(k_range)
    ax1.grid(True)

    sns.lineplot(x=k_range, y=silhouette_scores, marker='o', ax=ax2)
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis for Optimal k')
    ax2.set_xticks(k_range)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# 文本分类预测
def predict_text_cluster(text, vectorizer, kmeans):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    predicted_cluster = kmeans.predict(text_vector)
    # 计算文本向量到每个聚类中心的距离
    distances = euclidean_distances(text_vector, kmeans.cluster_centers_)
    # 将距离转换为概率（使用倒数和归一化）
    probabilities = 1 / (1 + distances)
    probabilities_normalized = probabilities / probabilities.sum()
    return predicted_cluster[0], probabilities_normalized.flatten()

def main():
    load_jieba()

    data_file = os.path.join(os.path.dirname(__file__), "Data.xlsx")
    df = load_and_preprocess_data(data_file)
    if df is None:
        return

    df = clean_data(df)  # 数据清洗

    excluded_keywords = ['一只', 'Cluster', '一个','我们','他们','你们','这个','那个','这些','那些','就是','什么','可以','没有','不是','现在','自己','大家','觉得','这样','很多','一些','因为','如果','然后','那么','所以','而且','但是','例如','比如','其实','只是','还是','已经','一直','非常','特别','真的','有点','有点儿','一下','一下儿','一次','一起','时候','地方','事情','问题','情况','方面','工作','生活','时间','东西','人员','进行','相关','不同','主要','重要','可能','应该','需要','必须','能够','可以','为了','通过','对于','关于','按照','根据','由于','随着','作为','以及','从而','不仅','而且','甚至','无论','除非','尽管','虽然','即使']
    n_top_keywords = 10

    # 向量化
    X = TfidfVectorizer().fit_transform(df['processed_content'])

    # 评估最佳k值
    evaluate_optimal_k(X, range(2, 11))

    optimal_k = int(input("根据肘部法则和轮廓系数图，输入你认为最佳的k值："))

    # 进行 KMeans 聚类
    df, cluster_keywords, vectorizer, kmeans, X = perform_kmeans_clustering(df, optimal_k, excluded_keywords, n_top_keywords)

    print(f"使用 k = {optimal_k} 的K均值聚类分析完成。\n")

    print("各聚类主题关键词：")
    for i in range(optimal_k):
        print(f"\n聚类 {i}:")
        print(f"权重前{n_top_keywords}的关键词: {', '.join(cluster_keywords[i])}")

    print("\n各聚类前5条文本示例：")
    for i in range(optimal_k):
        print(f"\nCluster {i}:")
        print(df[df['Cluster'] == i].head(5)[['微博正文', 'Cluster']])

    while True:
        user_text = input("\n请输入一段文本进行分类（输入 exit 退出）：")
        if user_text.lower() == 'exit':
            break

        cluster, probabilities = predict_text_cluster(user_text, vectorizer, kmeans)
        print(f"这段文本最可能属于的类是：{cluster}")
        for i, probability in enumerate(probabilities):
            print(f"属于聚类{i}的可能性：{probability:.2f}")

if __name__ == "__main__":
    main()