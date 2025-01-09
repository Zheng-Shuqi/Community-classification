import os
import jieba
import pandas as pd
import openpyxl
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns

# 加载 jieba 分词器
def load_jieba():
    print("正在初始化 jieba 分词器...")
    jieba.initialize()  # 显式初始化
    print("jieba 分词器初始化完成。")

# 数据预处理模块
def preprocess_text(text):
    if isinstance(text, str):
        return ' '.join(jieba.cut(text))
    return ''

# 读取Excel文件并显示进度条
def read_excel_with_progress(file_path, sheet_name):
    print("开始加载Excel数据...")
    try:
        wb = openpyxl.load_workbook(file_path, read_only=True)  # 以只读模式打开，提高效率
        sheet = wb[sheet_name]
        total_rows = sheet.max_row
        rows = sheet.values

        df = pd.DataFrame(tqdm(rows, total=total_rows, desc="读取Excel行", unit="行"), columns=[cell.value for cell in sheet[1]])

        print("Excel数据加载完成。")
        return df

    except FileNotFoundError:
        print(f"错误：数据文件 {file_path} 未找到。")
        return None
    except KeyError:
        print(f"错误：工作表 '{sheet_name}' 未找到。")
        return None
    except Exception as e:
        print(f"读取Excel文件时发生错误：{e}")
        return None

# 数据清洗模块
def clean_data(df):
    # 移除空行
    df.dropna(subset=['微博正文'], inplace=True)
    # 移除重复行
    df.drop_duplicates(subset=['微博正文'], inplace=True)
    # 其他清洗操作
    df['微博正文'] = df['微博正文'].str.replace(r'\s+', ' ', regex=True)  # 移除多余空格
    return df

def load_and_preprocess_data(data_file):
    df = read_excel_with_progress(data_file, sheet_name='Sheet1')
    if df is None:
        return None
    df['微博正文'] = df['微博正文'].fillna('')
    df['processed_content'] = df['微博正文'].apply(preprocess_text)
    return df.head(60000)  # 只选择前60000条记录

def _get_top_keywords_for_clusters(X, vectorizer, kmeans, n_terms, excluded_keywords):
    labels = kmeans.labels_
    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    for i in range(kmeans.n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        sorted_indices = cluster_center.argsort()[::-1]
        top_keywords = [feature_names[idx] for idx in sorted_indices if feature_names[idx] not in excluded_keywords][:n_terms]
        cluster_keywords[i] = top_keywords
    return cluster_keywords

def evaluate_optimal_k(X, k_range):
    distortions = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        else:
            silhouette_scores.append(0)  # k=1时添加占位符，保持长度一致

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # 绘制肘部法则图
    sns.lineplot(x=k_range, y=distortions, marker='o', ax=ax1)
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Distortion (Inertia)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.set_xticks(k_range)
    ax1.grid(True)

    # 绘制轮廓系数图
    sns.lineplot(x=k_range, y=silhouette_scores, marker='o', ax=ax2)  # 移除 k_range[1:] 切片，使用完整的k_range
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis for Optimal k')
    ax2.set_xticks(k_range)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def perform_kmeans_clustering(df, optimal_k, excluded_keywords, n_top_keywords):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_content'])
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    kmeans.fit(X)
    df['Cluster'] = kmeans.labels_
    cluster_keywords = _get_top_keywords_for_clusters(X, vectorizer, kmeans, n_top_keywords, excluded_keywords)
    return df, cluster_keywords, vectorizer, kmeans, X

def predict_text_cluster(text, vectorizer, kmeans):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    predicted_cluster = kmeans.predict(text_vector)
    distances = euclidean_distances(text_vector, kmeans.cluster_centers_)
    probabilities = 1 / (1 + distances)
    probabilities_normalized = probabilities / probabilities.sum()
    return predicted_cluster[0], probabilities_normalized.flatten()

def main():
    load_jieba()  # 在程序开始时加载 jieba

    # 使用相对路径加载数据
    data_file = os.path.join(os.path.dirname(__file__), "Data.xlsx")
    df = load_and_preprocess_data(data_file)
    if df is None:  # 处理读取文件失败的情况
        return

    excluded_keywords = ['一只', 'Cluster', '一个','我们','他们','你们','这个','那个','这些','那些','就是','什么','可以','没有','不是','现在','自己','大家','觉得','这样','很多','一些','因为','如果','然后','那么','所以','而且','但是','例如','比如','其实','只是','还是','已经','一直','非常','特别','真的','有点','有点儿','一下','一下儿','一次','一起','时候','地方','事情','问题','情况','方面','工作','生活','时间','东西','人员','进行','相关','不同','主要','重要','可能','应该','需要','必须','能够','可以','为了','通过','对于','关于','按照','根据','由于','随着','作为','以及','从而','不仅','而且','甚至','无论','除非','尽管','虽然','即使']  # 扩充停用词表
    n_top_keywords = 10

    X = TfidfVectorizer().fit_transform(df['processed_content'])

    evaluate_optimal_k(X, range(2, 11))
    optimal_k = int(input("根据肘部法则和轮廓系数图，输入你认为最佳的k值："))

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