# Unsupervised Text Classification

This project uses unsupervised learning to automatically group similar text entries from a large dataset (around 80,000 entries). It uses the elbow method and silhouette score to determine the optimal number of groups (clusters) and can then classify new text into these groups.

Key features:

*   Unsupervised learning (no labeled data needed).
*   Uses elbow method and silhouette score for optimal clustering.
*   Classifies new text.
*   Handles large datasets (around 80,000 entries) with memory management tips.

This project specifically focuses on processing a substantial amount of text data (approximately 80,000 entries) using unsupervised learning techniques. The optimal number of clusters is determined using both the elbow method and silhouette score. The trained model can then be used to classify new, unseen text.

Because processing 80,000 entries can be memory-intensive, the code is designed to be adaptable. You can adjust the amount of data used for classification within the files to manage memory usage. More details on how to use the code and manage memory are provided below.

![Figure_1](https://github.com/user-attachments/assets/a2334455-ee93-4ba1-912e-558f06cc6dd6)
