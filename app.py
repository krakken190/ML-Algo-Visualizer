import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, silhouette_score
)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN

# Setup
st.set_page_config("ML Visual Lab", layout="wide")
st.title("🧠 ML Learning Lab")

# === Sidebar Controls ===
st.sidebar.header("🔍 Explore ML Categories")
learning_type = st.sidebar.radio("Learning Type", ["Supervised", "Unsupervised"])

# === Algorithm Selector Based on Learning Type ===
if learning_type == "Supervised":
    algorithm = st.sidebar.selectbox("Choose Supervised Algorithm", ["Logistic Regression", "SVM", "Decision Tree"])
elif learning_type == "Unsupervised":
    algorithm = st.sidebar.selectbox("Choose Unsupervised Algorithm", ["K-Means", "DBSCAN"])
# else:
#     st.sidebar.markdown("ℹ️ Reinforcement Learning needs an environment and can't be visualized here.")

# === Common Controls ===
noise = st.sidebar.slider("Data Noise", 0.0, 0.5, 0.3, 0.01)

# === Generate Dataset ===
if learning_type == "Supervised":
    X, y = make_moons(n_samples=300, noise=noise, random_state=42)
elif learning_type == "Unsupervised":
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.2 + noise, random_state=42)

# === Supervised Section ===
if learning_type == "Supervised":

    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)

    if algorithm == "Logistic Regression":
        C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        model = LogisticRegression(C=C)

    elif algorithm == "SVM":
        kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf"])
        C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        if kernel == "rbf":
            gamma = st.sidebar.slider("Gamma", 0.001, 1.0, 0.1, 0.001)
            model = SVC(kernel=kernel, C=C, gamma=gamma)
        else:
            model = SVC(kernel=kernel, C=C)

    elif algorithm == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 15, 3)
        model = DecisionTreeClassifier(max_depth=max_depth)

    # Split & Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # === Metrics ===
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)

# === Unsupervised Section ===
elif learning_type == "Unsupervised":

    if algorithm == "K-Means":
        k = st.sidebar.slider("Number of Clusters (k)", 2, 6, 3)
        model = KMeans(n_clusters=k, random_state=42)
    elif algorithm == "DBSCAN":
        eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("Min Samples", 3, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit & Predict
    model.fit(X)
    y_pred_test = model.labels_

    # Silhouette score
    valid_labels = len(set(y_pred_test)) > 1 and -1 not in set(y_pred_test)
    silhouette = silhouette_score(X, y_pred_test) if valid_labels else None

# === Visualization Functions ===
def plot_boundary(ax, model, X, y, title="Boundary"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    try:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
    except:
        pass
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k', s=40)
    ax.set_title(title)
    ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right")

def plot_misclassified(ax, X, y_true, y_pred, title="Misclassified"):
    misclassified = y_true != y_pred
    ax.scatter(X[~misclassified][:, 0], X[~misclassified][:, 1], c='green', label='Correct', edgecolor='k')
    ax.scatter(X[misclassified][:, 0], X[misclassified][:, 1], c='red', label='Wrong', edgecolor='k')
    ax.set_title(title)
    ax.legend()

# === PLOTS ===
st.subheader("📊 Visual Comparison")
col1, col2, col3 = st.columns(3)

if learning_type == "Supervised":
    with col1:
        fig1, ax1 = plt.subplots()
        plot_boundary(ax1, model, X_test, y_test, "Test Set Boundary")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        plot_boundary(ax2, model, X_train, y_train, "Train Set Boundary")
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots()
        plot_misclassified(ax3, X_test, y_test, y_pred_test, "Misclassified Points")
        st.pyplot(fig3)

elif learning_type == "Unsupervised":
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.scatter(X[:, 0], X[:, 1], c=y_pred_test, cmap="rainbow", edgecolor="k")
        ax1.set_title("Cluster Assignments")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
        ax2.set_title("True Labels (if known)")
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots()
        if silhouette:
            ax3.bar(["Silhouette Score"], [silhouette])
            ax3.set_ylim(0, 1)
            st.pyplot(fig3)
        else:
            ax3.text(0.1, 0.5, "Silhouette Score\nUnavailable", fontsize=12)
            ax3.axis("off")
            st.pyplot(fig3)

# === Metrics Section ===
if learning_type == "Supervised":
    st.markdown("---")
    tab1, tab2 = st.tabs(["📊 Confusion Matrix", "📈 Metrics"])

    with tab1:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1], ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    with tab2:
        st.subheader("Evaluation Metrics")
        st.markdown(f"""
        - **Accuracy**: `{acc:.2f}`
        - **Precision**: `{prec:.2f}`
        - **Recall**: `{recall:.2f}`
        - **F1 Score**: `{f1:.2f}`
        """)
        with st.expander("📘 What These Metrics Mean"):
            st.markdown("""
            - **Accuracy**: Proportion of total correct predictions.
            - **Precision**: Out of predicted positives, how many are correct.
            - **Recall**: Out of actual positives, how many were detected.
            - **F1 Score**: Harmonic mean of precision and recall. Useful when classes are imbalanced.
            """)

elif learning_type == "Unsupervised":
    st.markdown("---")
    st.subheader("🧪 Evaluation for Clustering")
    if silhouette:
        st.markdown(f"**Silhouette Score**: `{silhouette:.2f}`")
    else:
        st.warning("Silhouette score could not be computed (possibly due to only 1 cluster or noise points).")
