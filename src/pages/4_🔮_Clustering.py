import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from utils.data_loader import get_available_datasets, get_dataset_info, load_dataset

st.set_page_config(
    page_title='Clustering Visualization',
    page_icon='🔮',
    layout='wide',
)

st.markdown('# 🔮 Clustering Visualization')
st.sidebar.header('Clustering')
st.write(
    """
    This page allows you to explore different clustering algorithms and visualize
    how they group data points. You can select different datasets and algorithms to
    see how they perform.
    """
)

# Define clustering datasets
clustering_datasets = [
    dataset
    for dataset in get_available_datasets()
    if dataset in ('iris', 'wine', 'breast_cancer', 'blobs', 'moons', 'circles')
]

# Dataset selection
dataset_info = get_dataset_info()
selected_dataset = st.sidebar.selectbox(
    'Select a dataset',
    options=clustering_datasets,
    format_func=lambda x: f'{x.capitalize()} Dataset',
    index=0,
)

st.sidebar.markdown(f'**Dataset Info:** {dataset_info[selected_dataset]}')

# Define clustering algorithms
clustering_algos = {
    'K-Means': {
        'model': KMeans,
        'params': {
            'n_clusters': {'min': 2, 'max': 10, 'default': 3},
            'random_state': {'default': 42},
        },
    },
    'DBSCAN': {
        'model': DBSCAN,
        'params': {
            'eps': {'min': 0.1, 'max': 2.0, 'default': 0.5, 'step': 0.1},
            'min_samples': {'min': 2, 'max': 20, 'default': 5},
        },
    },
    'Hierarchical Clustering': {
        'model': AgglomerativeClustering,
        'params': {
            'n_clusters': {'min': 2, 'max': 10, 'default': 3},
            'linkage': {
                'options': ['ward', 'complete', 'average', 'single'],
                'default': 'ward',
            },
        },
    },
    'Gaussian Mixture': {
        'model': GaussianMixture,
        'params': {
            'n_components': {'min': 2, 'max': 10, 'default': 3},
            'covariance_type': {
                'options': ['full', 'tied', 'diag', 'spherical'],
                'default': 'full',
            },
            'random_state': {'default': 42},
        },
    },
    'Spectral Clustering': {
        'model': SpectralClustering,
        'params': {
            'n_clusters': {'min': 2, 'max': 10, 'default': 3},
            'random_state': {'default': 42},
        },
    },
}

# Algorithm selection
selected_algorithm = st.sidebar.selectbox(
    'Select a clustering algorithm', options=list(clustering_algos.keys()), index=0
)

# Load the dataset
X, y, feature_names, target_names, description = load_dataset(selected_dataset)

# Dataset description
with st.expander('Dataset Description', expanded=False):
    st.markdown(f'**{selected_dataset.capitalize()} Dataset**')
    st.text(description)
    st.markdown(f'**Shape:** {X.shape[0]} samples, {X.shape[1]} features')
    if target_names is not None:
        st.markdown(f"**Classes:** {', '.join(target_names)}")

# Feature selection for visualization
if X.shape[1] > 2:
    st.sidebar.markdown('### Feature Selection')
    st.sidebar.markdown('For visualization, you can either select 2 features or use dimensionality reduction:')

    viz_method = st.sidebar.radio('Visualization Method', options=['Select Features', 'PCA', 't-SNE'], index=0)

    if viz_method == 'Select Features':
        feature_1 = st.sidebar.selectbox('Feature 1', options=feature_names, index=0)
        feature_2 = st.sidebar.selectbox('Feature 2', options=feature_names, index=1)

        # Get indices of selected features
        feature_1_idx = feature_names.index(feature_1)
        feature_2_idx = feature_names.index(feature_2)

        # Filter X to only include selected features for visualization
        X_viz = X[:, [feature_1_idx, feature_2_idx]]
        viz_feature_names = [feature_1, feature_2]

    elif viz_method == 'PCA':
        # Apply PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        X_viz = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        viz_feature_names = [
            f'PCA1 ({explained_variance[0]:.2%} variance)',
            f'PCA2 ({explained_variance[1]:.2%} variance)',
        ]

    else:  # t-SNE
        # For t-SNE, we'll compute it when needed to avoid slowing down the UI
        from sklearn.manifold import TSNE

        perplexity = min(30, X.shape[0] - 1)  # t-SNE parameter
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_viz = tsne.fit_transform(X)
        viz_feature_names = ['t-SNE1', 't-SNE2']
else:
    X_viz = X
    viz_feature_names = feature_names

# Standardize the data
st.sidebar.markdown('### Data Preprocessing')
standardize = st.sidebar.checkbox('Standardize Features', value=True)

if standardize:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X

# Algorithm parameters
st.sidebar.markdown('### Algorithm Parameters')
algo_params = {}

for param_name, param_info in clustering_algos[selected_algorithm]['params'].items():
    if 'min' in param_info and 'max' in param_info:
        # Numeric parameter with slider
        step = param_info.get('step', 1)
        if isinstance(step, int):
            algo_params[param_name] = st.sidebar.slider(
                param_name,
                min_value=param_info['min'],
                max_value=param_info['max'],
                value=param_info['default'],
                step=step,
            )
        else:
            algo_params[param_name] = st.sidebar.slider(
                param_name,
                min_value=float(param_info['min']),
                max_value=float(param_info['max']),
                value=float(param_info['default']),
                step=float(param_info.get('step', 0.1)),
            )
    elif 'options' in param_info:
        # Categorical parameter with selectbox
        options = param_info['options']
        default_idx = options.index(param_info['default']) if 'default' in param_info else 0
        algo_params[param_name] = st.sidebar.selectbox(param_name, options=options, index=default_idx)
    else:
        # Fixed parameter
        algo_params[param_name] = param_info['default']

# Train the clustering model
model_class = clustering_algos[selected_algorithm]['model']
model = model_class(**algo_params)

# Fit the model
if selected_algorithm == 'Gaussian Mixture':
    model.fit(X_scaled)
    labels = model.predict(X_scaled)
else:
    labels = model.fit_predict(X_scaled)

# Create a DataFrame for visualization
df_viz = pd.DataFrame(X_viz, columns=viz_feature_names)
df_viz['Cluster'] = labels

# Count the number of clusters (excluding noise points labeled as -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Calculate clustering metrics (if possible)
metrics = {}
if n_clusters > 1 and n_clusters < len(X_scaled) - 1:
    try:
        metrics['Silhouette Score'] = silhouette_score(X_scaled, labels)
    except Exception:
        metrics['Silhouette Score'] = 'N/A'

    try:
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(X_scaled, labels)
    except Exception:
        metrics['Calinski-Harabasz Index'] = 'N/A'

    try:
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(X_scaled, labels)
    except Exception:
        metrics['Davies-Bouldin Index'] = 'N/A'

# Display clustering statistics
st.subheader('Clustering Results')
col1, col2, col3 = st.columns(3)

with col1:
    st.metric('Number of Clusters', n_clusters)

with col2:
    if 'Silhouette Score' in metrics:
        if isinstance(metrics['Silhouette Score'], str):
            st.metric('Silhouette Score', metrics['Silhouette Score'])
        else:
            st.metric('Silhouette Score', f"{metrics['Silhouette Score']:.4f}")

with col3:
    if 'Davies-Bouldin Index' in metrics:
        if isinstance(metrics['Davies-Bouldin Index'], str):
            st.metric('Davies-Bouldin Index', metrics['Davies-Bouldin Index'])
        else:
            st.metric('Davies-Bouldin Index', f"{metrics['Davies-Bouldin Index']:.4f}")

# Show cluster sizes
st.subheader('Cluster Sizes')
cluster_sizes = pd.Series(labels).value_counts().sort_index()
cluster_size_df = pd.DataFrame({'Cluster': cluster_sizes.index, 'Size': cluster_sizes.values})

fig = px.bar(
    cluster_size_df,
    x='Cluster',
    y='Size',
    color='Cluster',
    labels={'Cluster': 'Cluster Label', 'Size': 'Number of Points'},
    title='Number of Points per Cluster',
)
st.plotly_chart(fig, use_container_width=True)

# Visualize the clusters (2D plot)
st.subheader('Cluster Visualization')

viz_type = st.radio(
    'Visualization Type',
    options=['Scatter Plot', 'Interactive Plot'],
    index=1,
    horizontal=True,
)

if viz_type == 'Scatter Plot':
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df_viz[viz_feature_names[0]],
        df_viz[viz_feature_names[1]],
        c=df_viz['Cluster'],
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='w',
    )

    # Add a legend
    legend1 = ax.legend(*scatter.legend_elements(), title='Clusters')
    ax.add_artist(legend1)

    # If we have true labels, show them as well
    if y is not None and len(np.unique(y)) > 1:
        ax2 = ax.twinx()
        ax2.set_yticklabels([])
        scatter2 = ax2.scatter([], [], c=[], cmap='Paired', alpha=0)  # Dummy to not affect the plot

        # Add a legend for true classes
        if target_names is not None:
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    markerfacecolor=f'C{i}',
                    markersize=10,
                    alpha=0.6,
                    label=name,
                )
                for i, name in enumerate(target_names)
            ]
            legend2 = ax2.legend(handles=legend_elements, title='True Classes', loc='upper left')
            ax2.add_artist(legend2)

    ax.set_xlabel(viz_feature_names[0])
    ax.set_ylabel(viz_feature_names[1])
    ax.set_title(f'{selected_algorithm} Clustering')
    st.pyplot(fig)
else:
    # True labels (if available)
    if y is not None:
        df_viz['True Label'] = y
        if target_names is not None:
            df_viz['True Class'] = [target_names[i] if i < len(target_names) else f'Class {i}' for i in y]
            color_discrete_map = {
                name: f'rgb({50 + i * 30}, {100 + i * 20}, {150 + i * 15})' for i, name in enumerate(target_names)
            }
        else:
            df_viz['True Class'] = [f'Class {i}' for i in y]
            color_discrete_map = None

        fig = px.scatter(
            df_viz,
            x=viz_feature_names[0],
            y=viz_feature_names[1],
            color='True Class' if 'True Class' in df_viz else 'True Label',
            symbol='Cluster',
            hover_data=['Cluster'],
            color_discrete_map=color_discrete_map,
            title=f'{selected_algorithm} Clustering with True Labels',
            labels={
                viz_feature_names[0]: viz_feature_names[0],
                viz_feature_names[1]: viz_feature_names[1],
                'Cluster': 'Cluster',
            },
        )
    else:
        fig = px.scatter(
            df_viz,
            x=viz_feature_names[0],
            y=viz_feature_names[1],
            color='Cluster',
            hover_data=['Cluster'],
            title=f'{selected_algorithm} Clustering',
            labels={
                viz_feature_names[0]: viz_feature_names[0],
                viz_feature_names[1]: viz_feature_names[1],
                'Cluster': 'Cluster',
            },
        )

    # Update layout for better visualization
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

    st.plotly_chart(fig, use_container_width=True)

# Evaluate against true labels (if available)
if y is not None:
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    st.subheader('Evaluation Against True Labels')

    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    ami = adjusted_mutual_info_score(y, labels)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Adjusted Rand Index', f'{ari:.4f}')
        st.markdown('Measures similarity between true and predicted clusters. 1.0 is perfect, 0.0 is random.')

    with col2:
        st.metric('Normalized Mutual Info', f'{nmi:.4f}')
        st.markdown('Measures mutual information between true and predicted clusters, normalized to [0, 1].')

    with col3:
        st.metric('Adjusted Mutual Info', f'{ami:.4f}')
        st.markdown('Like NMI but adjusted for chance. 1.0 is perfect, 0.0 is random.')
