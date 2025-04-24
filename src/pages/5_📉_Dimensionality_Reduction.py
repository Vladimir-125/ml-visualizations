import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import umap.umap_ as umap
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler

from utils.data_loader import get_available_datasets, get_dataset_info, load_dataset

st.set_page_config(
    page_title='Dimensionality Reduction',
    page_icon='📉',
    layout='wide',
)

st.markdown('# 📉 Dimensionality Reduction')
st.sidebar.header('Dimensionality Reduction')
st.write(
    """
    This page allows you to explore different dimensionality reduction techniques and visualize
    high-dimensional data in 2D and 3D. You can select different datasets and algorithms to see
    how they project the data into lower dimensions while preserving important structures.
    """
)

# Define datasets
datasets = [
    dataset
    for dataset in get_available_datasets()
    if dataset in ('iris', 'wine', 'breast_cancer', 'blobs', 'moons', 'circles')
]

# Dataset selection
dataset_info = get_dataset_info()
selected_dataset = st.sidebar.selectbox(
    'Select a dataset',
    options=datasets,
    format_func=lambda x: f'{x.capitalize()} Dataset',
    index=0,
)

st.sidebar.markdown(f'**Dataset Info:** {dataset_info[selected_dataset]}')

# Define dimensionality reduction techniques
dim_reduction_algos = {
    'PCA': {
        'model': PCA,
        'params': {
            'n_components': {'min': 2, 'max': 3, 'default': 2},
            'random_state': {'default': 42},
        },
        'description': (
            'Principal Component Analysis: Linear dimensionality reduction using ' 'Singular Value Decomposition.'
        ),
    },
    't-SNE': {
        'model': TSNE,
        'params': {
            'n_components': {'min': 2, 'max': 3, 'default': 2},
            'perplexity': {'min': 5, 'max': 50, 'default': 30, 'step': 5},
            'learning_rate': {'min': 10, 'max': 1000, 'default': 200, 'step': 10},
            'random_state': {'default': 42},
        },
        'description': (
            't-Distributed Stochastic Neighbor Embedding: Nonlinear dimensionality reduction '
            'that preserves local neighborhoods.'
        ),
    },
    'UMAP': {
        'model': umap.UMAP,
        'params': {
            'n_components': {'min': 2, 'max': 3, 'default': 2},
            'n_neighbors': {'min': 2, 'max': 100, 'default': 15, 'step': 1},
            'min_dist': {'min': 0.0, 'max': 0.99, 'default': 0.1, 'step': 0.05},
            'random_state': {'default': 42},
        },
        'description': (
            'Uniform Manifold Approximation and Projection: Manifold learning technique ' 'for dimension reduction.'
        ),
    },
    'LDA': {
        'model': LinearDiscriminantAnalysis,
        'params': {'n_components': {'min': 2, 'max': 3, 'default': 2}},
        'description': (
            'Linear Discriminant Analysis: Finds linear combinations of features that best ' 'separate classes.'
        ),
    },
    'Isomap': {
        'model': Isomap,
        'params': {
            'n_components': {'min': 2, 'max': 3, 'default': 2},
            'n_neighbors': {'min': 5, 'max': 50, 'default': 5, 'step': 1},
        },
        'description': 'Isometric Mapping: Nonlinear dimensionality reduction through geodesic distances.',
    },
    'MDS': {
        'model': MDS,
        'params': {
            'n_components': {'min': 2, 'max': 3, 'default': 2},
            'random_state': {'default': 42},
        },
        'description': 'Multidimensional Scaling: Projects data to lower dimensions while preserving distances.',
    },
    'LLE': {
        'model': LocallyLinearEmbedding,
        'params': {
            'n_components': {'min': 2, 'max': 3, 'default': 2},
            'n_neighbors': {'min': 5, 'max': 50, 'default': 5, 'step': 1},
            'random_state': {'default': 42},
        },
        'description': (
            'Locally Linear Embedding: Nonlinear dimensionality reduction by preserving local '
            'neighborhood structures.'
        ),
    },
    'Kernel PCA': {
        'model': KernelPCA,
        'params': {
            'n_components': {'min': 2, 'max': 3, 'default': 2},
            'kernel': {
                'options': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'],
                'default': 'rbf',
            },
            'random_state': {'default': 42},
        },
        'description': 'Kernel Principal Component Analysis: Nonlinear dimensionality reduction using kernels.',
    },
    'Truncated SVD': {
        'model': TruncatedSVD,
        'params': {
            'n_components': {'min': 2, 'max': 3, 'default': 2},
            'random_state': {'default': 42},
        },
        'description': (
            'Truncated Singular Value Decomposition: Dimensionality reduction using SVD, ' 'works on sparse matrices.'
        ),
    },
}

# Algorithm selection
selected_algorithm = st.sidebar.selectbox(
    'Select a dimensionality reduction technique',
    options=list(dim_reduction_algos.keys()),
    index=0,
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

# Algorithm description
with st.expander('Algorithm Description', expanded=False):
    st.markdown(f'**{selected_algorithm}**')
    st.markdown(dim_reduction_algos[selected_algorithm]['description'])

# Check if LDA is selected but dataset has only one class
if selected_algorithm == 'LDA' and (y is None or len(np.unique(y)) < 2):
    st.error('LDA requires a dataset with at least two classes for dimensionality reduction.')
    st.stop()

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

for param_name, param_info in dim_reduction_algos[selected_algorithm]['params'].items():
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

# Choose visualization dimension
n_components = algo_params.get('n_components', 2)
viz_dimension = st.sidebar.radio('Visualization Dimension', options=['2D', '3D'], index=0 if n_components == 2 else 1)

# Update n_components based on viz_dimension
if viz_dimension == '2D' and n_components != 2:
    algo_params['n_components'] = 2
elif viz_dimension == '3D' and n_components != 3:
    algo_params['n_components'] = 3

# Display loading spinner for computationally intensive algorithms
with st.spinner(f'Applying {selected_algorithm} to the data...'):
    # Apply dimensionality reduction
    model_class = dim_reduction_algos[selected_algorithm]['model']

    # Special case for LDA which requires labels
    if selected_algorithm == 'LDA':
        model = model_class(n_components=algo_params['n_components'])
        X_reduced = model.fit_transform(X_scaled, y)
    else:
        model = model_class(**algo_params)
        X_reduced = model.fit_transform(X_scaled)

# Create a dataframe for visualization
if viz_dimension == '2D':
    df_viz = pd.DataFrame(X_reduced[:, :2], columns=['Component 1', 'Component 2'])
else:
    df_viz = pd.DataFrame(X_reduced[:, :3], columns=['Component 1', 'Component 2', 'Component 3'])

# Add target information if available
if y is not None:
    if target_names is not None:
        df_viz['Class'] = [target_names[i] if i < len(target_names) else f'Class {i}' for i in y]
    else:
        df_viz['Class'] = [f'Class {i}' for i in y]

# Explained variance ratio (if available)
if hasattr(model, 'explained_variance_ratio_'):
    explained_variance = model.explained_variance_ratio_
    st.subheader('Explained Variance Ratio')

    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative_variance = np.cumsum(explained_variance)

    ax.bar(
        range(1, len(explained_variance) + 1),
        explained_variance,
        alpha=0.7,
        label='Individual',
    )
    ax.step(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        where='mid',
        label='Cumulative',
    )
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Explained Variance by Components')
    ax.legend()

    st.pyplot(fig)

    if len(explained_variance) >= 2:
        st.markdown(f'**Component 1 explains {explained_variance[0]:.2%} of the variance.**')
        st.markdown(f'**Component 2 explains {explained_variance[1]:.2%} of the variance.**')
        if len(explained_variance) >= 3 and viz_dimension == '3D':
            st.markdown(f'**Component 3 explains {explained_variance[2]:.2%} of the variance.**')

# Visualize the reduced data
st.subheader(f'{selected_algorithm} Visualization ({viz_dimension})')

if viz_dimension == '2D':
    if y is not None and 'Class' in df_viz:
        # Create a categorical plot with classes
        fig = px.scatter(
            df_viz,
            x='Component 1',
            y='Component 2',
            color='Class',
            hover_name='Class',
            title=f'{selected_algorithm} - 2D Projection',
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.G10,
        )
    else:
        # Create a simple scatter plot
        fig = px.scatter(
            df_viz,
            x='Component 1',
            y='Component 2',
            title=f'{selected_algorithm} - 2D Projection',
            template='plotly_white',
        )
else:  # 3D visualization
    if y is not None and 'Class' in df_viz:
        # Create a 3D categorical plot with classes
        fig = px.scatter_3d(
            df_viz,
            x='Component 1',
            y='Component 2',
            z='Component 3',
            color='Class',
            hover_name='Class',
            title=f'{selected_algorithm} - 3D Projection',
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.G10,
        )
    else:
        # Create a simple 3D scatter plot
        fig = px.scatter_3d(
            df_viz,
            x='Component 1',
            y='Component 2',
            z='Component 3',
            title=f'{selected_algorithm} - 3D Projection',
            template='plotly_white',
        )

    # Configure 3D aspects
    fig.update_layout(
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3',
            aspectmode='cube',
        )
    )

# Update layout for better visualization
fig.update_traces(
    marker=dict(size=8, line=dict(width=1, color='white')),
    selector=dict(mode='markers'),
)
fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

# Show the plot
st.plotly_chart(fig, use_container_width=True)

# Feature importance for PCA (display loadings)
if selected_algorithm == 'PCA' and hasattr(model, 'components_'):
    st.subheader('Feature Importance (Loadings)')

    loadings = model.components_
    n_components = loadings.shape[0]

    # Create a DataFrame for the loadings
    loadings_df = pd.DataFrame(
        loadings.T,
        columns=[f'PC{i + 1}' for i in range(n_components)],
        index=feature_names,
    )

    # Display loadings as a table
    st.dataframe(loadings_df)

    # Visualize loadings
    fig, ax = plt.subplots(figsize=(12, 10))

    # For better visualization, we'll plot loadings for the first two components
    sns.heatmap(
        loadings_df.iloc[:, :2],
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        ax=ax,
    )

    plt.title('PCA Loadings (Feature Importance)')
    plt.tight_layout()
    st.pyplot(fig)

    # Biplots are 2D visualizations of features and observations
    if n_components >= 2:
        st.subheader('PCA Biplot')

        # Create a biplot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot the observations
        observations = X_reduced[:, :2]
        ax.scatter(
            observations[:, 0],
            observations[:, 1],
            c=y if y is not None else None,
            cmap='viridis',
            alpha=0.6,
            s=50,
        )

        # Plot feature vectors
        for i, feature in enumerate(feature_names):
            ax.arrow(
                0,
                0,
                loadings[0, i] * 5,
                loadings[1, i] * 5,
                head_width=0.2,
                head_length=0.2,
                fc='red',
                ec='red',
            )
            ax.text(
                loadings[0, i] * 5.2,
                loadings[1, i] * 5.2,
                feature,
                color='red',
                ha='center',
                va='center',
            )

        # Add circle
        circle = plt.Circle((0, 0), 1, color='r', fill=False)
        ax.add_patch(circle)

        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(alpha=0.3)

        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
        plt.title('PCA Biplot')

        # Equal scaling
        ax.set_aspect('equal')
        plt.tight_layout()

        st.pyplot(fig)
