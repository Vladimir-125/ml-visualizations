"""Utility functions for loading datasets."""
import numpy as np
import streamlit as st
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
    make_regression,
)


@st.cache_data
def load_dataset(dataset_name, random_state=42):
    """Load a dataset by name.

    Parameters:
        dataset_name (str): Name of the dataset to load
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X, y, feature_names, target_names, description)
    """
    if dataset_name == 'iris':
        data = load_iris()
        X, y = data.data, data.target
        return X, y, data.feature_names, data.target_names, data.DESCR

    elif dataset_name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        return X, y, data.feature_names, data.target_names, data.DESCR

    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        return X, y, data.feature_names, data.target_names, data.DESCR

    elif dataset_name == 'diabetes':
        data = load_diabetes()
        X, y = data.data, data.target
        return X, y, data.feature_names, None, data.DESCR

    elif dataset_name == 'classification':
        X, y = make_classification(
            n_samples=100,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            random_state=random_state,
        )
        feature_names = [f'Feature {i + 1}' for i in range(X.shape[1])]
        target_names = [f'Class {i}' for i in range(len(np.unique(y)))]
        description = 'Synthetic classification dataset with 2 features and 2 classes.'
        return X, y, feature_names, target_names, description

    elif dataset_name == 'regression':
        X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=random_state)
        feature_names = [f'Feature {i + 1}' for i in range(X.shape[1])]
        description = 'Synthetic regression dataset with 1 feature.'
        return X, y, feature_names, None, description

    elif dataset_name == 'blobs':
        X, y = make_blobs(n_samples=300, centers=4, n_features=2, random_state=random_state)
        feature_names = [f'Feature {i + 1}' for i in range(X.shape[1])]
        target_names = [f'Cluster {i}' for i in range(len(np.unique(y)))]
        description = 'Synthetic clustering dataset with 2 features and 4 clusters.'
        return X, y, feature_names, target_names, description

    elif dataset_name == 'moons':
        X, y = make_moons(n_samples=200, noise=0.1, random_state=random_state)
        feature_names = [f'Feature {i + 1}' for i in range(X.shape[1])]
        target_names = [f'Class {i}' for i in range(len(np.unique(y)))]
        description = 'Two interleaving half circles.'
        return X, y, feature_names, target_names, description

    elif dataset_name == 'circles':
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=random_state)
        feature_names = [f'Feature {i + 1}' for i in range(X.shape[1])]
        target_names = [f'Class {i}' for i in range(len(np.unique(y)))]
        description = 'A large circle containing a smaller circle in 2d.'
        return X, y, feature_names, target_names, description

    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')


@st.cache_data
def get_available_datasets():
    """Return a list of available datasets."""
    return [
        'iris',
        'wine',
        'breast_cancer',
        'diabetes',
        'classification',
        'regression',
        'blobs',
        'moons',
        'circles',
    ]


@st.cache_data
def get_dataset_info():
    """Return information about available datasets."""
    return {
        'iris': 'Iris flower dataset (classification, 4 features, 3 classes)',
        'wine': 'Wine recognition dataset (classification, 13 features, 3 classes)',
        'breast_cancer': 'Breast cancer dataset (classification, 30 features, 2 classes)',
        'diabetes': 'Diabetes dataset (regression, 10 features)',
        'classification': 'Synthetic classification dataset (2 features, 2 classes)',
        'regression': 'Synthetic regression dataset (1 feature)',
        'blobs': 'Synthetic clustering dataset (2 features, 4 clusters)',
        'moons': 'Two interleaving half circles (2 features, 2 classes)',
        'circles': 'Concentric circles (2 features, 2 classes)',
    }
