import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Add the src directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import get_available_datasets, get_dataset_info, load_dataset

st.set_page_config(
    page_title='Data Exploration',
    page_icon='📊',
    layout='wide',
)

st.markdown('# 📊 Data Exploration')
st.sidebar.header('Data Exploration')
st.write(
    """
    This page allows you to explore different datasets commonly used in
    machine learning and data science. You can visualize the data distribution,
    correlations, and basic statistics.
    """
)

# Dataset selection
dataset_info = get_dataset_info()
dataset_options = get_available_datasets()

selected_dataset = st.sidebar.selectbox(
    'Select a dataset',
    options=dataset_options,
    format_func=lambda x: f'{x.capitalize()} Dataset',
    index=0,
)

st.sidebar.markdown(f'**Dataset Info:** {dataset_info[selected_dataset]}')

# Load the dataset
X, y, feature_names, target_names, description = load_dataset(selected_dataset)

# Create a dataframe for easier manipulation
if target_names is not None:
    target_column = 'Target'
    df = pd.DataFrame(X, columns=feature_names)
    df[target_column] = [target_names[i] if i < len(target_names) else str(i) for i in y]
else:
    target_column = 'Target Value'
    df = pd.DataFrame(X, columns=feature_names)
    df[target_column] = y

# Dataset description
with st.expander('Dataset Description', expanded=True):
    st.markdown(f'**{selected_dataset.capitalize()} Dataset**')
    st.text(description)
    st.markdown(f'**Shape:** {X.shape[0]} samples, {X.shape[1]} features')
    if target_names is not None:
        st.markdown(f"**Classes:** {', '.join(target_names)}")

# Basic statistics
st.subheader('Basic Statistics')
col1, col2 = st.columns(2)

with col1:
    st.dataframe(df.describe())

with col2:
    if target_names is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[target_column].value_counts().plot(kind='bar', ax=ax)
        plt.title('Class Distribution')
        plt.ylabel('Count')
        plt.xlabel('Class')
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.hist(df[target_column], bins=20)
        plt.title('Target Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        st.pyplot(fig)

# Data preview
st.subheader('Data Preview')
st.dataframe(df.head(10))

# Feature distributions
st.subheader('Feature Distributions')
col1, col2 = st.columns(2)

with col1:
    if len(feature_names) > 1:
        feature_x = st.selectbox('Select feature for X axis', options=feature_names)
    else:
        feature_x = feature_names[0]

with col2:
    if len(feature_names) > 1:
        remaining_features = [f for f in feature_names if f != feature_x]
        feature_y = st.selectbox('Select feature for Y axis', options=remaining_features)
    else:
        feature_y = None

# Visualization
st.subheader('Visualization')

if target_names is not None:
    if feature_y is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            df[feature_x],
            df[feature_y],
            c=[list(target_names).index(t) if t in target_names else 0 for t in df[target_column]],
            alpha=0.6,
            cmap='viridis',
        )
        plt.colorbar(scatter, label='Class')
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(f'{feature_x} vs {feature_y} colored by class')
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=target_column, y=feature_x, data=df, ax=ax)
        plt.title(f'Distribution of {feature_x} by class')
        st.pyplot(fig)
else:
    if feature_y is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df[feature_x], df[feature_y], c=df[target_column], alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, label=target_column)
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(f'{feature_x} vs {feature_y} colored by {target_column}')
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.scatter(df[feature_x], df[target_column])
        plt.xlabel(feature_x)
        plt.ylabel(target_column)
        plt.title(f'{feature_x} vs {target_column}')
        st.pyplot(fig)

# Correlation heatmap
if len(feature_names) > 1:
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = df[feature_names].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Feature Correlation Heatmap')
    st.pyplot(fig)

# Pairplot for small datasets
if len(feature_names) > 1 and len(feature_names) <= 5 and X.shape[0] <= 1000:
    st.subheader('Pairplot')
    with st.spinner('Generating pairplot...'):
        fig = sns.pairplot(df, hue=target_column, diag_kind='kde', markers='o')
        fig.fig.suptitle('Pairplot of Features', y=1.02)
        st.pyplot(fig.fig)
