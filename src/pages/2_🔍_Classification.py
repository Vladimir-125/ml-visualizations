import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.data_loader import get_available_datasets, get_dataset_info, load_dataset

st.set_page_config(
    page_title='Classification Visualization',
    page_icon='🔍',
    layout='wide',
)

st.markdown('# 🔍 Classification Visualization')
st.sidebar.header('Classification')
st.write(
    """
    This page allows you to explore different classification algorithms and visualize
    their decision boundaries. You can select different datasets and algorithms to
    see how they perform.
    """
)

# Define classification datasets
classification_datasets = [
    dataset
    for dataset in get_available_datasets()
    if dataset in ('iris', 'wine', 'breast_cancer', 'classification', 'moons', 'circles')
]

# Dataset selection
dataset_info = get_dataset_info()
selected_dataset = st.sidebar.selectbox(
    'Select a dataset',
    options=classification_datasets,
    format_func=lambda x: f'{x.capitalize()} Dataset',
    index=0,
)

st.sidebar.markdown(f'**Dataset Info:** {dataset_info[selected_dataset]}')

# Define classification algorithms
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000),
}

# Algorithm selection
selected_algorithm = st.sidebar.selectbox('Select an algorithm', options=list(classifiers.keys()), index=0)

# Load the dataset
X, y, feature_names, target_names, description = load_dataset(selected_dataset)

# Dataset description
with st.expander('Dataset Description', expanded=False):
    st.markdown(f'**{selected_dataset.capitalize()} Dataset**')
    st.text(description)
    st.markdown(f'**Shape:** {X.shape[0]} samples, {X.shape[1]} features')
    if target_names is not None:
        st.markdown(f"**Classes:** {', '.join(target_names)}")

# Feature selection for visualization if more than 2 features
if X.shape[1] > 2:
    st.sidebar.markdown('### Feature Selection')
    st.sidebar.markdown('Select 2 features for visualization:')
    feature_1 = st.sidebar.selectbox('Feature 1', options=feature_names, index=0)
    feature_2 = st.sidebar.selectbox('Feature 2', options=feature_names, index=1)

    # Get indices of selected features
    feature_1_idx = feature_names.index(feature_1)
    feature_2_idx = feature_names.index(feature_2)

    # Filter X to only include selected features
    X_selected = X[:, [feature_1_idx, feature_2_idx]]
    selected_feature_names = [feature_1, feature_2]
else:
    X_selected = X
    selected_feature_names = feature_names

# Standardize the data
st.sidebar.markdown('### Data Preprocessing')
standardize = st.sidebar.checkbox('Standardize Features', value=True)

if standardize:
    scaler = StandardScaler()
    X_selected = scaler.fit_transform(X_selected)

# Train-test split
test_size = st.sidebar.slider('Test Size', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
random_state = st.sidebar.slider('Random State', min_value=0, max_value=100, value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=random_state)

# Model parameters (simplified for demonstration)
st.sidebar.markdown('### Model Parameters')

if selected_algorithm == 'Logistic Regression':
    C = st.sidebar.slider('Regularization (C)', min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    classifiers[selected_algorithm].set_params(C=C)

elif selected_algorithm == 'K Nearest Neighbors':
    n_neighbors = st.sidebar.slider('Number of Neighbors', min_value=1, max_value=20, value=5, step=1)
    classifiers[selected_algorithm].set_params(n_neighbors=n_neighbors)

elif selected_algorithm == 'Support Vector Machine':
    C = st.sidebar.slider('Regularization (C)', min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    kernel = st.sidebar.selectbox('Kernel', options=['linear', 'poly', 'rbf', 'sigmoid'], index=2)
    classifiers[selected_algorithm].set_params(C=C, kernel=kernel)

elif selected_algorithm == 'Decision Tree':
    max_depth = st.sidebar.slider('Maximum Depth', min_value=1, max_value=20, value=5, step=1)
    classifiers[selected_algorithm].set_params(max_depth=max_depth)

elif selected_algorithm == 'Random Forest':
    n_estimators = st.sidebar.slider('Number of Trees', min_value=10, max_value=200, value=100, step=10)
    max_depth = st.sidebar.slider('Maximum Depth', min_value=1, max_value=20, value=5, step=1)
    classifiers[selected_algorithm].set_params(n_estimators=n_estimators, max_depth=max_depth)

elif selected_algorithm == 'Neural Network':
    hidden_layer_sizes = st.sidebar.slider('Hidden Layer Size', min_value=5, max_value=100, value=50, step=5)
    classifiers[selected_algorithm].set_params(hidden_layer_sizes=(hidden_layer_sizes,))

# Train the model
classifier = classifiers[selected_algorithm]
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test) if hasattr(classifier, 'predict_proba') else None

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    target_names=target_names if target_names is not None else None,
)
conf_matrix = confusion_matrix(y_test, y_pred)

st.subheader('Model Performance')
col1, col2 = st.columns(2)

with col1:
    st.metric('Accuracy', f'{accuracy:.4f}')
    st.markdown('### Confusion Matrix')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    if target_names is not None:
        ax.set_xticklabels(target_names)
        ax.set_yticklabels(target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

with col2:
    st.markdown('### Classification Report')
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# Visualize decision boundary (only works for 2D data)
if X_selected.shape[1] == 2:
    st.subheader('Decision Boundary Visualization')

    # Create a meshgrid to visualize the decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the class for each point in the meshgrid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # Plot the training points
    scatter = ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        edgecolors='k',
        cmap='viridis',
        alpha=0.8,
        marker='o',
        s=100,
        label='Training',
    )

    # Plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        edgecolors='k',
        cmap='viridis',
        alpha=0.8,
        marker='^',
        s=100,
        label='Testing',
    )

    plt.colorbar(scatter, label='Class')
    plt.legend()
    if X.shape[1] > 2:
        plt.xlabel(selected_feature_names[0])
        plt.ylabel(selected_feature_names[1])
    else:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    plt.title(f'Decision Boundary - {selected_algorithm}')

    st.pyplot(fig)
else:
    st.warning(
        'Decision boundary visualization is only available for 2D data. Please select 2 features for visualization.'
    )
