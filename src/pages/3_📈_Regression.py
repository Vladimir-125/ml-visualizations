import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from utils.data_loader import get_available_datasets, get_dataset_info, load_dataset

st.set_page_config(
    page_title='Regression Visualization',
    page_icon='📈',
    layout='wide',
)

st.markdown('# 📈 Regression Visualization')
st.sidebar.header('Regression')
st.write(
    """
    This page allows you to explore different regression algorithms and visualize
    their predictions. You can select different datasets and algorithms to see how
    they perform.
    """
)

# Define regression datasets
regression_datasets = [dataset for dataset in get_available_datasets() if dataset in ('diabetes', 'regression')]

# Dataset selection
dataset_info = get_dataset_info()
selected_dataset = st.sidebar.selectbox(
    'Select a dataset',
    options=regression_datasets,
    format_func=lambda x: f'{x.capitalize()} Dataset',
    index=0,
)

st.sidebar.markdown(f'**Dataset Info:** {dataset_info[selected_dataset]}')

# Define regression algorithms
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Elastic Net': ElasticNet(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector Regression': SVR(),
    'K Nearest Neighbors': KNeighborsRegressor(),
}

# Algorithm selection
selected_algorithm = st.sidebar.selectbox('Select an algorithm', options=list(regressors.keys()), index=0)

# Load the dataset
X, y, feature_names, _, description = load_dataset(selected_dataset)

# Dataset description
with st.expander('Dataset Description', expanded=False):
    st.markdown(f'**{selected_dataset.capitalize()} Dataset**')
    st.text(description)
    st.markdown(f'**Shape:** {X.shape[0]} samples, {X.shape[1]} features')

# Feature selection for visualization
if X.shape[1] > 1:
    st.sidebar.markdown('### Feature Selection')
    st.sidebar.markdown('Select a feature for visualization:')
    selected_feature = st.sidebar.selectbox('Feature', options=feature_names, index=0)

    # Get index of selected feature
    feature_idx = feature_names.index(selected_feature)

    # Filter X to only include selected feature for visualization
    X_vis = X[:, feature_idx].reshape(-1, 1)
    selected_feature_name = selected_feature
else:
    X_vis = X
    selected_feature_name = feature_names[0]

# Standardize the data
st.sidebar.markdown('### Data Preprocessing')
standardize = st.sidebar.checkbox('Standardize Features', value=True)

if standardize:
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
else:
    X_scaled = X
    y_scaled = y

# Train-test split
test_size = st.sidebar.slider('Test Size', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
random_state = st.sidebar.slider('Random State', min_value=0, max_value=100, value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=random_state)

# Model parameters (simplified for demonstration)
st.sidebar.markdown('### Model Parameters')

if selected_algorithm == 'Ridge Regression':
    alpha = st.sidebar.slider('Alpha', min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    regressors[selected_algorithm].set_params(alpha=alpha)

elif selected_algorithm == 'Lasso Regression':
    alpha = st.sidebar.slider('Alpha', min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    regressors[selected_algorithm].set_params(alpha=alpha)

elif selected_algorithm == 'Elastic Net':
    alpha = st.sidebar.slider('Alpha', min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    l1_ratio = st.sidebar.slider('L1 Ratio', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    regressors[selected_algorithm].set_params(alpha=alpha, l1_ratio=l1_ratio)

elif selected_algorithm == 'Decision Tree':
    max_depth = st.sidebar.slider('Maximum Depth', min_value=1, max_value=20, value=5, step=1)
    regressors[selected_algorithm].set_params(max_depth=max_depth)

elif selected_algorithm == 'Random Forest':
    n_estimators = st.sidebar.slider('Number of Trees', min_value=10, max_value=200, value=100, step=10)
    max_depth = st.sidebar.slider('Maximum Depth', min_value=1, max_value=20, value=5, step=1)
    regressors[selected_algorithm].set_params(n_estimators=n_estimators, max_depth=max_depth)

elif selected_algorithm == 'Gradient Boosting':
    n_estimators = st.sidebar.slider('Number of Estimators', min_value=10, max_value=200, value=100, step=10)
    learning_rate = st.sidebar.slider('Learning Rate', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    regressors[selected_algorithm].set_params(n_estimators=n_estimators, learning_rate=learning_rate)

elif selected_algorithm == 'Support Vector Regression':
    C = st.sidebar.slider('Regularization (C)', min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    kernel = st.sidebar.selectbox('Kernel', options=['linear', 'poly', 'rbf', 'sigmoid'], index=2)
    regressors[selected_algorithm].set_params(C=C, kernel=kernel)

elif selected_algorithm == 'K Nearest Neighbors':
    n_neighbors = st.sidebar.slider('Number of Neighbors', min_value=1, max_value=20, value=5, step=1)
    regressors[selected_algorithm].set_params(n_neighbors=n_neighbors)

# Train the model
regressor = regressors[selected_algorithm]
regressor.fit(X_train, y_train)

# Predictions
y_pred = regressor.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader('Model Performance')
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('Mean Squared Error', f'{mse:.4f}')

with col2:
    st.metric('Root Mean Squared Error', f'{rmse:.4f}')

with col3:
    st.metric('Mean Absolute Error', f'{mae:.4f}')

with col4:
    st.metric('R² Score', f'{r2:.4f}')

# Visualize predictions (simplified for 1D case)
st.subheader('Prediction Visualization')

if 1 in (X.shape[1], X_vis.shape[1]):
    # For 1D data, we'll plot the regression line
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get the feature values for visualization
    if standardize:
        # Create a separate scaler for just the visualization feature
        # to avoid the dimension mismatch
        scaler_X_vis = StandardScaler()
        X_vis_scaled = scaler_X_vis.fit_transform(X_vis)

        # Use the appropriate feature from training/test data
        if X.shape[1] > 1:
            X_vis_train = X_train[:, feature_idx].reshape(-1, 1)
            X_vis_test = X_test[:, feature_idx].reshape(-1, 1)
        else:
            X_vis_train = X_train
            X_vis_test = X_test

        # Create a range of values for the regression line
        X_line = np.linspace(X_vis_train.min(), X_vis_train.max(), 100).reshape(-1, 1)
        if X.shape[1] > 1:
            X_line_full = np.zeros((100, X.shape[1]))
            X_line_full[:, feature_idx] = X_line.ravel()
            y_line = regressor.predict(X_line_full)
        else:
            y_line = regressor.predict(X_line)

        # Plot the training data
        ax.scatter(X_vis_train, y_train, color='blue', alpha=0.6, label='Training Data', s=50)

        # Plot the testing data
        ax.scatter(X_vis_test, y_test, color='green', alpha=0.6, label='Testing Data', s=50)

        # Plot the predictions
        ax.scatter(
            X_vis_test,
            y_pred,
            color='red',
            alpha=0.6,
            label='Predictions',
            s=50,
            marker='x',
        )

        # Plot the regression line
        ax.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')

        # If data was standardized, show original scale on a second axis
        if standardize:
            ax_orig = ax.twinx()
            # Dummy to not affect the plot
            ax_orig.scatter([], [], alpha=0)

            # Set labels with original scale
            x_ticks = ax.get_xticks()
            y_ticks = ax.get_yticks()

            ax_orig.set_yticks(y_ticks)
            labels = [f'{scaler_y.inverse_transform([[y]])[0][0]:.2f}' for y in y_ticks]
            ax_orig.set_yticklabels(labels)
            ax_orig.set_ylabel('Original Target Scale')
    else:
        # If not standardized, just use the original data
        # Plot the training data
        if X.shape[1] > 1:
            X_train_plot = X_train[:, feature_idx].reshape(-1, 1)
            X_test_plot = X_test[:, feature_idx].reshape(-1, 1)
        else:
            X_train_plot = X_train
            X_test_plot = X_test

        ax.scatter(X_train_plot, y_train, color='blue', alpha=0.6, label='Training Data', s=50)

        # Plot the testing data
        ax.scatter(X_test_plot, y_test, color='green', alpha=0.6, label='Testing Data', s=50)

        # Plot the predictions
        ax.scatter(
            X_test_plot,
            y_pred,
            color='red',
            alpha=0.6,
            label='Predictions',
            s=50,
            marker='x',
        )

        # Create a range of values for the regression line
        X_line = np.linspace(X_vis.min(), X_vis.max(), 100).reshape(-1, 1)
        if X.shape[1] > 1:
            X_line_full = np.zeros((100, X.shape[1]))
            X_line_full[:, feature_idx] = X_line.ravel()
            y_line = regressor.predict(X_line_full)
        else:
            y_line = regressor.predict(X_line)

        # Plot the regression line
        ax.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')

    ax.set_xlabel(selected_feature_name)
    ax.set_ylabel('Target' if not standardize else 'Standardized Target')
    ax.set_title(f'{selected_algorithm} Prediction')
    ax.legend()
    st.pyplot(fig)
else:
    # For higher dimensional data, show a scatter plot of actual vs predicted
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(y_test, y_pred, alpha=0.6, s=50)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)

    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{selected_algorithm} - Actual vs Predicted')

    # If data was standardized, show original scale on both axes
    if standardize:
        # Set labels with original scale
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()

        x_labels = [f'{scaler_y.inverse_transform([[x]])[0][0]:.2f}' for x in x_ticks]
        y_labels = [f'{scaler_y.inverse_transform([[y]])[0][0]:.2f}' for y in y_ticks]

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

    st.pyplot(fig)

# Feature importance (if applicable)
if hasattr(regressor, 'feature_importances_') and X.shape[1] > 1:
    st.subheader('Feature Importance')

    importances = regressor.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    st.pyplot(fig)
elif hasattr(regressor, 'coef_') and X.shape[1] > 1:
    st.subheader('Model Coefficients')

    coefs = regressor.coef_
    if coefs.ndim > 1:
        coefs = coefs[0]

    # Create a dataframe of coefficients
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})

    # Sort by absolute coefficient value
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.barh(range(len(coef_df)), coef_df['Coefficient'])
    plt.yticks(range(len(coef_df)), coef_df['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Model Coefficients')

    # Add a vertical line at x=0 for reference
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)

    st.pyplot(fig)
    st.dataframe(coef_df)
