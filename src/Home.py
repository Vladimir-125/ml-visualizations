import streamlit as st

st.set_page_config(
    page_title='ML Visualizations',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown('# Welcome to ML Visualizations 🧠')

st.sidebar.success('Select a visualization from the sidebar.')

st.markdown(
    """
    This application provides interactive visualizations to help understand common
    Machine Learning and Data Science concepts.

    ### Visualizations Available

    - **📊 Data Exploration**: Visualize datasets, distributions, and correlations
    - **🔍 Classification**: Understand decision boundaries of different classification algorithms
    - **📈 Regression**: Compare different regression techniques and their predictions
    - **🔮 Clustering**: Visualize how clustering algorithms group data points
    - **📉 Dimensionality Reduction**: See data in reduced dimensions using PCA, t-SNE, etc.

    ### How to use

    Select a visualization from the sidebar to explore different ML concepts. Each
    visualization includes interactive elements to help you understand the underlying
    principles.

    ### Resources

    - [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
    - [Towards Data Science](https://towardsdatascience.com/)
    - [Kaggle Learn](https://www.kaggle.com/learn)

    ### About

    This project was created to help students and practitioners learn ML concepts
    through interactive visualizations. Feel free to contribute to the project on GitHub!
    """
)

# Add a footer with GitHub link
st.markdown(
    """
    ---
    <div style="text-align: center">
        Created with ❤️ using <a href="https://streamlit.io">Streamlit</a>
    </div>
    """,
    unsafe_allow_html=True,
)
