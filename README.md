# End-to-end-Mushroom-dataset-project-with-FastAPI(Python, scikit-learn, FastAPI)

Built an end-to-end machine learning pipeline to classify mushrooms as edible or poisonous using the UCI Mushroom dataset.

Performed data ingestion, preprocessing, and feature encoding using Pandas, ColumnTransformer, and OneHotEncoder.

Trained and evaluated a Decision Tree classifier within a scikit-learn Pipeline, achieving high predictive accuracy.

Persisted the trained model using joblib for reuse in production.

Developed a FastAPI REST API to serve real-time predictions from the trained model.

Designed structured request validation using Pydantic models and returned predictions via HTTP endpoints.

Implemented a clean separation between model training and model serving, following production ML best practices.

Tech stack: Python, Pandas, scikit-learn, FastAPI, Pydantic, Joblib