from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import DataFrame, read_csv


# File to read the data and create a model for the WCDB dataset
def load_data(path: str) -> DataFrame:
    return read_csv(path,
                    names=[
                        'ID', 'Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness',
                        'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension', 'Radius SE', 'Texture SE',
                        'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE',
                        'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE', 'Radius W', 'Texture W',
                        'Perimeter W', 'Area W', 'Smoothness W', 'Compactness W', 'Concavity W', 'Concave Points W',
                        'Symmetry W', 'Fractal Dimension W'],
                    )


def split_data(x: DataFrame, y: DataFrame) -> (DataFrame, DataFrame, DataFrame, DataFrame):
    return train_test_split(x, y, test_size=0.2, random_state=42)


def create_model(save_path: str) -> str:
    data: DataFrame = load_data("data/data.csv")

    # We drop all the columns and make the diagnosis over radius, perimeter and area
    full_x = data.drop(['ID', 'Diagnosis', 'Texture', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                        'Symmetry', 'Fractal Dimension', 'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE',
                        'Smoothness SE', 'Compactness SE', 'Concavity SE', 'Concave Points SE', 'Symmetry SE',
                        'Fractal Dimension SE', 'Radius W', 'Texture W', 'Perimeter W', 'Area W', 'Smoothness W',
                        'Compactness W', 'Concavity W', 'Concave Points W', 'Symmetry W', 'Fractal Dimension W'],
                       axis=1)
    full_y = data['Diagnosis']

    print("Data for the regression:")
    full_x.info()

    print()
    for col in full_x.columns:
        print(col, "max:", full_x[col].max(), "min:", full_x[col].min())
    print()

    x_train, x_test, y_train, y_test = split_data(full_x, full_y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize the Random Forest classifier
    model.fit(x_train, y_train)  # Train the model

    y_pred = model.predict(x_test)  # Predict the test data
    print("Accuracy: ", accuracy_score(y_test, y_pred))  # Print the accuracy of the model

    # Save the model
    from joblib import dump
    dump(model, save_path)

    return save_path


create_model("output/random_forest_model.joblib")
