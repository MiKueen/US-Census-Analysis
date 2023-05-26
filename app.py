from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the preprocessor object
preprocessor = joblib.load('artifacts/preprocessor.pkl')

# Load the selected indices
selected_indices = np.load('artifacts/selected_indices.npy')

# Load the trained model
model = joblib.load('artifacts/gradient_boosting.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Handle the file upload
    file = request.files['file']
    df = pd.read_csv(file)  # Assuming the dataset is in CSV format

    # Preprocess the dataset
    X_preprocessed = preprocessor.transform(df)

    # Select the relevant features
    X_selected = X_preprocessed[:, selected_indices]

    # Make predictions
    prediction = model.predict(X_selected)

    # Access the scalar predicted value
    predicted_value = prediction[0]

    # Get the actual value from the DataFrame
    actual_value = df.iloc[0]  # Assuming you want to display the first row

    # Display the predictions
    return render_template('results.html', prediction=predicted_value, input_data=actual_value)

if __name__ == '__main__':
    app.run(debug=True)
