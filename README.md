**README**

**Substance Abuse Treatment Completion Prediction Web App**

This is a web application built using Flask that predicts whether a patient undergoing substance abuse treatment is likely to complete the treatment based on demographic and clinical information. The prediction model is implemented using a PyTorch neural network. Users can input various categorical features related to the patient's demographics, substance use history, treatment history, and other relevant factors. The application then provides a prediction on whether the patient is likely to complete the treatment successfully.

**Instructions**

1. **Installation**

   - Ensure you have Python installed on your system.
   - Install the required Python packages by running:
     ```
     pip install flask torch scikit-learn
     ```
   - Additionally, ensure that you have the saved model file named 'model.pth' and the encoder dictionary file named 'encoder_dict.pkl' in the same directory as the Python script.

2. **Running the Application**

   - Navigate to the directory containing the Python script.
   - Run the following command:
     ```
     python app.py
     ```
   - This will start the Flask development server.
   - Open a web browser and go to `http://localhost:5000` to access the web application.

3. **Usage**

   - Upon accessing the web application, you will see a form with various dropdown menus containing categorical options.
   - Select the appropriate options for each category based on the patient's information.
   - Click the "Predict" button.
   - The application will provide a prediction on whether the patient is likely to complete the substance abuse treatment successfully.

**Files**

- `app.py`: This is the main Python script that contains the Flask application code, model loading, and prediction functions.
- `index.html`: This HTML template file contains the structure and layout for the web interface.
- `result.html`: This HTML template file contains the layout for displaying the prediction result.
- `model.pth`: This file contains the saved PyTorch model parameters.
- `encoder_dict.pkl`: This file contains the saved encoder dictionary used for categorical feature encoding.

**Notes**

- The prediction model is implemented using a PyTorch neural network.
- Categorical features provided by the user are encoded using label encoding based on the encoder dictionary.
- The application provides predictions on whether a patient is likely to complete substance abuse treatment based on demographic and clinical information.

**Disclaimer**

- This application provides predictions based on a machine learning model and input data provided by the user. The predictions should be considered as estimates, and actual treatment outcomes may vary.
- The accuracy of the predictions depends on various factors, including the quality and quantity of the input data and the performance of the prediction model.

Feel free to reach out if you have any questions or encounter any issues while running the application.
