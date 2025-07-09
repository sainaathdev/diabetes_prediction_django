# from django.shortcuts import render
# from .ml_utils import predict_diabetes

# def home(request):
#     result = None
#     if request.method == 'POST':
#         features = [
#             float(request.POST.get('Pregnancies')),
#             float(request.POST.get('Glucose')),
#             float(request.POST.get('BloodPressure')),
#             float(request.POST.get('SkinThickness')),
#             float(request.POST.get('Insulin')),
#             float(request.POST.get('BMI')),
#             float(request.POST.get('DiabetesPedigreeFunction')),
#             float(request.POST.get('Age'))
#         ]
#         result = predict_diabetes(features)
#     return render(request, 'index.html', {'result': result})
from django.shortcuts import render
import joblib
import numpy as np
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'diabetes_model.pkl')
model = joblib.load(model_path)

def home(request):
    if request.method == 'POST':
        try:
            # Get form values
            pregnancies = int(request.POST['pregnancies'])
            glucose = int(request.POST['glucose'])
            bloodpressure = int(request.POST['bloodpressure'])
            skinthickness = int(request.POST['skinthickness'])
            insulin = int(request.POST['insulin'])
            bmi = float(request.POST['bmi'])
            dpf = float(request.POST['dpf'])
            age = int(request.POST['age'])

            # Create input array
            input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
            prediction = model.predict(input_data)

            # Set result message
            if prediction[0] == 1:
                result = """
                🩺 <strong>Prediction:</strong> The person is likely to have diabetes (Positive).<br><br>
                🥗 <strong>Diet Recommendations:</strong><br>
                - Follow a balanced low-carb diet (e.g., whole grains, non-starchy vegetables).<br>
                - Avoid sugary drinks and refined carbs.<br>
                - Eat more fiber and lean proteins (like legumes, fish, tofu).<br>
                - Stay hydrated and maintain a healthy meal schedule.<br><br>
                🏃‍♀️ <strong>Lifestyle Tips:</strong><br>
                - Exercise daily (at least 30 minutes).<br>
                - Monitor blood glucose regularly.<br>
                - Consult a healthcare provider or dietician.
                """
            else:
                result = """
                ✅ <strong>Prediction:</strong> The person is not likely to have diabetes (Negative).<br><br>
                🎉 <strong>Great job!</strong> Keep maintaining a healthy lifestyle.<br><br>
                💡 <strong>Precautions & Care:</strong><br>
                - Stay physically active.<br>
                - Eat a well-balanced, nutritious diet.<br>
                - Avoid excess sugar and processed food.<br>
                - Go for regular check-ups and screenings.
                """

            return render(request, 'index.html', {'result': result})
        except Exception as e:
            return render(request, 'index.html', {'result': f'Error: {str(e)}'})
    
    return render(request, 'index.html')
# This view handles the home page and processes the form submission.
# It loads the pre-trained model, processes the input data, and returns the prediction result with