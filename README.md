---
title: CardioPredict - Heart Disease Risk Assessment
emoji: ❤️
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
---
❤️ Cardiovascular Risk Prediction Project
1. Introduction

Cardiovascular diseases are one of the leading causes of death worldwide. Early identification of risk factors can help in preventing serious heart-related conditions. This project focuses on predicting the risk of cardiovascular disease using Machine Learning techniques based on user-provided health data.

2. Objective

The main objective of this project is to develop a machine learning-based system that can predict whether a person has a low or high risk of cardiovascular disease. The project also aims to demonstrate the practical application of machine learning in the healthcare domain.

3. System Description

The system accepts basic health details such as age, gender, blood pressure, cholesterol levels, and other medical parameters through a web interface. The input data is preprocessed using a trained scaler and then passed to a trained machine learning model. Based on the analysis, the model predicts the cardiovascular risk and displays the result to the user.

4. Technologies Used

Python

Flask (Web Framework)

Machine Learning

joblib (Model saving and loading)

HTML (User Interface)

Vercel (Deployment support)

5. Project Structure

The project consists of the main application file app.py, a trained machine learning model stored in optimal_model.joblib, a scaler file scaler.joblib for data preprocessing, requirements.txt for dependency management, Procfile and vercel.json for deployment configuration, and a templates folder containing HTML files for the user interface.

6. Methodology

Data provided by the user is first scaled using a pre-trained scaler. The scaled data is then processed by the machine learning model to classify the cardiovascular risk. The final prediction result is displayed through the web application.

7. How to Run the Project

To run the project locally, clone the repository, install the required Python libraries using the requirements.txt file, and execute the app.py file. After running the application, open a web browser and access the application using the local server address shown in the terminal.

8. Results

The system successfully predicts whether the user falls under a low-risk or high-risk category for cardiovascular disease based on the input data.

9. Applications

This project can be used as an educational tool to understand the application of machine learning in healthcare. It can also serve as a base model for more advanced medical decision-support systems.

10. Limitations

The prediction depends on the quality and range of the data used to train the model. This system should not be used as a replacement for professional medical diagnosis.

11. Conclusion

The Cardiovascular Risk Prediction Project demonstrates how machine learning can be effectively used to analyze health data and predict disease risk. The project highlights the importance of technology in supporting early healthcare decision-making.

12. Disclaimer

This project is developed for academic and educational purposes only and should not be considered a medical diagnostic tool.

13. Author

Navya Lohitha
GitHub Username: Navyalohitha
