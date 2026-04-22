# VITYARTHI AI PRO  
## Intelligent Student Productivity & Wellness Prediction System

VITYARTHI AI PRO is an advanced Python-based AI/ML project designed to help students manage daily tasks, analyze productivity, predict task completion time, detect burnout risk, and suggest healthy activities for better academic and personal balance.

This project combines **Task Management**, **Machine Learning**, **Routine Analysis**, **Anomaly Detection**, and **Visual Analytics** into a single intelligent productivity assistant.

---

# 📌 Project Objective

The main objective of this project is to create a smart student productivity system that not only tracks tasks but also uses **Artificial Intelligence and Machine Learning** to:

- Predict daily productivity levels
- Estimate average task completion time
- Analyze routine patterns
- Detect abnormal or unhealthy work behavior
- Identify burnout risk
- Recommend hobbies and healthy activities
- Provide weekly performance insights

This project solves a real-life problem faced by students: **poor time management, stress, screen overuse, and productivity imbalance**.

---

# 🚀 Key Features

## 1. Smart Task Management
- Add tasks with:
  - Task name
  - Category
  - Priority
  - Estimated completion time
  - Due date
- Mark tasks as completed
- Store actual time taken for completed tasks

## 2. Productivity Prediction (AI/ML)
- Uses **Random Forest Classifier**
- Predicts daily productivity level:
  - **Productive**
  - **Balanced**
  - **Low Productive**

## 3. Task Time Prediction
- Uses **Random Forest Regressor**
- Predicts average task completion time for the day

## 4. Routine Pattern Analysis
- Uses **K-Means Clustering**
- Groups user behavior into routine clusters based on activity patterns

## 5. Burnout Risk Detection
- Predicts burnout risk:
  - **Low**
  - **Medium**
  - **High**
- Uses lifestyle indicators such as:
  - stress level
  - sleep hours
  - screen time
  - overdue tasks

## 6. Anomaly Detection
- Uses **Isolation Forest**
- Detects unusual or abnormal daily patterns such as:
  - sudden stress spikes
  - excessive screen time
  - unusually low productivity

## 7. Smart Hobby Recommendation
- Suggests hobbies and wellness activities based on:
  - productivity level
  - stress level
  - screen time
  - free hours

## 8. Visual Analytics
- Generates graphs using **Matplotlib**
- Displays:
  - tasks completed over time
  - stress trend
  - screen time distribution

## 9. Weekly AI Report
- Generates a weekly summary with:
  - average tasks completed
  - average stress level
  - average screen time
  - average sleep hours
  - weekly productivity score out of 100

## 10. Model Persistence
- Saves trained models using **Joblib**
- Avoids unnecessary retraining
- Makes the system more efficient and realistic

---

# 🧠 Machine Learning Techniques Used

This project uses multiple machine learning algorithms to make it a complete AI/ML-based productivity system.

## 1. Random Forest Classifier
Used for:
- Predicting the user's daily productivity level

## 2. Random Forest Regressor
Used for:
- Predicting average task completion time

## 3. K-Means Clustering
Used for:
- Identifying behavioral routine patterns

## 4. Isolation Forest
Used for:
- Detecting anomalies or unusual routine days

## 5. StandardScaler
Used for:
- Feature scaling before clustering and model prediction

## 6. Label Encoding
Used for:
- Converting productivity labels into machine-readable format

---

# 🛠️ Technologies Used

- **Python 3**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Joblib**
- **JSON**
- **CSV**

---

# 📂 Project Structure

```bash
VITYARTHI_AI_PRO/
│
├── vityarthi_ai_pro.py              # Main Python project file
├── README.md                        # Project documentation
├── advanced_tasks.json              # Stores user tasks
├── advanced_routine_history.csv     # Stores routine history dataset
│
└── saved_models/
    ├── productivity_classifier.pkl
    ├── time_regressor.pkl
    ├── scaler.pkl
    ├── kmeans.pkl
    └── label_encoder.pkl# SmartRoutine-tracker-AI
The AI-based daily tracker successfully manages tasks, studies user routine patterns, predicts productivity, and suggests hobbies. It acts like a smart personal assistant that continuously learns from user data, making it a practical real-life machine learning application.
# SmartRoutine AI: An Intelligent Daily Task Tracker and Hobby Recommendation System

## Overview
SmartRoutine AI is a Python-based AI and Machine Learning project that helps users manage daily tasks, analyze routine patterns, predict productivity levels, and recommend new hobbies based on behavior. The system stores routine data, retrains itself with new entries, and acts like a smart personal productivity assistant.

## Features
- Add, view, and complete daily tasks
- Store tasks using JSON
- Log daily routine details
- Predict day type: Productive / Balanced / Lazy
- Use Random Forest Classifier for productivity prediction
- Use KMeans Clustering for routine pattern analysis
- Suggest hobbies based on free time, stress level, and task completion
- Retrain automatically as new routine data is added

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- JSON
- CSV
- Matplotlib (optional for future enhancement)

## Project Structure
- `main.py` - Main Python program
- `daily_tasks.json` - Stores tasks
- `routine_history.csv` - Stores routine history
- `README.md` - Project documentation

## How It Works
1. User adds tasks with priorities.
2. User marks tasks as completed.
3. At the end of the day, the system logs:
   - tasks completed
   - free hours
   - stress level
   - screen time
4. The AI model:
   - predicts the day type using Random Forest
   - identifies behavior pattern using KMeans clustering
   - suggests a new hobby based on the routine
5. New data is appended to the dataset so the system improves over time.

## Algorithms Used
### 1. Random Forest Classifier
Used to classify the user’s day into:
- Productive
- Balanced
- Lazy

### 2. KMeans Clustering
Used to identify hidden patterns in user behavior and group routine types.

## Sample Output
```text
===== AI DAILY REPORT =====
Predicted Day Type: Balanced
Routine Cluster: 1
Suggested Hobby: Photography
AI Suggestion: You are doing well. Try a creative hobby for growth.
```

## Installation
```bash
pip install pandas numpy scikit-learn
```

## Run the Project
```bash
python main.py
```

## Future Enhancements
- GUI using Tkinter
- Graphs using Matplotlib
- Weekly productivity dashboard
- Mood-based recommendations
- Notifications and reminders
- Database integration
- Deep learning-based personalization

## Conclusion
SmartRoutine AI is a practical real-life AI/ML mini project that combines task management, productivity analysis, routine learning, and hobby recommendation in one intelligent system.

## Author
Prepared for academic mini-project / AI-ML project submission.
