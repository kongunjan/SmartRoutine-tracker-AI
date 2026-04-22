# SmartRoutine-tracker-AI
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
