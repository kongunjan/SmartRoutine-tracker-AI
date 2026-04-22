import os
import json
import random
from datetime import datetime, timedelta

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

# =========================================
# FILE PATHS
# =========================================
TASK_FILE = "advanced_tasks.json"
ROUTINE_FILE = "advanced_routine_history.csv"
MODEL_DIR = "saved_models"

CLASSIFIER_FILE = os.path.join(MODEL_DIR, "productivity_classifier.pkl")
REGRESSOR_FILE = os.path.join(MODEL_DIR, "time_regressor.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
KMEANS_FILE = os.path.join(MODEL_DIR, "kmeans.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")

# =========================================
# SETUP
# =========================================
def ensure_setup():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

# =========================================
# DEFAULT DATA CREATION
# =========================================
def create_default_routine_data():
    if not os.path.exists(ROUTINE_FILE):
        data = {
            "date": pd.date_range(end=datetime.today(), periods=40).astype(str),
            "tasks_completed": [3,5,2,6,4,7,1,8,5,6,2,3,7,8,4,6,5,4,7,3,2,8,6,5,4,7,3,2,6,8,5,4,7,6,3,2,5,7,8,4],
            "tasks_pending": [4,2,5,1,3,1,6,1,2,2,5,4,1,1,3,2,2,3,1,4,5,1,2,2,3,1,4,5,2,1,2,3,1,2,4,5,3,1,1,3],
            "free_hours": [5,3,6,2,4,1,7,2,3,2,6,5,1,2,4,3,4,5,2,6,7,2,3,4,5,2,6,7,3,2,4,5,2,3,6,7,4,2,1,5],
            "stress_level": [7,5,8,3,6,2,9,3,5,4,8,7,2,3,6,4,5,6,3,8,9,2,4,5,6,3,8,9,4,3,5,6,3,4,8,9,5,3,2,6],
            "screen_time": [6,5,7,4,5,3,8,4,5,4,7,6,3,4,5,4,5,6,4,7,8,3,4,5,6,4,7,8,5,4,5,6,4,5,7,8,6,4,3,5],
            "sleep_hours": [6,7,5,8,6,8,4,8,7,7,5,6,8,8,6,7,7,6,8,5,4,8,7,7,6,8,5,4,7,8,7,6,8,7,5,4,6,8,8,6],
            "overdue_tasks": [2,1,3,0,1,0,4,0,1,1,3,2,0,0,1,1,1,2,0,3,4,0,1,1,2,0,3,4,1,0,1,2,0,1,3,4,2,0,0,1],
            "focus_score": [60,75,45,88,70,92,35,90,78,82,50,58,94,91,72,80,76,68,89,47,38,93,81,79,66,87,49,40,77,90,74,69,88,83,46,39,71,92,95,67],
            "mood_score": [5,7,4,8,6,9,3,8,7,7,4,5,9,8,6,7,7,6,8,4,3,9,7,7,6,8,4,3,7,8,7,6,8,7,4,3,6,8,9,6],
            "avg_task_time": [70,50,95,40,60,35,110,30,55,45,90,80,25,28,58,48,52,62,33,98,120,27,49,57,65,36,105,118,46,31,54,63,34,44,99,115,61,29,24,59],
        }

        df = pd.DataFrame(data)

        # Create productivity labels based on richer logic
        labels = []
        burnout = []
        for _, row in df.iterrows():
            score = (
                row["tasks_completed"] * 8
                + row["focus_score"] * 0.5
                + row["mood_score"] * 2
                + row["sleep_hours"] * 2
                - row["stress_level"] * 4
                - row["screen_time"] * 2
                - row["overdue_tasks"] * 5
            )

            if score >= 75:
                labels.append("Productive")
            elif score >= 45:
                labels.append("Balanced")
            else:
                labels.append("Low Productive")

            if row["stress_level"] >= 8 or row["sleep_hours"] <= 4 or row["overdue_tasks"] >= 4:
                burnout.append("High")
            elif row["stress_level"] >= 6 or row["sleep_hours"] <= 6:
                burnout.append("Medium")
            else:
                burnout.append("Low")

        df["productivity_label"] = labels
        df["burnout_risk"] = burnout

        df.to_csv(ROUTINE_FILE, index=False)

# =========================================
# TASK MANAGEMENT
# =========================================
def load_tasks():
    if os.path.exists(TASK_FILE):
        with open(TASK_FILE, "r") as file:
            return json.load(file)
    return []

def save_tasks(tasks):
    with open(TASK_FILE, "w") as file:
        json.dump(tasks, file, indent=4)

def add_task():
    task_name = input("Enter task name: ")
    category = input("Enter category (Study/Coding/Health/Personal/etc): ")
    priority = input("Enter priority (High/Medium/Low): ")
    estimated_time = float(input("Enter estimated time (minutes): "))
    due_days = int(input("Enter due in how many days (0 for today): "))

    task = {
        "task": task_name,
        "category": category,
        "priority": priority,
        "estimated_time": estimated_time,
        "actual_time": None,
        "completed": False,
        "created_at": str(datetime.now()),
        "due_date": str((datetime.now() + timedelta(days=due_days)).date())
    }

    tasks = load_tasks()
    tasks.append(task)
    save_tasks(tasks)
    print(f"Task '{task_name}' added successfully!")

def show_tasks():
    tasks = load_tasks()
    if not tasks:
        print("No tasks available.")
        return

    print("\n========== TASK LIST ==========")
    for i, task in enumerate(tasks):
        status = "✔ Done" if task["completed"] else "✘ Pending"
        print(f"{i}. {task['task']} | {task['category']} | Priority: {task['priority']} | Est: {task['estimated_time']} min | Due: {task['due_date']} | {status}")

def complete_task():
    tasks = load_tasks()
    show_tasks()
    if not tasks:
        return

    try:
        index = int(input("Enter task index to mark complete: "))
        if 0 <= index < len(tasks):
            tasks[index]["completed"] = True
            actual_time = float(input("Enter actual time taken (minutes): "))
            tasks[index]["actual_time"] = actual_time
            save_tasks(tasks)
            print(f"Task '{tasks[index]['task']}' marked as completed!")
        else:
            print("Invalid task index.")
    except:
        print("Invalid input.")

# =========================================
# LABELING LOGIC
# =========================================
def generate_productivity_label(tasks_completed, stress_level, screen_time, sleep_hours, overdue_tasks, focus_score, mood_score):
    score = (
        tasks_completed * 8
        + focus_score * 0.5
        + mood_score * 2
        + sleep_hours * 2
        - stress_level * 4
        - screen_time * 2
        - overdue_tasks * 5
    )

    if score >= 75:
        return "Productive"
    elif score >= 45:
        return "Balanced"
    else:
        return "Low Productive"

def generate_burnout_risk(stress_level, sleep_hours, overdue_tasks, screen_time):
    if stress_level >= 8 or sleep_hours <= 4 or overdue_tasks >= 4 or screen_time >= 9:
        return "High"
    elif stress_level >= 6 or sleep_hours <= 6 or overdue_tasks >= 2:
        return "Medium"
    else:
        return "Low"

# =========================================
# LOG DAILY ROUTINE
# =========================================
def log_daily_routine():
    tasks = load_tasks()
    tasks_completed = sum(1 for t in tasks if t["completed"])
    tasks_pending = sum(1 for t in tasks if not t["completed"])

    try:
        free_hours = float(input("Enter free hours today: "))
        stress_level = int(input("Enter stress level (1-10): "))
        screen_time = float(input("Enter screen time (hours): "))
        sleep_hours = float(input("Enter sleep hours: "))
        overdue_tasks = int(input("Enter overdue tasks count: "))
        focus_score = int(input("Enter focus score (1-100): "))
        mood_score = int(input("Enter mood score (1-10): "))

        completed_tasks = [t for t in tasks if t["completed"] and t["actual_time"] is not None]
        if completed_tasks:
            avg_task_time = np.mean([t["actual_time"] for t in completed_tasks])
        else:
            avg_task_time = random.randint(40, 90)

        productivity_label = generate_productivity_label(
            tasks_completed, stress_level, screen_time, sleep_hours, overdue_tasks, focus_score, mood_score
        )

        burnout_risk = generate_burnout_risk(stress_level, sleep_hours, overdue_tasks, screen_time)

        new_entry = pd.DataFrame([{
            "date": str(datetime.now().date()),
            "tasks_completed": tasks_completed,
            "tasks_pending": tasks_pending,
            "free_hours": free_hours,
            "stress_level": stress_level,
            "screen_time": screen_time,
            "sleep_hours": sleep_hours,
            "overdue_tasks": overdue_tasks,
            "focus_score": focus_score,
            "mood_score": mood_score,
            "avg_task_time": avg_task_time,
            "productivity_label": productivity_label,
            "burnout_risk": burnout_risk
        }])

        if os.path.exists(ROUTINE_FILE):
            df = pd.read_csv(ROUTINE_FILE)
            df = pd.concat([df, new_entry], ignore_index=True)
        else:
            df = new_entry

        df.to_csv(ROUTINE_FILE, index=False)
        print("Routine logged successfully! AI has learned from your data.")

    except:
        print("Invalid input. Please enter valid numbers.")

# =========================================
# MODEL TRAINING
# =========================================
def train_models():
    df = pd.read_csv(ROUTINE_FILE)

    features = [
        "tasks_completed", "tasks_pending", "free_hours", "stress_level",
        "screen_time", "sleep_hours", "overdue_tasks", "focus_score", "mood_score"
    ]

    X = df[features]
    y_class = df["productivity_label"]
    y_reg = df["avg_task_time"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    le = LabelEncoder()
    y_class_encoded = le.fit_transform(y_class)

    # Train/test split for classification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_class_encoded, test_size=0.2, random_state=42
    )

    # Productivity Classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    # Time Prediction Regressor
    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X_scaled, y_reg)

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Save models
    joblib.dump(clf, CLASSIFIER_FILE)
    joblib.dump(reg, REGRESSOR_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(kmeans, KMEANS_FILE)
    joblib.dump(le, ENCODER_FILE)

    print("\n===== MODEL TRAINING COMPLETE =====")
    print(f"Productivity Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

# =========================================
# LOAD MODELS
# =========================================
def load_models():
    if not all(os.path.exists(f) for f in [CLASSIFIER_FILE, REGRESSOR_FILE, SCALER_FILE, KMEANS_FILE, ENCODER_FILE]):
        print("Models not found. Training models first...")
        train_models()

    clf = joblib.load(CLASSIFIER_FILE)
    reg = joblib.load(REGRESSOR_FILE)
    scaler = joblib.load(SCALER_FILE)
    kmeans = joblib.load(KMEANS_FILE)
    le = joblib.load(ENCODER_FILE)

    return clf, reg, scaler, kmeans, le

# =========================================
# HOBBY / SMART RECOMMENDATION ENGINE
# =========================================
def suggest_hobby(predicted_label, stress_level, screen_time, free_hours):
    hobbies = {
        "Productive": ["Learn Python", "Read a technical book", "Build a mini project", "Practice coding challenges"],
        "Balanced": ["Photography", "Cooking", "Music practice", "Sketching"],
        "Low Productive": ["Walking", "Yoga", "Meditation", "Gardening"]
    }

    if stress_level >= 8:
        return random.choice(["Meditation", "Breathing exercises", "Nature walk", "Light music"])

    if screen_time >= 8:
        return random.choice(["Outdoor walk", "Yoga", "Gardening", "Stretching"])

    if free_hours >= 5:
        bonus = ["Language learning", "Chess", "Blogging", "Painting"]
        return random.choice(hobbies[predicted_label] + bonus)

    return random.choice(hobbies[predicted_label])

# =========================================
# AI DAILY ANALYSIS
# =========================================
def ai_daily_analysis():
    tasks = load_tasks()
    tasks_completed = sum(1 for t in tasks if t["completed"])
    tasks_pending = sum(1 for t in tasks if not t["completed"])

    try:
        free_hours = float(input("Enter free hours today: "))
        stress_level = int(input("Enter stress level (1-10): "))
        screen_time = float(input("Enter screen time (hours): "))
        sleep_hours = float(input("Enter sleep hours: "))
        overdue_tasks = int(input("Enter overdue tasks count: "))
        focus_score = int(input("Enter focus score (1-100): "))
        mood_score = int(input("Enter mood score (1-10): "))

        clf, reg, scaler, kmeans, le = load_models()

        input_df = pd.DataFrame([{
            "tasks_completed": tasks_completed,
            "tasks_pending": tasks_pending,
            "free_hours": free_hours,
            "stress_level": stress_level,
            "screen_time": screen_time,
            "sleep_hours": sleep_hours,
            "overdue_tasks": overdue_tasks,
            "focus_score": focus_score,
            "mood_score": mood_score
        }])

        input_scaled = scaler.transform(input_df)

        pred_class_encoded = clf.predict(input_scaled)[0]
        predicted_label = le.inverse_transform([pred_class_encoded])[0]

        predicted_task_time = reg.predict(input_scaled)[0]

        cluster = kmeans.predict(input_scaled)[0]

        burnout_risk = generate_burnout_risk(stress_level, sleep_hours, overdue_tasks, screen_time)

        hobby = suggest_hobby(predicted_label, stress_level, screen_time, free_hours)

        # Anomaly Detection
        df = pd.read_csv(ROUTINE_FILE)
        anomaly_features = [
            "tasks_completed", "tasks_pending", "free_hours", "stress_level",
            "screen_time", "sleep_hours", "overdue_tasks", "focus_score", "mood_score"
        ]
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(df[anomaly_features])

        anomaly_flag = iso.predict(input_df)[0]
        anomaly_text = "Unusual Day Detected" if anomaly_flag == -1 else "Normal Pattern Day"

        print("\n========== AI DAILY REPORT ==========")
        print(f"Predicted Productivity: {predicted_label}")
        print(f"Predicted Avg Task Completion Time: {predicted_task_time:.2f} minutes")
        print(f"Routine Cluster: {cluster}")
        print(f"Burnout Risk: {burnout_risk}")
        print(f"Anomaly Detection: {anomaly_text}")
        print(f"Recommended Hobby: {hobby}")

        if burnout_risk == "High":
            print("AI Suggestion: High burnout risk. Reduce screen time, take proper rest, and do a light activity.")
        elif predicted_label == "Low Productive":
            print("AI Suggestion: Start tomorrow with 1 high-priority task and reduce distractions.")
        elif predicted_label == "Balanced":
            print("AI Suggestion: Good balance. Add one skill-building task for growth.")
        else:
            print("AI Suggestion: Excellent performance. Consider learning an advanced new skill.")

    except:
        print("Invalid input. Please enter valid numbers.")

# =========================================
# VISUAL ANALYTICS
# =========================================
def show_visual_analytics():
    if not os.path.exists(ROUTINE_FILE):
        print("No routine data found.")
        return

    df = pd.read_csv(ROUTINE_FILE)

    plt.figure(figsize=(10, 5))
    plt.plot(df["tasks_completed"], marker='o')
    plt.title("Tasks Completed Over Time")
    plt.xlabel("Day Index")
    plt.ylabel("Tasks Completed")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df["stress_level"], marker='o')
    plt.title("Stress Level Trend")
    plt.xlabel("Day Index")
    plt.ylabel("Stress Level")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(df["screen_time"], bins=8)
    plt.title("Screen Time Distribution")
    plt.xlabel("Screen Time (hours)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# =========================================
# WEEKLY AI REPORT
# =========================================
def weekly_report():
    if not os.path.exists(ROUTINE_FILE):
        print("No routine data found.")
        return

    df = pd.read_csv(ROUTINE_FILE)

    if len(df) < 7:
        recent = df
    else:
        recent = df.tail(7)

    avg_tasks = recent["tasks_completed"].mean()
    avg_stress = recent["stress_level"].mean()
    avg_screen = recent["screen_time"].mean()
    avg_sleep = recent["sleep_hours"].mean()

    productivity_score = (
        avg_tasks * 10
        + avg_sleep * 4
        - avg_stress * 5
        - avg_screen * 2
    )

    productivity_score = max(0, min(100, productivity_score))

    print("\n========== WEEKLY AI REPORT ==========")
    print(f"Average Tasks Completed: {avg_tasks:.2f}")
    print(f"Average Stress Level: {avg_stress:.2f}")
    print(f"Average Screen Time: {avg_screen:.2f} hours")
    print(f"Average Sleep Hours: {avg_sleep:.2f}")
    print(f"Weekly Productivity Score: {productivity_score:.2f}/100")

    if productivity_score >= 80:
        print("Excellent week! You maintained strong productivity and balance.")
    elif productivity_score >= 60:
        print("Good week! Small improvements in sleep or focus can make it even better.")
    else:
        print("Warning: Your weekly productivity is low. Focus on sleep, lower stress, and prioritize key tasks.")

# =========================================
# MENU SYSTEM
# =========================================
def main():
    ensure_setup()
    create_default_routine_data()

    while True:
        print("\n========== VITYARTHI AI PRO ==========")
        print("1. Add Task")
        print("2. Show Tasks")
        print("3. Complete Task")
        print("4. Log Daily Routine")
        print("5. Train / Retrain AI Models")
        print("6. AI Analyze My Day")
        print("7. Show Visual Analytics")
        print("8. Weekly AI Report")
        print("9. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            add_task()
        elif choice == "2":
            show_tasks()
        elif choice == "3":
            complete_task()
        elif choice == "4":
            log_daily_routine()
        elif choice == "5":
            train_models()
        elif choice == "6":
            ai_daily_analysis()
        elif choice == "7":
            show_visual_analytics()
        elif choice == "8":
            weekly_report()
        elif choice == "9":
            print("Exiting VITYARTHI AI PRO. Stay productive and stay healthy!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()import json
import os
import random
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================
# FILE PATHS
# =========================
TASK_FILE = "daily_tasks.json"
ROUTINE_FILE = "routine_history.csv"

# =========================
# DEFAULT DATASET CREATION
# =========================
def create_default_routine_data():
    if not os.path.exists(ROUTINE_FILE):
        data = {
            "tasks_completed": [3, 5, 2, 6, 4, 7, 1, 8, 5, 6, 2, 3, 7, 8, 4],
            "free_hours": [5, 3, 6, 2, 4, 1, 7, 2, 3, 2, 6, 5, 1, 2, 4],
            "stress_level": [7, 5, 8, 3, 6, 2, 9, 3, 5, 4, 8, 7, 2, 3, 6],
            "screen_time": [6, 5, 7, 4, 5, 3, 8, 4, 5, 4, 7, 6, 3, 4, 5],
            "productivity_label": [
                "Balanced", "Productive", "Lazy", "Productive", "Balanced",
                "Productive", "Lazy", "Productive", "Balanced", "Productive",
                "Lazy", "Balanced", "Productive", "Productive", "Balanced"
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(ROUTINE_FILE, index=False)

# =========================
# TASK MANAGEMENT
# =========================
def load_tasks():
    if os.path.exists(TASK_FILE):
        with open(TASK_FILE, "r") as file:
            return json.load(file)
    return []

def save_tasks(tasks):
    with open(TASK_FILE, "w") as file:
        json.dump(tasks, file, indent=4)

def add_task(task_name, priority):
    tasks = load_tasks()
    task = {
        "task": task_name,
        "priority": priority,
        "completed": False,
        "created_at": str(datetime.now())
    }
    tasks.append(task)
    save_tasks(tasks)
    print(f"Task '{task_name}' added successfully!")

def complete_task(task_index):
    tasks = load_tasks()
    if 0 <= task_index < len(tasks):
        tasks[task_index]["completed"] = True
        save_tasks(tasks)
        print(f"Task '{tasks[task_index]['task']}' marked as completed!")
    else:
        print("Invalid task index.")

def show_tasks():
    tasks = load_tasks()
    if not tasks:
        print("No tasks available.")
        return
    print("\n--- DAILY TASK LIST ---")
    for i, task in enumerate(tasks):
        status = "✔ Done" if task["completed"] else "✘ Pending"
        print(f"{i}. {task['task']} | Priority: {task['priority']} | {status}")

# =========================
# ROUTINE HISTORY
# =========================
def log_daily_routine(tasks_completed, free_hours, stress_level, screen_time):
    new_entry = pd.DataFrame([{
        "tasks_completed": tasks_completed,
        "free_hours": free_hours,
        "stress_level": stress_level,
        "screen_time": screen_time,
        "productivity_label": auto_label(tasks_completed, free_hours, stress_level)
    }])

    if os.path.exists(ROUTINE_FILE):
        df = pd.read_csv(ROUTINE_FILE)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_csv(ROUTINE_FILE, index=False)
    print("Routine logged successfully! AI has learned from your new data.")

# =========================
# AUTO LABEL FOR SELF-TRAINING
# =========================
def auto_label(tasks_completed, free_hours, stress_level):
    if tasks_completed >= 6 and stress_level <= 4:
        return "Productive"
    elif tasks_completed <= 2 and free_hours >= 5:
        return "Lazy"
    else:
        return "Balanced"

# =========================
# TRAIN CLASSIFIER MODEL
# =========================
def train_productivity_model():
    df = pd.read_csv(ROUTINE_FILE)
    X = df[["tasks_completed", "free_hours", "stress_level", "screen_time"]]
    y = df["productivity_label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# =========================
# CLUSTER ROUTINE PATTERN
# =========================
def analyze_routine_pattern():
    df = pd.read_csv(ROUTINE_FILE)
    X = df[["tasks_completed", "free_hours", "stress_level", "screen_time"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df["routine_cluster"] = clusters
    df.to_csv(ROUTINE_FILE, index=False)

    return df, scaler, kmeans

# =========================
# HOBBY SUGGESTION ENGINE
# =========================
def suggest_hobby(tasks_completed, free_hours, stress_level, predicted_label):
    hobbies = {
        "Productive": ["Reading books", "Learning coding", "Journaling", "Fitness challenge"],
        "Balanced": ["Photography", "Cooking", "Music practice", "Sketching"],
        "Lazy": ["Walking", "Yoga", "Gardening", "Meditation"]
    }

    # Stress-based override
    if stress_level >= 8:
        return random.choice(["Meditation", "Breathing exercises", "Nature walk", "Light music"])

    # Free-time based recommendation
    if free_hours >= 5:
        bonus_hobbies = ["Painting", "Language learning", "Blogging", "Chess"]
        return random.choice(hobbies[predicted_label] + bonus_hobbies)

    return random.choice(hobbies[predicted_label])

# =========================
# AI DAILY ANALYSIS
# =========================
def ai_daily_analysis(tasks_completed, free_hours, stress_level, screen_time):
    model = train_productivity_model()
    df, scaler, kmeans = analyze_routine_pattern()

    input_data = pd.DataFrame([{
        "tasks_completed": tasks_completed,
        "free_hours": free_hours,
        "stress_level": stress_level,
        "screen_time": screen_time
    }])

    predicted_label = model.predict(input_data)[0]

    scaled_input = scaler.transform(input_data)
    cluster = kmeans.predict(scaled_input)[0]

    hobby = suggest_hobby(tasks_completed, free_hours, stress_level, predicted_label)

    print("\n===== AI DAILY REPORT =====")
    print(f"Predicted Day Type: {predicted_label}")
    print(f"Routine Cluster: {cluster}")
    print(f"Suggested Hobby: {hobby}")

    if predicted_label == "Lazy":
        print("AI Suggestion: Reduce screen time and add one focused task tomorrow.")
    elif predicted_label == "Balanced":
        print("AI Suggestion: You are doing well. Try a creative hobby for growth.")
    else:
        print("AI Suggestion: Excellent routine! Consider learning an advanced skill.")

# =========================
# MENU SYSTEM
# =========================
def main():
    create_default_routine_data()

    while True:
        print("\n====== SMART AI DAILY TRACKER ======")
        print("1. Add Task")
        print("2. Show Tasks")
        print("3. Complete Task")
        print("4. Log Daily Routine")
        print("5. AI Analyze My Day")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            task_name = input("Enter task name: ")
            priority = input("Enter priority (High/Medium/Low): ")
            add_task(task_name, priority)

        elif choice == "2":
            show_tasks()

        elif choice == "3":
            show_tasks()
            index = int(input("Enter task index to mark complete: "))
            complete_task(index)

        elif choice == "4":
            tasks = load_tasks()
            tasks_completed = sum(1 for t in tasks if t["completed"])
            free_hours = float(input("Enter your free hours today: "))
            stress_level = int(input("Enter your stress level (1-10): "))
            screen_time = float(input("Enter your screen time in hours: "))

            log_daily_routine(tasks_completed, free_hours, stress_level, screen_time)

        elif choice == "5":
            tasks = load_tasks()
            tasks_completed = sum(1 for t in tasks if t["completed"])
            free_hours = float(input("Enter your free hours today: "))
            stress_level = int(input("Enter your stress level (1-10): "))
            screen_time = float(input("Enter your screen time in hours: "))

            ai_daily_analysis(tasks_completed, free_hours, stress_level, screen_time)

        elif choice == "6":
            print("Exiting Smart AI Tracker. Stay productive!")
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
