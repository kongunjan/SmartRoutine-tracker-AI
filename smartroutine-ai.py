import json
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