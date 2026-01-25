import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# Sample historical data
# --------------------------
data = [
    {"student":"Juan","VSO":8,"Pronouns":6,"Affix":4},
    {"student":"Maria","VSO":9,"Pronouns":8,"Affix":9},
    {"student":"Pedro","VSO":5,"Pronouns":4,"Affix":6},
    {"student":"Ana","VSO":6,"Pronouns":7,"Affix":5},
    {"student":"Luis","VSO":7,"Pronouns":5,"Affix":6},
]

df = pd.DataFrame(data)

# --------------------------
# Weak area detection
# --------------------------
threshold = 6  # below this is considered weak

def get_weak_areas(row, threshold=6):
    weak = []
    for col in ['VSO','Pronouns','Affix']:
        if row[col] < threshold:
            weak.append(col)
    return weak

df['WeakAreas'] = df.apply(get_weak_areas, axis=1)

# --------------------------
# Feedback generator
# --------------------------
def generate_feedback(weak_areas):
    if not weak_areas:
        return "Excellent! Keep up the good work."
    feedback = "Student needs to improve " + ", ".join(weak_areas) + ". Suggested exercises: "
    exercises = {
        "VSO": "Practice VSO sentence ordering exercises.",
        "Pronouns": "Review pronoun usage in sentences.",
        "Affix": "Do affix drills and root word transformations."
    }
    feedback += "; ".join([exercises[w] for w in weak_areas])
    return feedback

df['Feedback'] = df['WeakAreas'].apply(generate_feedback)

# --------------------------
# Train AI model for weak area prediction
# --------------------------
# Convert WeakAreas to binary columns
for col in ['VSO','Pronouns','Affix']:
    df[col+'_weak'] = df['WeakAreas'].apply(lambda x: 1 if col in x else 0)

X = df[['VSO','Pronouns','Affix']]
y = df[['VSO_weak','Pronouns_weak','Affix_weak']]

model = MultiOutputClassifier(RandomForestClassifier())
model.fit(X, y)

# --------------------------
# Console input for new student
# --------------------------
print("=== Filipino Grammar AI Tutor ===")
vso_score = int(input("Enter VSO score (0-10): "))
pronouns_score = int(input("Enter Pronouns score (0-10): "))
affix_score = int(input("Enter Affix score (0-10): "))

new_student = pd.DataFrame({"VSO":[vso_score],"Pronouns":[pronouns_score],"Affix":[affix_score]})
predicted_weak = model.predict(new_student)[0]
predicted_weak_areas = [col.replace("_weak","") for col,val in zip(y.columns,predicted_weak) if val==1]

feedback = generate_feedback(predicted_weak_areas)

# --------------------------
# Output results
# --------------------------
print("\n=== Analysis Result ===")
if predicted_weak_areas:
    print(f"Weak Areas Detected: {', '.join(predicted_weak_areas)}")
else:
    print("No weak areas detected! Excellent performance.")

print("Feedback:")
print(feedback)

# Show historical student performance
print("\n=== Historical Performance ===")
print(df[['student','VSO','Pronouns','Affix','WeakAreas','Feedback']])
