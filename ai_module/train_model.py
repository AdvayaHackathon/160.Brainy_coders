import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the dataset
data_path = os.path.join("..", "datasets", "simple_alzheimers.csv")
data = pd.read_csv(data_path)

# Convert 'Yes'/'No' to 1/0
binary_columns = ['Forgetfulness', 'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks']
for col in binary_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

# Encode Diagnosis: Positive -> 1, Negative -> 0
label_encoder = LabelEncoder()
data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])  # Positive = 1, Negative = 0

# Drop the 'Name' column
X = data.drop(columns=['Name', 'Diagnosis'])
y = data['Diagnosis']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save the model and label encoder
joblib.dump(model, 'alz_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✅ Model saved as 'alz_model.pkl'")
