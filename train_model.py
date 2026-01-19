import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Load Data
data = pd.read_csv('isl_keypoints.csv')
X = data.iloc[:, 1:].values  # Features (coordinates)
y = data.iloc[:, 0].values   # Labels (A, B, C...)

# 2. Encode Labels (A -> 0, B -> 1...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
num_classes = len(label_encoder.classes_)

# Save the label mapping for later use
np.save('label_mapping.npy', label_encoder.classes_)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 4. Build Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(84,)), # 42 coords (x,y) * 2 hands
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 6. Save Model
model.save('isl_model.h5')
print("Model trained and saved!")
