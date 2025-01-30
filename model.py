import joblib
from sklearn.ensemble import RandomForestRegressor

# Train the model
rf = RandomForestRegressor()
rf.fit(X_train, Y_train)

# Make predictions (optional for testing)
Y_pred = rf.predict(X_test_scaled)

# Save the trained model
joblib.dump(rf, "model.pkl")
print("Model saved successfully!")
