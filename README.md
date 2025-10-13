Malaysia Rental Rate Predictor (Streamlit App)
This project is an extension project from my previous KL/Selangor rental price prediction project. This is a web application built with Streamlit to predict the rental category (Low, Medium, or High) of residential properties in Kuala Lumpur and Selangor, Malaysia.

The prediction is based on property features, amenities, and location data.

🧠 Model & Prediction Logic
The core prediction is handled by an XGBoost Classifier trained on local property data.

Prediction Categories:

Low: From RM1,575 and below

Medium: Between RM1,576 to RM3,545

High: From RM3,546 and above

Feature Engineering: Features like location, property type, furnishing, and unit size are label-encoded, and amenities are treated as boolean flags.

Thresholding: The app uses an OPTIMAL_THRESHOLD (loaded from optimal_threshold.joblib) to refine the classification boundary, particularly between the Medium and High categories, to improve real-world accuracy (though this feature is currently in debugging mode).