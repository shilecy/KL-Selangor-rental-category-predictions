import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64
    
# --- 0. FILE CHECK & SETUP ---

# NOTE: These files must exist in the same directory as this script.
try:
    # Load the trained pipeline (which includes all preprocessing steps and the XGBoost model)
    model = joblib.load('final_xgb_full_pipeline.joblib')
    optimal_threshold_data = joblib.load('optimal_threshold.joblib')

    # Load the optimal threshold determined during model evaluation
    if isinstance(optimal_threshold_data, dict):
        if 'optimal_threshold_class_2' in optimal_threshold_data:
            OPTIMAL_THRESHOLD = optimal_threshold_data['optimal_threshold_class_2']
        elif 'threshold' in optimal_threshold_data:
            OPTIMAL_THRESHOLD = optimal_threshold_data['threshold']
        else:
            # Changed fallback to 0.40 to allow more 'High' predictions
            st.warning("Could not find threshold key in file. Using default 0.40.")
            OPTIMAL_THRESHOLD = 0.40
    elif isinstance(optimal_threshold_data, float):
        OPTIMAL_THRESHOLD = optimal_threshold_data
    else:
        # Changed fallback to 0.40 to allow more 'High' predictions
        st.warning("Could not definitively extract optimal threshold. Using default 0.40.")
        OPTIMAL_THRESHOLD = 0.40

        
except FileNotFoundError:
    st.error("🚨 Missing model files! Ensure 'final_xgb_full_pipeline.joblib' and 'optimal_threshold.joblib' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error loading model or threshold files: {e}")
    st.stop()

# --- 1. MAPPINGS AND CONSTANTS ---

# Define rental categories
# *** FIX APPLIED: Mappings for 0 (High) and 2 (Low) have been SWAPPED to correct model inversion. ***
rental_categories = {
    0: "High - From RM3,546 and above",
    1: "Medium - Between RM1,576 to RM3,545",
    2: "Low - From RM1,575 and below"
}

# Define mappings (MUST be placed before the function that uses it)
location_mapping = {
    'Kuala Lumpur': {
        'Ampang': 0, 'Ampang Hilir': 1, 'Bandar Permai Perdana': 2, 'Bandar Manjalara': 3, 
        'Bandar Tasik Selatan': 4, 'Bangsar': 5, 'Bangsar South': 6, 'Brickfields': 7, 
        'Bukit Bintang': 8, 'Bukit Jalil': 9, 'Bukit Tunku': 10, 'Chan Sow Lin': 11, 
        'Cheras': 12, 'City Centre': 13, 'Damansara': 14, 'Damansara Heights': 15, 
        'Desa Pandan': 16, 'Desa Park City': 17, 'Desa Petaling': 18, 'Gombak': 19, 
        'Jalan Ipoh': 20, 'Jalan Kuching': 21, 'Jalan Sultan Ismail': 22, 'Jinjang': 23, 
        'Kepong': 24, 'Keramat': 25, 'KL City': 26, 'KL Eco City': 27, 
        'KL Sentral': 28, 'KLCC': 29, 'Kuchai Lama': 30, 'Mid Valley': 31, 
        'Mont Kiara': 32, 'Old Klang Road': 33, 'Others': 34, 'OUG': 35, 
        'Pandan Indah': 36, 'Pandan Jaya': 37, 'Pandan Perdana': 38, 'Pantai': 39, 
        'Puchong': 40, 'Pudu': 41, 'Salak Selatan': 42, 'Segambut': 43, 
        'Sentul': 44, 'Seputeh': 45, 'Serdang': 46, 'Setapak': 47, 
        'Setiawangsa': 48, 'Solaris Dutamas': 49, 'Sri Damansara': 50, 'Sri Hartamas': 51, 
        'Sri Petaling': 52, 'Sungai Besi': 53, 'Sungai Penchala': 54, 'Taman Desa': 55, 
        'Taman Melawati': 56, 'Taman Tun Dr Ismail': 57, 'Titiwangsa': 58, 'Wangsa Maju': 60
    },
    'Selangor': {
        'Alam Impian': 61, 'Ampang': 62, 'Ara Damansara': 63, 'Balakong': 64, 
        'Bandar Botanic': 65, 'Bandar Bukit Raja': 66, 'Bandar Bukit Tinggi': 67, 'Bandar Kinrara': 68, 
        'Bandar Mahkota Cheras': 69, 'Bandar Saujana Putra': 70, 'Bandar Sri Damansara': 71, 'Bandar Sg Long': 72, 
        'Bandar Sunway': 73, 'Bandar Utama': 74, 'Bangi': 75, 'Banting': 76, 
        'Batu Caves': 77, 'Beranang': 78, 'Bukit Beruntung': 79, 'Bukit Jelutong': 80, 
        'Bukit Subang': 81, 'Cheras': 82, 'Cyberjaya': 83, 'Damansara Damai': 84, 
        'Damansara Jaya': 85, 'Damansara Perdana': 86, 'Dengkil': 87, 'Glenmarie': 88, 
        'Gombak': 89, 'Hulu Langat': 90, 'I-City': 91, 'Jenjarum': 92, 
        'Kajang': 93, 'Kapar': 94, 'Kelana Jaya': 95, 'Klang': 96, 
        'Kota Damansara': 97, 'Kota Kemuning': 98, 'Kuala Langat': 99, 'Kuala Selangor': 100, 
        'Mutiara Damansara': 101, 'Petaling Jaya': 102, 'Port Klang': 103, 'Puchong': 104, 
        'Puchong South': 105, 'Pulau Indah': 106, 'Puncak Alam': 107, 'Puncak Jalil': 108, 
        'Putra Heights': 109, 'Rawang': 110, 'Salak Tinggi': 111, 'Saujana Utama': 112, 
        'Selayang': 113, 'Semenyih': 114, 'Sepang': 115, 'Serdang': 116, 
        'Serendah': 117, 'Seri Kembangan': 118, 'Setia Alam': 119, 'Shah Alam': 120, 
        'Subang Bestari': 121, 'Subang Jaya': 122, 'Sungai Buloh': 123, 'Teluk Panglima Garang': 124, 
        'Ulu Kelang': 125, 'USJ': 126,
    }
}

property_type_mapping = {
    'Apartment': 0, 'Condominium': 1, 'Duplex': 2, 'Flat': 3, 
    'Serviced Residence': 4, 'Studio': 5
}

furnish_mapping = {
    'Not Furnish': 0, 'Partially Furnish': 1, 'Fully Furnish': 2
}

SIZE_BINS_MAPPING = {
    '0-250': 0, '251-500': 1, '501-850': 2, '851-1000': 3, 
    '1001-1250': 4, '1251-1500': 5, '1501-1850': 6, 
    '1851-2000': 7, '2001-2250': 8, '2251-2500': 9
}

# --- 2. HELPER FUNCTIONS ---

def safe_int(value):
    """Safely converts string to int, defaulting to 0."""
    try:
        # Check if value is a string (like from a selectbox) before converting
        if isinstance(value, str):
            value = value.strip()
            if not value: return 0
        return int(value)
    except (ValueError, TypeError):
        return 0

def get_base64_of_bin_file(bin_file):
    """Encodes a local file to base64 for use in CSS."""
    # Note: Assumes 'background.jpg' is available for the UI
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return "" # Return empty if file fails to load

def set_background(png_file):
    """Applies custom CSS for a background image."""
    bin_str = get_base64_of_bin_file(png_file)
    if not bin_str:
        return # Skip if file loading failed
        
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .stApp > header, .stApp > footer {{
        background-color: rgba(0,0,0,0); /* Make header/footer transparent */
    }}
    .stApp > div:first-child {{
        padding-top: 2rem; /* Add padding to prevent content overlap with header */
    }}
    /* Style for the main content area (e.g., the input boxes) */
    .main .block-container {{
        background: rgba(255, 255, 255, 0.9); /* Semi-transparent white background */
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


def apply_optimal_threshold(probabilities, threshold):
    """
    Applies the custom threshold logic to prevent over-prediction of Class 2 ('High').
    Returns the predicted class index (0, 1, or 2).
    
    *** DEBUGGING MODE: Return the class with the highest raw probability ***
    """
    return np.argmax(probabilities)


# --- 3. PREPROCESS AND PREDICT FUNCTION (28 UI inputs, producing 28 columns in EXACT ORDER) ---

def preprocess_and_predict(completion_year, location, region, property_type, furnished, rooms, parking, bathroom, size_bins, 
              Barbeque_area, Club_house, Gymnasium, Jogging_Track, Lift, Minimart, 
              Multipurpose_hall, Parking_bool, Playground, Sauna, Security, Squash_Court, 
              Swimming_Pool, Tennis_Court, Air_Cond, Cooking_Allowed, Internet, Near_KTM_LRT, Washing_Machine):

    # CRUCIAL FIX: The dictionary keys MUST be in the exact order of the model's training features.
    input_data_dict = {
        # 9 Property/Numerical Features
        'completion_year': 2010,           # Hardcoded Dummy value
        'location': location,              # Input (Location Code)
        'region': 0,                       # Hardcoded Dummy value (assuming region is code 0 if not used)
        'property_type': property_type,    # Input (Property Type Code)
        'furnished': furnished,            # Input (Furnish Code)
        'rooms': rooms,                    # Input (Rooms Count)
        'parking': parking,                # Input (Numerical Carparks Count - lowercase)
        'bathroom': bathroom,              # Input (Bathrooms Count)
        'size_bins': size_bins,            # Input (Size Bin Code)
        
        # 19 Amenity/Facility Features (Boolean flags) (Features 10-28)
        'Barbeque area': int(Barbeque_area), 
        'Club house': int(Club_house), 
        'Gymnasium': int(Gymnasium), 
        'Jogging Track': int(Jogging_Track),
        'Lift': int(Lift), 
        'Minimart': int(Minimart),
        'Multipurpose hall': int(Multipurpose_hall), 
        'Parking': int(Parking_bool),       # Input (Boolean Presence, Capitalized 'Parking')
        'Playground': int(Playground), 
        'Sauna': int(Sauna),
        'Security': int(Security), 
        'Squash Court': int(Squash_Court), 
        'Swimming Pool': int(Swimming_Pool), 
        'Tennis Court': int(Tennis_Court), 
        'Air-Cond': int(Air_Cond), 
        'Cooking Allowed': int(Cooking_Allowed), 
        'Internet': int(Internet), 
        'Near KTM/LRT': int(Near_KTM_LRT), 
        'Washing Machine': int(Washing_Machine)
    }

    # Creating the DataFrame from the ordered dictionary
    input_data_df = pd.DataFrame([input_data_dict])
    
    # Run prediction
    probabilities = model.predict_proba(input_data_df)[0]
    
    # Apply the custom threshold to get the final prediction (0, 1, or 2)
    final_class_prediction_index = apply_optimal_threshold(probabilities, OPTIMAL_THRESHOLD)
    
    # MODIFIED: Return both the category string and the raw probabilities array
    return rental_categories[final_class_prediction_index], probabilities


# --- 4. STREAMLIT UI & INPUT GATHERING ---

try:
    set_background('background.jpg')
except FileNotFoundError:
    st.warning("⚠️ Background image 'background.jpg' not found. Using default Streamlit background.")
except Exception as e:
    st.warning(f"⚠️ Could not load background image: {e}")

# Customizing title text color & button style
st.markdown("<h1 style='color: orange;'>CHECK YOUR RENTAL RATES NOW!</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stButton>button {
        color: white;
        background-color: orange;
        border-radius: 8px;
    }
    h1 { font-size: 45px;}
    h3 { font-size: 20px;}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("### Predict the rental category (Low, Medium, or High) for a property listing.")

# ----------------- Input Columns -----------------
col1, col2 = st.columns(2)

with col1:
    # --- ADDED REGION DROPDOWN ---
    region_options = list(location_mapping.keys())
    region_select = st.selectbox("1. Select Region (Kuala Lumpur or Selangor)", 
                                 options=[""] + region_options,
                                 key="region_select")
    
    # Filter locations based on selected region
    if region_select and region_select != "":
        available_locations = list(location_mapping[region_select].keys())
    else:
        available_locations = []
        
    location = st.selectbox("2. Select Location", 
                            options=[""] + available_locations,
                            key="location_select")
    
    # Get the numeric location code
    location_code = -1
    if region_select and location and location != "":
        location_code = location_mapping[region_select].get(location, -1)
    
    property_type = st.selectbox("4. Select Property Type", options=[""] + list(property_type_mapping.keys()))
    property_type_code = property_type_mapping.get(property_type, -1)

    furnish = st.selectbox("5. Select Your Furnish Type", options=[""] + list(furnish_mapping.keys()))
    furnish_code = furnish_mapping.get(furnish, -1)

with col2:

    bedrooms = st.selectbox("3. Number of Bedrooms", options=["", "1", "2", "3", "4", "5", "6"])
    bedrooms_int = safe_int(bedrooms) # Use fixed variable name

    bathrooms = st.selectbox("6. Number of Bathrooms", options=["", "1", "2", "3", "4"])
    bathrooms_int = safe_int(bathrooms) # Use fixed variable name
    
    # Highly Important Categorical Feature
    size_bins = st.selectbox("8. Select Unit Size Range (size_bins)", options=[""] + list(SIZE_BINS_MAPPING.keys()))
    size_bins_code = SIZE_BINS_MAPPING.get(size_bins, -1)
    
    # Highly Important Numerical Feature (lowercase 'parking' for the model input)
    parking_count = st.selectbox("7. Number of Carparks", options=["", "1", "2", "3", "4"])
    parking_count_int = safe_int(parking_count)

st.markdown("---")
st.subheader("Property Amenities and Facilities")

# ----------------- Checkbox Columns (19 Features) -----------------
feature_col, facility_col, amenity_col = st.columns(3)

with feature_col:
    st.markdown("##### Unit Features")
    aircon = st.checkbox("Air-Conditioning")
    cooking = st.checkbox("Cooking Allowed")
    internet = st.checkbox("Internet")
    washing_machine = st.checkbox("Washing Machine")
    # Boolean Parking Presence (Capitalized 'Parking' for the model input)
    parking_bool = st.checkbox("Parking")

with facility_col:
    st.markdown("##### Residential Facilities")
    bbq_area = st.checkbox("Barbeque Area")
    club_house = st.checkbox("Club House")
    gym = st.checkbox("Gymnasium")
    jogging_track = st.checkbox("Jogging Track")
    lift = st.checkbox("Lift Access")
    multipurpose_hall = st.checkbox("Multipurpose Hall")
    playground = st.checkbox("Playground")
    sauna = st.checkbox("Sauna")
    security = st.checkbox("24hr Security")
    squash_court = st.checkbox("Squash Court")
    swimming_pool = st.checkbox("Swimming Pool")
    tennis_court = st.checkbox("Tennis Court")
    
with amenity_col:
    st.markdown("##### Amenities Nearby")
    mrt_lrt_ktm = st.checkbox("Near KTM/LRT")
    minimart = st.checkbox("Minimart")


# --- 5. INPUT CLEANING & BUTTON LOGIC ---

st.markdown("---")
btn = st.button("Predict Rental Category", use_container_width=True, type="primary")

if btn:
    # 1. Validation check for ALL CRITICAL FIELDS (Now includes region)
    if (region_select == "" or 
        location == "" or 
        location_code == -1 or
        property_type == "" or 
        property_type_code == -1 or
        bedrooms == "" or 
        bathrooms == "" or 
        furnish == "" or
        furnish_code == -1 or
        parking_count == "" or 
        size_bins == "" or 
        size_bins_code == -1
        ):
        
        st.error("⚠️ Please ensure all mandatory fields (Selectboxes) are filled before predicting.")
    else:
        # 2. Call the prediction function (28 arguments now, matching the function signature)
        try:            
            # UNPACKING TWO RETURN VALUES: The prediction category (string) and the raw probabilities (array)
            prediction_category, probabilities = preprocess_and_predict(
                completion_year=2010,       # Dummy value to satisfy argument 1
                location=location_code,     # Argument 2: Location Code
                region=0,                   # Dummy value to satisfy argument 3
                property_type=property_type_code, # Argument 4
                furnished=furnish_code,     # Argument 5
                rooms=bedrooms_int,         # Argument 6
                parking=parking_count_int,  # Argument 7: Numerical Carparks Count
                bathroom=bathrooms_int,     # Argument 8
                size_bins=size_bins_code,   # Argument 9
                
                # 19 Amenities follow in order
                Barbeque_area=bbq_area, 
                Club_house=club_house, 
                Gymnasium=gym, 
                Jogging_Track=jogging_track, 
                Lift=lift, 
                Minimart=minimart, 
                Multipurpose_hall=multipurpose_hall, 
                Parking_bool=parking_bool, 
                Playground=playground, 
                Sauna=sauna, 
                Security=security, 
                Squash_Court=squash_court, 
                Swimming_Pool=swimming_pool, 
                Tennis_Court=tennis_court, 
                Air_Cond=aircon, 
                Cooking_Allowed=cooking, 
                Internet=internet, 
                Near_KTM_LRT=mrt_lrt_ktm, 
                Washing_Machine=washing_machine
            )
            st.balloons()
            st.markdown(f'## The Predicted Rental Rate Category is: <span style="color: black; font-weight: bold;">{prediction_category}', 
            unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
            st.warning("Please verify the data types and column names/order in your model training.")

st.markdown("""
<style>
/* Additional custom styling for better look on the background */
button {
    font-size: 1.1rem !important;
    padding: 10px 20px !important;
}
.stSelectbox label, .stTextInput label, .stRadio label {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- 6. DISCLAIMER ---

disclaimer_text = """
**Disclaimer:**  
The Analytics are provided “AS IS” and we do not warranty as to its accuracy. We are not responsible or liable for any claims, damages, losses, expenses, costs or liabilities whatsoever. Please seek professional advice before relying on the Analytics. The Analytics are based on the data available at the date of publication and may be subject to further revision as and when more data is made available to us. We reserve the rights to modify, alter, delete or withdraw the Analytics at any time without notice to you. All news, information, contents and other material displayed on the Website and Services including the Postings are for your general information purpose only and are no substitute for independent research and/or verifications and should not be regarded as a substitute for professional, legal, financial or real estate advice.
"""
st.markdown(disclaimer_text)