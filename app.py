import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd

st.title("Outcomes Prediction App")
st.info("""This app uses uploaded pre-processing settings and a neural network model to predict outcomes based on user input values. While 
        it has basic ability to accomodate variations, it's intended to be used for models generated using this [Colab Notebook](https://colab.research.google.com/drive/1NtxhdVMr3PTlQ5gSGznqJdx1drQIkrCO?usp=sharing#scrollTo=GJ4hroAYyHwu).
        Ask Dr. David Liebovitz at Northwestern Feinberg School of Medicine for more information. """)

# ============================
# Step 1: Upload Files
# ============================
st.header("Upload Preprocessor and Model Files")

# Upload the preprocessor (.pkl) file
preproc_file = st.sidebar.file_uploader("Upload the preprocessor (.pkl) file", type=["pkl"])
if preproc_file is not None:
    try:
        preprocessor = pickle.load(preproc_file)
        st.success("Preprocessor loaded successfully!")
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        preprocessor = None
else:
    preprocessor = None

# Upload the Keras model (.keras or .h5) file
model_file = st.sidebar.file_uploader("Upload the Keras model file (.keras or .h5)", type=["keras", "h5"])
if model_file is not None:
    try:
        # Write the uploaded model to a temporary file
        temp_model_path = "temp_model.keras"
        with open(temp_model_path, "wb") as f:
            f.write(model_file.getbuffer())
        model = tf.keras.models.load_model(temp_model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
else:
    model = None

# ============================
# Step 2: Dynamic Input Form
# ============================
if preprocessor is not None and model is not None:
    st.sidebar.header("Enter Patient Data")
    
    # Ask for the name of the label to predict
    label_name = st.sidebar.text_input("Enter the name of the label to predict", "Outcome")
    
    # Retrieve the feature names and encoding details from the preprocessor.
    # These should match what was saved during training.
    feature_columns = preprocessor.get("feature_columns", [])
    encoding_details = preprocessor.get("encoding_details", {})

    # Prepare a mapping for case-insensitive checks.
    # For each column in feature_columns, we can use col.lower() for robust comparisons.
    # For example, "Chol/HDL ratio" will be converted to "chol/hdl ratio".
    
    # --- Check for computed features ---
    # 1. BMI: if columns for "bmi", "height", and "weight" exist.
    computed_bmi = False
    bmi_key = height_key = weight_key = None
    for col in feature_columns:
        if col.lower() == "bmi":
            bmi_key = col
            break
    if bmi_key is not None:
        # Look for height and weight keys (exact match on lower-case).
        for col in feature_columns:
            if col.lower() == "height":
                height_key = col
            if col.lower() == "weight":
                weight_key = col
        if height_key is not None and weight_key is not None:
            computed_bmi = True

    # 2. Waist/Hip: search for any column that contains "waist/hip" and ensure that both "waist" and "hip" exist.
    computed_wh = False
    wh_key = None
    waist_key = hip_key = None
    for col in feature_columns:
        if "waist/hip" in col.lower():
            wh_key = col
            break
    if wh_key is not None:
        for col in feature_columns:
            if col.lower() == "waist":
                waist_key = col
            if col.lower() == "hip":
                hip_key = col
        if waist_key is not None and hip_key is not None:
            computed_wh = True

    # 3. Chol/HDL: search for a column whose name contains "chol/hdl" (e.g., "Chol/HDL ratio"),
    # and then identify the source columns containing "chol" and "hdl" (excluding the computed column).
    computed_chol_hdl = False
    chol_hdl_key = None  # Name of the computed ratio column (e.g., "Chol/HDL ratio")
    chol_key = None      # Source cholesterol column
    hdl_key = None       # Source HDL column

    for col in feature_columns:
        if "chol/hdl" in col.lower():
            chol_hdl_key = col
            break
    # Identify the source column for cholesterol (avoid the computed ratio column)
    for col in feature_columns:
        if "chol" in col.lower() and ("chol/hdl" not in col.lower()):
            chol_key = col
            break
    # Identify the source column for HDL (avoid the computed ratio column)
    for col in feature_columns:
        if "hdl" in col.lower() and ("chol/hdl" not in col.lower()):
            hdl_key = col
            break
    if chol_hdl_key is not None and chol_key is not None and hdl_key is not None:
        computed_chol_hdl = True

    # Dictionary to collect user inputs.
    user_input = {}

    st.sidebar.write("Please provide values for the following features:")
    
    # Loop over each feature. For computed features, skip prompting.
    for col in feature_columns:
        col_lower = col.lower()
        # Skip prompting for BMI if it is computed.
        if computed_bmi and col_lower == "bmi":
            continue
        # Skip prompting for waist/hip if the column name contains "waist/hip"
        if computed_wh and "waist/hip" in col_lower:
            continue
        # Skip prompting for Chol/HDL if the column name contains "chol/hdl"
        if computed_chol_hdl and "chol/hdl" in col_lower:
            continue

        # If the column is a binary categorical feature, show a select box.
        if col in encoding_details:
            # The encoding_details for this column is assumed to be a dict like: {0: most_frequent, 1: least_frequent}
            options = [encoding_details[col][0], encoding_details[col][1]]
            selected = st.sidebar.selectbox(f"{col}", options=options)
            # Convert the raw input to the encoded value.
            encoded_value = 0 if selected == encoding_details[col][0] else 1
            user_input[col] = encoded_value
        else:
            # Otherwise, assume a numeric feature.
            user_input[col] = st.sidebar.number_input(f"Enter value for {col}", value=0.0)
    
    # ============================
    # Step 3: Compute Derived Features & Predict
    # ============================
    if st.button("Predict"):
        # Compute BMI from height and weight if required.
        if computed_bmi:
            try:
                h = user_input[height_key]
                w = user_input[weight_key]
                if h <= 0:
                    st.error("Height must be greater than zero to compute BMI.")
                else:
                    # Use U.S. unit conversion: BMI = (weight in pounds / (height in inches)^2) * 703
                    computed_bmi_value = (w / (h * h)) * 703
                    st.write(f"Computed BMI (from {height_key} and {weight_key}): {computed_bmi_value:.2f}")
                    # Add the computed BMI to the user input.
                    user_input[bmi_key] = computed_bmi_value
            except Exception as e:
                st.error(f"Error computing BMI: {e}")

        # Compute waist/hip ratio if required.
        if computed_wh:
            try:
                waist_val = user_input[waist_key]
                hip_val = user_input[hip_key]
                if hip_val == 0:
                    st.error("Hip value cannot be zero for waist/hip ratio calculation.")
                else:
                    computed_wh_value = waist_val / hip_val
                    st.write(f"Computed Waist/Hip Ratio (from {waist_key} and {hip_key}): {computed_wh_value:.2f}")
                    # Add the computed ratio to the user input.
                    user_input[wh_key] = computed_wh_value
            except Exception as e:
                st.error(f"Error computing Waist/Hip Ratio: {e}")

        # Compute Chol/HDL ratio if required.
        if computed_chol_hdl:
            try:
                chol_val = user_input[chol_key]
                hdl_val = user_input[hdl_key]
                if hdl_val == 0:
                    st.error("HDL value cannot be zero for Chol/HDL ratio calculation.")
                else:
                    computed_chol_hdl_value = chol_val / hdl_val
                    st.write(f"Computed Chol/HDL Ratio (from {chol_key} and {hdl_key}): {computed_chol_hdl_value:.2f}")
                    # Add the computed ratio to the user input.
                    user_input[chol_hdl_key] = computed_chol_hdl_value
            except Exception as e:
                st.error(f"Error computing Chol/HDL Ratio: {e}")

        # Create a DataFrame ensuring the columns are in the same order as expected by the model.
        try:
            input_df = pd.DataFrame([user_input], columns=feature_columns)
        except Exception as e:
            st.error(f"Error constructing input DataFrame: {e}")
            input_df = None

        if input_df is not None:
            # Retrieve the scaler from the preprocessor and transform the input.
            scaler = preprocessor.get("scaler", None)
            if scaler is None:
                st.error("Scaler object not found in preprocessor file!")
            else:
                try:
                    # The scaler expects a 2D array.
                    scaled_input = scaler.transform(input_df)
                except Exception as e:
                    st.error(f"Error during scaling: {e}")
                    scaled_input = None

                if scaled_input is not None:
                    try:
                        # Make prediction using the loaded Keras model.
                        prediction = model.predict(scaled_input)
                        st.subheader("Prediction Result")
                        st.write("Raw model output:", prediction)
                        
                        # Assuming a binary classification model outputting a probability,
                        # convert the probability to a percentage.
                        predicted_probability = prediction[0][0] * 100
                        display_probability = f"{predicted_probability:.2f}%"
                        st.info(f"Your model predicts the probability of {label_name} is :orange[{display_probability}] for the values entered!")
                                                                      
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload both the preprocessor (.pkl) file and the model (.keras/.h5) file to continue.")
