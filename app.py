import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd

st.title("Outcomes Prediction App")
st.info(
    """This app uses uploaded pre-processing settings and a neural network model to predict outcomes based on user input values.
    You can either upload your own files or choose from a list of sample files provided.
    While it has a basic ability to accommodate variations, it's intended to be used for models generated using this 
    [Colab Notebook](https://colab.research.google.com/drive/1NtxhdVMr3PTlQ5gSGznqJdx1drQIkrCO?usp=sharing#scrollTo=GJ4hroAYyHwu).
    Ask Dr. David Liebovitz at Northwestern Feinberg School of Medicine for more information."""
)

# ============================
# Step 1: File Source Selection
# ============================
file_source = st.sidebar.radio("Choose file source", options=["Upload your own", "Use sample files"])

if file_source == "Use sample files":
    st.header("Select Sample Preprocessor and Model Files")
    st.info("Select the sample files to use from the options below.")
    
    # Define available sample file options.
    sample_preproc_options = {
        "DM Preprocessor": "sample_pre_processing/dm_preprocessor.pkl",
        # Additional preprocessor options can be added here.
    }
    sample_model_options = {
        "DM K-Fold Model": "sample_models/dm_k_fold_model.keras",
        # Additional model options can be added here.
    }
    
    # Use select boxes to choose a sample file for each.
    selected_preproc = st.sidebar.selectbox("Select a sample preprocessor file", list(sample_preproc_options.keys()))
    selected_model = st.sidebar.selectbox("Select a sample model file", list(sample_model_options.keys()))
    
    sample_preproc_path = sample_preproc_options[selected_preproc]
    sample_model_path = sample_model_options[selected_model]
    
    # Load the sample preprocessor.
    try:
        with open(sample_preproc_path, "rb") as f:
            preprocessor = pickle.load(f)
        st.success(f"Sample preprocessor '{selected_preproc}' loaded successfully!")
    except Exception as e:
        st.error(f"Error loading sample preprocessor from {sample_preproc_path}: {e}")
        preprocessor = None

    # Load the sample model.
    try:
        model = tf.keras.models.load_model(sample_model_path)
        st.success(f"Sample model '{selected_model}' loaded successfully!")
    except Exception as e:
        st.error(f"Error loading sample model from {sample_model_path}: {e}")
        model = None

else:
    st.header("Upload Preprocessor and Model Files")
    st.info("Upload your own files if not using sample files.")
    
    # Upload the preprocessor (.pkl) file.
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

    # Upload the Keras model (.keras or .h5) file.
    model_file = st.sidebar.file_uploader("Upload the Keras model file (.keras or .h5)", type=["keras", "h5"])
    if model_file is not None:
        try:
            # Write the uploaded model to a temporary file.
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
    st.sidebar.header("Enter Model Inputs")
    
    # Ask for the label name.
    label_name = st.sidebar.text_input("Enter the name of the label to predict", "Outcome")
    
    # Retrieve feature columns and encoding details from the preprocessor.
    feature_columns = preprocessor.get("feature_columns", [])
    encoding_details = preprocessor.get("encoding_details", {})

    # --- Check for computed features ---
    computed_bmi = False
    bmi_key = height_key = weight_key = None
    for col in feature_columns:
        if col.lower() == "bmi":
            bmi_key = col
            break
    if bmi_key is not None:
        for col in feature_columns:
            if col.lower() == "height":
                height_key = col
            if col.lower() == "weight":
                weight_key = col
        if height_key is not None and weight_key is not None:
            computed_bmi = True

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

    computed_chol_hdl = False
    chol_hdl_key = None
    chol_key = None
    hdl_key = None
    for col in feature_columns:
        if "chol/hdl" in col.lower():
            chol_hdl_key = col
            break
    for col in feature_columns:
        if "chol" in col.lower() and ("chol/hdl" not in col.lower()):
            chol_key = col
            break
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
        if computed_bmi and col_lower == "bmi":
            continue
        if computed_wh and "waist/hip" in col_lower:
            continue
        if computed_chol_hdl and "chol/hdl" in col_lower:
            continue

        if col in encoding_details:
            options = [encoding_details[col][0], encoding_details[col][1]]
            selected = st.sidebar.selectbox(f"{col}", options=options)
            encoded_value = 0 if selected == encoding_details[col][0] else 1
            user_input[col] = encoded_value
        else:
            user_input[col] = st.sidebar.number_input(f"Enter value for {col}", value=0.0)
    
    # ============================
    # Step 3: Compute Derived Features & Predict
    # ============================
    if st.button("Predict"):
        if computed_bmi:
            try:
                h = user_input[height_key]
                w = user_input[weight_key]
                if h <= 0:
                    st.error("Height must be greater than zero to compute BMI.")
                else:
                    computed_bmi_value = (w / (h * h)) * 703  # U.S. conversion formula.
                    st.write(f"Computed BMI (from {height_key} and {weight_key}): {computed_bmi_value:.2f}")
                    user_input[bmi_key] = computed_bmi_value
            except Exception as e:
                st.error(f"Error computing BMI: {e}")

        if computed_wh:
            try:
                waist_val = user_input[waist_key]
                hip_val = user_input[hip_key]
                if hip_val == 0:
                    st.error("Hip value cannot be zero for waist/hip ratio calculation.")
                else:
                    computed_wh_value = waist_val / hip_val
                    st.write(f"Computed Waist/Hip Ratio (from {waist_key} and {hip_key}): {computed_wh_value:.2f}")
                    user_input[wh_key] = computed_wh_value
            except Exception as e:
                st.error(f"Error computing Waist/Hip Ratio: {e}")

        if computed_chol_hdl:
            try:
                chol_val = user_input[chol_key]
                hdl_val = user_input[hdl_key]
                if hdl_val == 0:
                    st.error("HDL value cannot be zero for Chol/HDL ratio calculation.")
                else:
                    computed_chol_hdl_value = chol_val / hdl_val
                    st.write(f"Computed Chol/HDL Ratio (from {chol_key} and {hdl_key}): {computed_chol_hdl_value:.2f}")
                    user_input[chol_hdl_key] = computed_chol_hdl_value
            except Exception as e:
                st.error(f"Error computing Chol/HDL Ratio: {e}")

        try:
            input_df = pd.DataFrame([user_input], columns=feature_columns)
        except Exception as e:
            st.error(f"Error constructing input DataFrame: {e}")
            input_df = None

        if input_df is not None:
            scaler = preprocessor.get("scaler", None)
            if scaler is None:
                st.error("Scaler object not found in preprocessor file!")
            else:
                try:
                    scaled_input = scaler.transform(input_df)
                except Exception as e:
                    st.error(f"Error during scaling: {e}")
                    scaled_input = None

                if scaled_input is not None:
                    try:
                        prediction = model.predict(scaled_input)
                        st.subheader("Prediction Result")
                        st.write("Raw model output:", prediction)
                        
                        predicted_probability = prediction[0][0] * 100
                        display_probability = f"{predicted_probability:.2f}%"
                        st.info(
                            f"Your model predicts the probability of {label_name} is :orange[{display_probability}] for the values entered!"
                        )
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload both the preprocessor (.pkl) file and the model (.keras/.h5) file, or select 'Use sample files' to continue.")
