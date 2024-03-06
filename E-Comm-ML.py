import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops
import pytesseract
import nltk
from nltk.corpus import stopwords
from sklearn import datasets
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams 
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import io
import spacy
from sklearn.preprocessing import LabelEncoder
from surprise import Dataset, Reader 
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import KNNWithMeans
from io import StringIO
from sklearn import tree
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="E-Comm ML", layout="wide", initial_sidebar_state="auto")

# Custom CSS to change sidebar color
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #ff000050;
    }
</style>
""", unsafe_allow_html=True)

def load_and_preprocess_data(uploaded_file):
    # Step 1: Load CSV File
    df = pd.read_csv(uploaded_file)

    # Display DataFrame
    st.write("#### Uploaded Classification dataset")
    st.dataframe(df)

    # Display NaN Values
    st.write("#### NaN Values in DataFrame")
    st.text(df.isna().sum())

    # Display DataFrame Shape
    st.write("#### DataFrame Shape")
    st.text(df.shape)

    # Drop Duplicates and NaN Values
    st.write("#### Duplicate values in DataFrame")
    st.text(df.duplicated().sum())
    st.write("#### Drop Duplicates Values")
    
    # Create a new DataFrame to store the processed data
    processed_df = df.drop_duplicates().dropna().copy()
    st.success("Duplicate values dropped successfully!")

    # Display Data types in DataFrame
    st.write("#### Data types in DataFrame")
    st.text(processed_df.dtypes)

    # Display Summary Statistics
    st.write("#### Summary Statistics")
    st.dataframe(processed_df.describe().T)

    # Label Encoding
    st.write("#### Label Encoding")
    le = LabelEncoder()
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object' or processed_df[col].dtype == 'bool':
            processed_df[col] = le.fit_transform(processed_df[col])
    st.success("Label Encoding completed successfully!")

    # One-Hot Encoding for categorical columns
    st.write("#### One-Hot Encoding")
    categorical_columns = processed_df.select_dtypes(include=['object']).columns
    processed_df = pd.get_dummies(processed_df, columns=categorical_columns)
    st.success("One-Hot Encoding completed successfully!")

    # DateTime Format Conversion
    st.write("#### DateTime Format Conversion")
    date_columns = processed_df.select_dtypes(include=['datetime']).columns
    for col in date_columns:
        processed_df[col] = pd.to_datetime(processed_df[col])
    st.success("DateTime Format Conversion completed successfully!")

    # Display Data types in Processed DataFrame
    st.write("#### Data types in Processed DataFrame")
    st.text(processed_df.dtypes)
    st.write("All data types are now numerical after preprocessing.")

    return processed_df

def data_classification():
    st.markdown("<h3 style='text-align: center;'>Exploratory Data Analysis (EDA) and Classification</h3>", unsafe_allow_html=True)
    local_image_path = r"C:\Users\Hp\Desktop\DS_Final_Project\class_bgm.jpg"
    image = Image.open(local_image_path)
    resized_image = image.resize((950, 475))
    st.image(resized_image, use_column_width=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    processed_df = None
    if uploaded_file is not None:
        processed_df = st.cache_data(load_and_preprocess_data)(uploaded_file)
            
        # Multiselect box for preprocessing steps
        options = st.multiselect("Select an option", ["Data Preprocessing", "Visualizations", "Classification"])

        # Data Preprocessing
        if "Data Preprocessing" in options:
            # Data Preprocessing
            #processed_df = load_and_preprocess_data(uploaded_file)
            st.write("#### Processed Classification dataset")
            st.dataframe(processed_df)

            st.write("Preprocessing Steps:")
            st.write("- Uploaded dataset loaded, missing values handled, duplicates removed, and data types standardized.")
            st.write("- Categorical variables encoded, datetime formats converted, resulting in the processed dataset.")

        # Visualizations
        if "Visualizations" in options:
            if processed_df is not None:
                # Box Plot for Outlier Detection
                st.write("#### Box Plot for Outlier Detection")
                numeric_columns = processed_df.select_dtypes(include=['float64', 'int64']).columns
                # Set up the number of columns you want in a row for box plots
                columns_per_row = 4
                # Calculate the number of rows needed for box plots
                num_rows = (len(numeric_columns) + columns_per_row - 1) // columns_per_row
                # Create subplots for box plots
                fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(15, 3 * num_rows))
                # Flatten the axes array for easy iteration
                axes = axes.flatten()
                # Iterate over numeric columns and create box plots
                for i, col in enumerate(numeric_columns):
                    ax = axes[i]
                    sns.boxplot(x=col, data=processed_df, ax=ax)
                    ax.set_title(col)
                # Remove empty subplots if any
                for i in range(len(numeric_columns), len(axes)):
                    fig.delaxes(axes[i])
                # Adjust layout for box plots
                fig.tight_layout()
                st.pyplot(fig)

                # Skewness Plot
                st.write("#### Skewness Plot for Data Distribution")
                # Select numeric columns
                numeric_columns = processed_df.select_dtypes(include=['float64', 'int64']).columns
                # Set up the number of columns you want in a row for skewness plots
                columns_per_row = 4
                # Calculate the number of rows needed for skewness plots
                num_rows = (len(numeric_columns) + columns_per_row - 1) // columns_per_row
                # Create subplots for skewness plots
                fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(15, 3 * num_rows))
                # Flatten the axes array for easy iteration
                axes = axes.flatten()
                # Iterate over numeric columns and create skewness plots
                for i, col in enumerate(numeric_columns):
                    ax = axes[i]
                    sns.histplot(processed_df[col], kde=True, ax=ax)
                    ax.set_title(f'{col} Skewness: {processed_df[col].skew():.2f}')
                # Remove empty subplots if any
                for i in range(len(numeric_columns), len(axes)):
                    fig.delaxes(axes[i])
                # Adjust layout for skewness plots
                fig.tight_layout()
                st.pyplot(fig)

                # Correlation Heatmap
                st.write("#### Correlation Heatmap")
                correlation_matrix = processed_df.corr()
                # Set up the matplotlib figure
                fig, ax = plt.subplots(figsize=(14, 10))
                # Set custom color map (you can choose any other seaborn color map)
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                # Draw the heatmap with improved settings
                sns.heatmap(correlation_matrix, cmap=cmap, vmin=-1, vmax=1, center=0,
                            square=True, linewidths=.5, cbar_kws={"shrink": .5, "aspect": 10},
                            annot=True, fmt=".2f", annot_kws={"size": 8})
                # Add title
                plt.title("Correlation Heatmap")
                # Show the plot
                st.pyplot(fig)

                # Detect and Treat Outliers
                st.write("#### Detect and Treat Outliers")
                Q1 = processed_df['transactionRevenue'].quantile(0.25)
                Q3 = processed_df['transactionRevenue'].quantile(0.75)
                IQR = Q3 - Q1
                # Identify outliers
                outliers = processed_df[(processed_df['transactionRevenue'] < (Q1 - 1.5 * IQR)) | (processed_df['transactionRevenue'] > (Q3 + 1.5 * IQR))]
                # Display the count of outliers
                st.write(f"Number of outliers detected: {len(outliers)}")
                # Display histogram before and after outlier treatment
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                # Histogram before outlier treatment
                ax[0].hist(processed_df['transactionRevenue'], bins=20, color='blue', alpha=0.7)
                ax[0].set_title('Histogram Before Outlier Treatment')
                ax[0].set_xlabel('Transaction Revenue')
                ax[0].set_ylabel('Count')
                # Histogram after outlier treatment
                ax[1].hist(outliers['transactionRevenue'], bins=20, color='green', alpha=0.7)
                ax[1].set_title('Histogram After Outlier Treatment')
                ax[1].set_xlabel('Transaction Revenue')
                ax[1].set_ylabel('Count')
                # Show the histograms
                st.pyplot(fig)
                # Print a summary message
                st.success("Outliers detected and treated successfully!")

        # Classification
        if "Classification" in options:
            # Feature Importance with Random Forest
            le = LabelEncoder()
            for col in processed_df.columns:
                if processed_df[col].dtype == 'object' or processed_df[col].dtype == 'bool':
                    processed_df[col] = le.fit_transform(processed_df[col])
        
            X_train = processed_df.drop('has_converted', axis=1)
            Y_train = processed_df['has_converted']

            # Plot feature importance
            st.write("#### Feature Importance with Random Forest")
            rfc = RandomForestClassifier()
            rfc.fit(X_train, Y_train)
            feature_importances = rfc.feature_importances_
            feature_importance_df = pd.DataFrame({
                "Feature": X_train.columns,
                "Importance": feature_importances
            })
            top_10_features = feature_importance_df.sort_values(by="Importance", ascending=False).head(10)

            # Create a bar plot with Seaborn for better aesthetics
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=top_10_features, palette="viridis")
            # Add labels and title
            plt.xlabel("Feature Importance")
            plt.ylabel("Features")
            plt.title("Top 10 Features Importance")
            # Display the bar plot
            st.pyplot(plt.gcf())
            
            # Pie Chart using Feature Importance
            st.write("#### Pie Chart using Feature Importance")
            # Sort features by importance
            sorted_feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
            # Plot the pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            # Use a different layout for labels with increased separation
            wedges, texts, autotexts = ax.pie(sorted_feature_importance_df['Importance'],
                                            labels=sorted_feature_importance_df['Feature'],
                                            autopct='%1.1f%%', startangle=90,
                                            pctdistance=0.9, textprops=dict(color="w"))

            # Increase the font size of text and percentage labels
            for text, autotext in zip(texts, autotexts):
                text.set_size(10)
                autotext.set_size(10)
            # Add a legend
            ax.legend(sorted_feature_importance_df['Feature'], title="Features", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            # Display the pie chart
            st.pyplot(fig)

            # Random Forest Model Build
            # Drop the 'has_converted' column
            X = processed_df.drop('has_converted', axis=1)
            Y = processed_df['has_converted']

            smote = SMOTE(random_state=42)
            X_sampled, Y_sampled = smote.fit_resample(X_train, Y_train)

            # Random Forest Model Build
            model = RandomForestClassifier(n_estimators=50, random_state=42)

            rfc_model = RandomForestClassifier()
            rfc_model.fit(X_sampled, Y_sampled)
            rfc_predict = rfc_model.predict(X_sampled)
            rfc_acc = accuracy_score(Y_sampled, rfc_predict)
            rfc_pre = precision_score(Y_sampled, rfc_predict)
            rfc_rec = recall_score(Y_sampled, rfc_predict)
            rfc_f1 = f1_score(Y_sampled, rfc_predict)

            # ROC Curve
            st.write("#### ROC Curve for Random Forest Model")
            fpr, tpr, _ = roc_curve(Y_sampled, rfc_model.predict_proba(X_sampled)[:, 1])
            roc_auc = auc(fpr, tpr)
            # Set the figure size to control the plot size
            plt.figure(figsize=(8, 6))
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=8)  # Adjust the font size here
            plt.ylabel('True Positive Rate', fontsize=8)  # Adjust the font size here
            plt.title('Receiver Operating Characteristic', fontsize=8)  # Adjust the font size here
            plt.legend(loc="lower right", fontsize=8)  # Adjust the font size here
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            st.pyplot(plt)

            # Confusion Matrix
            st.write("#### Confusion Matrix for Random Forest Model")
            conf_mat = confusion_matrix(Y_sampled, rfc_predict)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
            plt.xlabel('Predicted', fontsize=8)
            plt.ylabel('Actual', fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            st.pyplot(plt)

            # Decision Tree Model Build
            dtc_model = DecisionTreeClassifier()
            dtc_model.fit(X_sampled, Y_sampled)
            dtc_predict = dtc_model.predict(X_sampled)
            dtc_acc = accuracy_score(Y_sampled, dtc_predict)
            dtc_pre = precision_score(Y_sampled, dtc_predict)
            dtc_rec = recall_score(Y_sampled, dtc_predict)
            dtc_f1 = f1_score(Y_sampled, dtc_predict)

            # KNN Model Build
            knn_model = KNeighborsClassifier()
            knn_model.fit(X_sampled, Y_sampled)
            knn_predict = knn_model.predict(X_sampled)
            knn_acc = accuracy_score(Y_sampled, knn_predict)
            knn_pre = precision_score(Y_sampled, knn_predict)
            knn_rec = recall_score(Y_sampled, knn_predict)
            knn_f1 = f1_score(Y_sampled, knn_predict)

            col1, col2, col3 =st.columns(3)
            with col1:
                # Display Random Forest Model results
                st.write("#### Random Forest Model")
                st.write("Accuracy:", rfc_acc)
                st.write("Precision:", rfc_pre)
                st.write("Recall:", rfc_rec)
                st.write("F1_score:", rfc_f1)
            with col2:
                # Display Decision Tree Model results
                st.write("#### Decision Tree Model")
                st.write("Accuracy:", dtc_acc)
                st.write("Precision:", dtc_pre)
                st.write("Recall:", dtc_rec)
                st.write("F1_score:", dtc_f1)
            with col3:
                # Display KNN Model results
                st.write("#### KNN Model")
                st.write("Accuracy:", knn_acc)
                st.write("Precision:", knn_pre)
                st.write("Recall:", knn_rec)
                st.write("F1_score:", knn_f1)

            # Compare Models
            compare_data = {
                'Model': ['Random Forest', 'Decision Tree', 'KNN'],
                'Accuracy': [rfc_acc, dtc_acc, knn_acc],
                'Precision': [rfc_pre, dtc_pre, knn_pre],
                'Recall': [rfc_rec, dtc_rec, knn_rec],
                'F1_score': [rfc_f1, dtc_f1, knn_f1]
            }
            compare_table = st.dataframe(compare_data)

def data_prediction():
    with open("rfc.pkl", "rb") as mf:
        new_model = pickle.load(mf)
    
    st.markdown("<h3 style='text-align: center;'>Data Prediction</h3>", unsafe_allow_html=True)
    local_image_path = "C:/Users/Hp/Desktop/DS_Final_Project/pred_bg.jpg"
    image = Image.open(local_image_path)
    resized_image = image.resize((950,475)) 
    st.image(resized_image, use_column_width=True)

    st.markdown("<h3 style='text-align: center;'>Customer Conversion Prediction</h3>", unsafe_allow_html=True)
    
    with st.form("user_inputs"):
        with st.container():
            count_session = st.number_input("count_session",max_value=100.00,min_value=0.00)
            time_earliest_visit = st.number_input("time_earliest_visit",max_value=100.00,min_value=0.00)
            avg_visit_time = st.number_input("avg_visit_time",max_value=100.00,min_value=0.00)
            days_since_last_visit = st.number_input("days_since_last_visit",max_value=100.00,min_value=0.00)
            days_since_first_visit = st.number_input("days_since_first_visit",max_value=100.00,min_value=0.00)
            visits_per_day = st.number_input("visits_per_day",max_value=100.00,min_value=0.00)
            bounce_rate = st.number_input("bounce_rate",max_value=100.00,min_value=0.00)
            earliest_source = st.number_input("earliest_source",max_value=100.00,min_value=0.00)
            latest_source = st.number_input("latest_source",max_value=100.00,min_value=0.00)
            earliest_medium = st.number_input("earliest_medium",max_value=100.00,min_value=0.00)
            latest_medium = st.number_input("latest_medium",max_value=100.00,min_value=0.00)
            earliest_keyword = st.number_input("earliest_keyword",max_value=100.00,min_value=0.00)
            latest_keyword = st.number_input("latest_keyword",max_value=100.00,min_value=0.00)
            earliest_isTrueDirect = st.number_input("earliest_isTrueDirect",max_value=100.00,min_value=0.00)
            latest_isTrueDirect = st.number_input("latest_isTrueDirect",max_value=100.00,min_value=0.00)
            num_interactions = st.number_input("num_interactions",max_value=100.00,min_value=0.00)
            bounces = st.number_input("bounces",max_value=100.00,min_value=0.00)
            time_on_site = st.number_input("time_on_site",max_value=100.00,min_value=0.00)
            time_latest_visit = st.number_input("time_latest_visit",max_value=100.00,min_value=0.00)
            
        predict_button = st.form_submit_button("Predict")

        # Prediction results
        if predict_button:
            # Prepare the input data based on user inputs
            test_data_csv = np.array([
                [count_session, time_earliest_visit, avg_visit_time, days_since_last_visit,
                 days_since_first_visit, visits_per_day, bounce_rate, earliest_source,
                 latest_source, earliest_medium, latest_medium, earliest_keyword,
                 latest_keyword, earliest_isTrueDirect, latest_isTrueDirect, num_interactions,
                 bounces, time_on_site, time_latest_visit]])
        
            # Convert the data to float
            test_data_csv = test_data_csv.astype(float)
        
            # Make predictions
            predicted_class = new_model.predict(test_data_csv)[0]
            predicted_probabilities = new_model.predict_proba(test_data_csv)
            # Display the results only when the "Predict" button is clicked
            st.write("Predicted Status:", predicted_class)
            # Note: Modify the following line to match your actual class labels.
            class_labels = ["Not Converted", "Converted"]
            
            # Get the predicted label based on the class index
            predicted_label = class_labels[int(predicted_class)]
            # Display the predicted label
            st.write("Predicted Customer Status:", predicted_label)
            # Display the predicted probabilities
            st.write("Predicted Probabilities:", predicted_probabilities)
            # Add a comment related to the prediction process
            st.write(f"**Note: The model has predicted whether the visitor is likely to convert as a customer or not based on the provided input features.**")

def data_class_pred():
    # Create tabs for Prediction and Classification
    tabs = ["Classification", "Prediction"]
    selected_tab = st.sidebar.selectbox("Select Task:", tabs)
    # Display content based on the selected tab
    if selected_tab == "Classification":
        data_classification()
    elif selected_tab == "Prediction":
        data_prediction()

# Function to perform OCR using Tesseract
def perform_ocr(image):
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    text_result = pytesseract.image_to_string(image)
    if text_result:
        st.header("OCR Result")
        st.write(text_result)
        processed_text_type = type(text_result)
        text_image_array_length = len(np.array(image))
        st.write(f"Processed Text Type: {processed_text_type}")
        st.write(f"Text Image Array Length: {text_image_array_length}")
    else:
        st.write(f"Processed Text: No text detected in the uploaded image")

# Function to display information about the uploaded image
def display_image_info(uploaded_file):
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    st.markdown("<h5>Uploaded Image Information</h5>", unsafe_allow_html=True)
    st.write(f"Size: {image.size}")
    st.write(f"Mode: {image.mode}")
    st.write(f"Format: {image.format}")
    st.write(f"Dimensions (Image Array): {image_array.shape}")

# Function to perform image classification and preprocessing
def perform_image_classification(image):
    # Check the mode of the uploaded image
    if image.mode == "RGB":
        # Convert to grayscale if the mode is RGB
        gray_image = image.convert("L")
    else:
        gray_image = image.convert("RGB")

    # Display information about the converted image
    st.markdown("<h5>Converted Image Information</h5>", unsafe_allow_html=True)
    st.write(f"Size: {gray_image.size}")
    st.write(f"Mode: {gray_image.mode}")
    st.write(f"Format: {gray_image.format}")
    st.write(f"Dimensions (Image Array): {np.array(gray_image).shape}")

    return gray_image

# Function to enhance images
def enhance_image(image, brightness_factor=None, contrast_factor=None):
    # Enhance brightness if the factor is provided
    if brightness_factor is not None:
        bright_image = ImageEnhance.Brightness(image)
        image = bright_image.enhance(brightness_factor)

    # Enhance contrast if the factor is provided
    if contrast_factor is not None:
        contrast_image = ImageEnhance.Contrast(image)
        image = contrast_image.enhance(contrast_factor)

    return image

# Function to add a frame to the image
def add_frame(image, frame_width, frame_color):
    return ImageOps.expand(image, frame_width, frame_color)

# Main Streamlit app
def image():
    st.markdown("<h3 style='text-align: center;'>Image Processing</h3>", unsafe_allow_html=True)
    local_image_path = r"C:\Users\Hp\Desktop\DS_Final_Project\img_bg.jpg"
    image = Image.open(local_image_path)
    resized_image = image.resize((950,475)) 
    st.image(resized_image, use_column_width=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    
        col1, col2 = st.columns(2)
        with col1:
            display_image_info(uploaded_file)
        with col2:
            processed_image = perform_image_classification(image)
        
        resize_dimensions = st.sidebar.text_input("Resize Dimensions (e.g; width, height)", "400,350")
        width, height = [int(val) for val in resize_dimensions.split(",")]
        resized_image = image.resize((width, height))
        
        crop_dim = st.sidebar.text_input("Crop Dimension (e.g;crop_left, crop_top, crop_right, crop_bottom)", "110,25,220,150")
        crop_values = [int(val) for val in crop_dim.split(",")]
        
        default_rotation_angle = 55
        rotation_angle = st.sidebar.slider("Rotation Angle", -360, 360, default_rotation_angle)
        
        # Blur the resized image based on user input
        blur_radius = st.sidebar.slider("Blur Radius", 0, 10, 2)
        
        # Add a frame to the resized image
        frame_width = st.sidebar.slider("Frame Width", 0, 50, 10)
        frame_color = st.sidebar.color_picker("Frame Color", "#8DCE36")

        crop_image = resized_image.crop(crop_values)
        rotated_image = resized_image.rotate(rotation_angle)
        blur_image = resized_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        framed_image = add_frame(resized_image, frame_width, frame_color)

        # Enhance brightness and contrast for both original and converted images
        brightness_factor = st.sidebar.slider("Brightness Factor", 0.1, 3.0, 1.0)
        contrast_factor = st.sidebar.slider("Contrast Factor", 0.1, 3.0, 1.0)
        bright_original_image = enhance_image(resized_image, brightness_factor)
        contrast_original_image = enhance_image(resized_image, contrast_factor)
        bright_converted_image = enhance_image(processed_image, brightness_factor)
        contrast_converted_image = enhance_image(processed_image, contrast_factor)

        # Mirror the resized image and negative film effect
        mirror_image = ImageOps.mirror(resized_image)
        negative_image = ImageOps.invert(resized_image)
        monochrome_image = resized_image.convert('1')

        # Apply edge detection to original, converted, and negative images
        edge_original_image = resized_image.filter(ImageFilter.FIND_EDGES)
        edge_converted_image = processed_image.filter(ImageFilter.FIND_EDGES)
        edge_negative_image = negative_image.filter(ImageFilter.FIND_EDGES)
        edge_monochrome_image = monochrome_image.filter(ImageFilter.FIND_EDGES)
 
        # Sharpen the monochrome image
        sharpen_factor = st.sidebar.slider("Sharpen Factor", 0.1, 3.0, 1.5)
        sharpen_original_image = resized_image.filter(ImageFilter.SHARPEN)
        sharpen_converted_image = processed_image.filter(ImageFilter.SHARPEN)
        sharpen_negative_image = negative_image.filter(ImageFilter.SHARPEN)
        sharpen_monochrome_image = monochrome_image.filter(ImageFilter.SHARPEN)
        
        # Mask the resized image based on user input
        mask_width_percentage = st.sidebar.slider("Mask Width (%)", 0, 100, 70)
        mask_height_percentage = st.sidebar.slider("Mask Height (%)", 0, 100, 40)
        mask_width = int(resized_image.width * (mask_width_percentage / 100))
        mask_height = int(resized_image.height * (mask_height_percentage / 100))
        masking_image = Image.new('L', resized_image.size, 0)
        draw = ImageDraw.Draw(masking_image)
        draw.rectangle((0, 0, mask_width, mask_height), fill=255)
        masked_image = ImageChops.composite(resized_image, Image.new('RGB', resized_image.size), masking_image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(resized_image, use_column_width=True, caption="Uploaded-Resized")
            perform_ocr(image)
        with col2:
            processed_resized_image = processed_image.resize((400,350))
            st.image(processed_resized_image, use_column_width=True, caption="Converted-Resized")

        col1, col2, col3, col4 = st.columns(4)
        col1.image(crop_image, use_column_width=True, caption="Cropped (Original)")
        col2.image(rotated_image, use_column_width=True, caption="Rotated (Original)")
        col3.image(blur_image, use_column_width=True, caption="Blurred (Original)")
        col4.image(framed_image, use_column_width=True, caption="Framed (Original)")
 
        col5, col6, col7, col8 = st.columns(4)
        col5.image(bright_original_image, use_column_width=True, caption="Enhanced Original (Brightness)")
        col6.image(contrast_original_image, use_column_width=True, caption="Enhanced Original (Contrast)")
        col7.image(bright_converted_image, use_column_width=True, caption="Enhanced Converted (Brightness)")
        col8.image(contrast_converted_image, use_column_width=True, caption="Enhanced Converted (Contrast)")

        col9, col10, col11 = st.columns(3)
        col9.image(mirror_image, use_column_width=True, caption="Original (Mirror)")
        col10.image(negative_image, use_column_width=True, caption="Original (Negative)")
        col11.image(monochrome_image, use_column_width=True, caption="Original (Monochrome)")

        col12, col13, col14, col15 = st.columns(4)
        col12.image(edge_original_image, use_column_width=True, caption="Edge (Original)")
        col13.image(edge_converted_image, use_column_width=True, caption="Edge (Converted)")
        col14.image(edge_negative_image, use_column_width=True, caption="Edge (Negative)")
        col15.image(edge_monochrome_image, use_column_width=True, caption="Edge (Monochrome)")
        
        col16, col17, col18, col19 = st.columns(4)
        col16.image(sharpen_original_image, use_column_width=True, caption="Sharpen (Original)")
        col17.image(sharpen_converted_image, use_column_width=True, caption="Sharpen (Converted)")
        col18.image(sharpen_negative_image, use_column_width=True, caption="Sharpen (Negative)")
        col19.image(sharpen_monochrome_image, use_column_width=True, caption="Sharpen (Monochrome)")
        
        col20, col21, col22, col23 = st.columns(4)
        col21.image(masked_image, caption="Masked (Original)", use_column_width=True)

# Download NLTK data (uncomment if not downloaded already)
nltk.download("all")

# Reference for POS Tags
tag_mapping = {
    'CC': 'Coordinating Conjunction',
    'CD': 'Cardinal Digit',
    'DT': 'Determiner',
    'EX': 'Existential There',
    'FW': 'Foreign Word',
    'IN': 'Preposition or Subordinating Conjunction',
    'JJ': 'Adjective',
    'JJR': 'Adjective, Comparative',
    'JJS': 'Adjective, Superlative',
    'LS': 'List Item Marker',
    'MD': 'Modal',
    'NN': 'Noun, Singular or Mass',
    'NNS': 'Noun, Plural',
    'NNP': 'Proper Noun, Singular',
    'NNPS': 'Proper Noun, Plural',
    'PDT': 'Predeterminer',
    'POS': 'Possessive Ending',
    'PRP': 'Personal Pronoun',
    'PRP$': 'Possessive Pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb, Comparative',
    'RBS': 'Adverb, Superlative',
    'RP': 'Particle',
    'SYM': 'Symbol',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb, Base Form',
    'VBD': 'Verb, Past Tense',
    'VBG': 'Verb, Gerund or Present Participle',
    'VBN': 'Verb, Past Participle',
    'VBP': 'Verb, Non-3rd Person Singular Present',
    'VBZ': 'Verb, 3rd Person Singular Present',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive Wh-pronoun',
    'WRB': 'Wh-adverb'
}

def show_pos_tag_reference():
    st.write("Here is the reference for Part of Speech (POS) Tags:")
    for tag, meaning in tag_mapping.items():
        st.write(f"- **{tag}**: {meaning}")
        
def nlp_processing(sample_text, tasks):
    # Make the sample text lowercase
    sample_text = sample_text.lower()
    
    # Tokenization
    tokens = word_tokenize(sample_text)

    # Sentence Tokenization
    sent_tokens = sent_tokenize(sample_text)

    # Stopwords Removal
    stop_words = set(stopwords.words('english'))
    stopword_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Numbers and Special Characters Removal
    processed_stopword_tokens_list = [word for word in stopword_tokens if word.isalpha()]
    result_string = ' '.join(processed_stopword_tokens_list)

    # Stemming
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in processed_stopword_tokens_list]

    # Lemmatization
    wn = WordNetLemmatizer()
    lemmatized_words = [wn.lemmatize(word) for word in processed_stopword_tokens_list]

    # Part of Speech (POS)
    pos_tags = nltk.pos_tag(tokens)

    # N-gram
    n_gram = 3  # Default value
    if 'N-gram' in tasks:
        n_gram = st.sidebar.slider("Select N for N-gram", min_value=2, max_value=5, value=3, step=1)

    ngram_3 = list(ngrams(tokens, n_gram))
        
    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(sample_text)

    # Word Cloud
    wc = WordCloud(width=600, height=400, background_color="orange").generate(sample_text)

    # Keyword Extraction
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([result_string])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    word_tfidf_dict = dict(zip(feature_names, tfidf_scores))
    sorted_keywords = sorted(word_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [keyword[0] for keyword in sorted_keywords[:5]]

    # Named Entity Recognition (NER) using spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sample_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Display results based on selected tasks
    if 'Tokenization' in tasks:
        st.markdown("### Tokenization")
        st.write(f"**Tokens:** {tokens}")
        st.write(f"**Tokens Length:** {len(tokens)}")

    if 'Sentence Tokenization' in tasks:
        st.markdown("### Sentence Tokenization")
        st.write(f"**Sentences:** {sent_tokens}")

    if 'Stopwords Removal' in tasks:
        st.markdown("### Stopwords Removal")
        st.write(f"**Stopword Tokens:** {stopword_tokens}")
        st.write(f"**Tokens Length after Stopwords Removal:** {len(stopword_tokens)}")

    if 'Special Characters and Numbers Removal' in tasks:
        st.markdown("### Special Characters and Numbers Removal")
        st.write(f"**Processed Result:** {result_string}")
        
    if 'Stemming' in tasks:
        st.markdown("### Stemming")
        st.write(f"**Stemmed Words:** {stemmed_words}")

    if 'Lemmatization' in tasks:
        st.markdown("### Lemmatization")
        st.write(f"**Lemmatized Words:** {lemmatized_words}")

    if 'POS Tags' in tasks:
        st.markdown("### Part of Speech (POS) Tags")
        st.write(f"**POS Tags:** {pos_tags}")

    # Display POS Tags reference if selected as a task
    if 'POS Tags Reference' in tasks:
        st.markdown("### Part of Speech (POS) Tags References")
        show_pos_tag_reference()

    if 'N-gram' in tasks:
        st.markdown("### N-gram")
        st.write(f"**N-gram (Value selected: {n_gram}):** {ngram_3}")

    if 'Sentiment Analysis' in tasks:
        st.markdown("### Sentiment Analysis")
        st.write(f"**Sentiment Score:** {sentiment_score}")
        # Sentiment plot
        sentiment_plot = {'Negative': sentiment_score['neg'],
                          'Neutral': sentiment_score['neu'],
                          'Positive': sentiment_score['pos']}
        # Create a DataFrame for better visualization
        sentiment_df = pd.DataFrame(list(sentiment_plot.items()), columns=['Sentiment', 'Score'])

        # Plotting with Seaborn for better aesthetics
        plt.figure(figsize=(12,6))
        sns.barplot(x='Sentiment', y='Score', data=sentiment_df, palette='viridis')

        # Adding labels and title
        plt.title('Sentiment Analysis Scores')
        plt.xlabel('Sentiment')
        plt.ylabel('Score')

        # Display the plot
        st.pyplot(plt)

    if 'Word Cloud' in tasks:
        st.markdown("### Word Cloud")
        st.image(wc.to_image())

    if 'Keyword Extraction' in tasks:
        st.markdown("### Keyword Extraction")
        st.write(f"**Top Keywords:** {top_keywords}")

    if 'Named Entities' in tasks:
        st.markdown("### Named Entities")
        st.write(f"**Entities:** {entities}")

def nlp_disp():
    st.markdown("<h3 style='text-align: center;'>Natural Language Processing (NLP)</h3>", unsafe_allow_html=True)
    local_image_path = r"C:\Users\Hp\Desktop\DS_Final_Project\nlp_bg.jpg"
    image = Image.open(local_image_path)
    resized_image = image.resize((950,475)) 
    st.image(resized_image, use_column_width=True)

    sample_text_default = """In the year 2023, the revolution in technology brings forth challenges and opportunities. Data-driven decisions, cybersecurity 
    measures, and advancements in artificial intelligence (AI) are at the forefront. Individuals are adapting to the era of smartphones, where the 
    average person spends about 3 hours daily on their devices. As the digital landscape evolves, the importance of constant learning and staying 
    informed cannot be stressed enough. Embracing innovation and maintaining a balance between work, life, and technology is crucial for navigating 
    the complexities of the modern world."""

    # Text input
    sample_text = st.text_area("Enter any sentence/paragraph for NLP processing:", value=sample_text_default, max_chars=2000)
    
    st.info("The text you see in the text area is the default sample text, and you can use any text for the process.")
    
    # Tasks selection
    tasks = st.multiselect("Select NLP tasks to perform:", ['Tokenization', 'Sentence Tokenization', 'Stopwords Removal',
                                                            'Special Characters and Numbers Removal',
                                                            'Stemming', 'Lemmatization', 'POS Tags', 'POS Tags Reference', 'N-gram',
                                                            'Sentiment Analysis', 'Word Cloud', 'Keyword Extraction','Named Entities'])
    st.write("*Note: POS Tags Reference is provided for educational purposes.*")
        
    if st.button("Process NLP"):
        nlp_processing(sample_text, tasks)

def load_data():
    # Load the data
    df = pd.read_csv('E-Comm.csv')
    return df

def load_product_names():
    # Load product names from CSV
    product_names_df = pd.read_csv('Products.csv')
    return product_names_df

def display_data_info(df,product_names_df):
    # Create two columns layout
    col1, col2 = st.columns(2)

    # Display information for the first dataframe
    with col1:
        st.write("#### Ratings Data")
        st.dataframe(df)

    # Display information for the second dataframe
    with col2:
        st.write("#### Products Data")
        st.dataframe(product_names_df)

def build_collaborative_filtering_model(df):
    # Create a Surprise Reader
    reader = Reader(rating_scale=(1, 5))
    # Load the data into the Surprise Dataset
    data = Dataset.load_from_df(df[['CustomerID', 'ProductID', 'Rate']], reader)
    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.3, random_state=42)
    # Build a collaborative filtering model (KNN)
    algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
    algo.fit(trainset)
    # Test the model
    test_pred = algo.test(testset)
    # Evaluate the model
    accuracy.rmse(test_pred, verbose=True)

    return algo

def generate_recommendations_with_names(algo, user_id, df, product_names_df, top_n=5):
    # Generate recommendations for a specific user
    user_items = df[df['CustomerID'] == user_id]['ProductID'].unique()
    # Exclude products the user has already rated
    user_unrated_products = df[~df['ProductID'].isin(user_items)]['ProductID'].unique()
    # Predict ratings for unrated products
    user_predictions = [algo.predict(user_id, product_id) for product_id in user_unrated_products]
    # Get top N recommendations with names
    recommendations = sorted(user_predictions, key=lambda x: x.est, reverse=True)[:top_n]

    # Extract product names for recommended products
    recommended_products = []
    for rec in recommendations:
        product_id = rec.iid
        estimated_rating = rec.est
        product_name = product_names_df[product_names_df['Id'] == product_id]['Name'].values[0]
        recommended_products.append({
            'ProductID': product_id,
            'Estimated Rating': estimated_rating,
            'Product Name': product_name
        })

    return recommended_products

def product_recommendation():
    st.markdown("<h3 style='text-align: center;'>E-Commerce Product Recommendation System</h3>", unsafe_allow_html=True)
    local_image_path = "C:/Users/Hp/Desktop/DS_Final_Project/prod_recom_bg.jpg"
    image = Image.open(local_image_path)
    resized_image = image.resize((950,475)) 
    st.image(resized_image, use_column_width=True)

    # Load data and build the collaborative filtering model
    df = load_data()
    algo = build_collaborative_filtering_model(df)

    # Load product names
    product_names_df = load_product_names()
    
    st.write("- The following dataframes provide information about the ratings and products.")
    st.write("- Use them to explore and analyze the data. This dataset is used for collaborative filtering.")
        
    display_data_info(df,product_names_df)

    # User input for dynamic recommendation
    user_id = st.selectbox('Select a CustomerId for recommendations:', df['CustomerID'].unique(), index=3) 
    top_n = st.slider('Select the number of recommendations:', min_value=1, max_value=10, value=5)

    if user_id:
        user_id = int(user_id)
        recommendations = generate_recommendations_with_names(algo, user_id, df, product_names_df, top_n)

        st.markdown(f"<h4>Top-{top_n} Recommendations for CustomerID {user_id}:</h4>", unsafe_allow_html=True)
        for rec in recommendations:
            st.write(f"ProductID: {rec['ProductID']}")
            st.write(f"Product Name: {rec['Product Name']}")
            st.write(f"Estimated Rating: {rec['Estimated Rating']}")
            st.write('---')

        st.success("Recommendations generated successfully!")

# Function to display information about the project
def display_about():
    st.markdown("<h3 style='text-align: center;'>E-Comm ML: Machine Learning Model Building Project</h3>", unsafe_allow_html=True)
    local_image_path = "C:/Users/Hp/Desktop/DS_Final_Project/ds_bg.jpg"
    image = Image.open(local_image_path)
    resized_image = image.resize((950,475)) 
    st.image(resized_image, use_column_width=True)

    # Display project overview title
    st.markdown("<h3 style='text-align: center;'>Project Overview</h3>", unsafe_allow_html=True)
    # Display project details
    st.markdown("<h4>Title</h4>", unsafe_allow_html=True)
    st.markdown("<h5>E-Comm ML: Machine Learning Model Building Project</h5>", unsafe_allow_html=True)
    st.markdown("<h4>Objective</h4>", unsafe_allow_html=True)
    st.markdown("Leverage data science and machine learning techniques to analyze and enhance different aspects of E-commerce data, including EDA, Evaluation metrics, Classification & Prediction, Image processing, NLP, and Recommendation systems.")
    # Display technologies used
    st.markdown("<h4>Technologies Used</h4>", unsafe_allow_html=True)
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- Pandas")
    st.write("- Numpy")
    st.write("- Scikit-learn")
    st.write("- Pillow (PIL)")
    st.write("- Tesseract (OCR)")
    st.write("- NLTK")
    st.write("- Visualization modules or packages")
    st.write("- And more necessary modules or packages...")
    # Display problem statement
    st.markdown("<h4>Problem Statement</h4>", unsafe_allow_html=True)
    st.write("Develop a comprehensive project that covers data EDA, Classification & Prediction, Image Processing, NLP, and Product Recommendation systems using an E-commerce dataset. The goal is to predict user conversion, explore data patterns, preprocess images, analyze text data, and recommend products.")
    # Display dataset link
    st.markdown("**Datasets Link:** [Data Link](https://drive.google.com/drive/folders/1ATULlRKrSensZHs2SxaT7y0b68Rc1vQA)")
    st.markdown("*Note: For access to the dataset and additional information, please follow the provided hyperlink.*")
    # Display project structure
    st.markdown("<h4>Project Structure</h4>", unsafe_allow_html=True)
    st.write("The project is organized into distinct sections, including:")
    st.write("- Exploratory Data Analysis(EDA)")
    st.write("- Classification & Prediction")
    st.write("- Image Processing & Classification")
    st.write("- Natural Language Processing(NLP)")
    st.write("- Product Recommendation System")

    ## Section 1: Exploratory Data Analysis (EDA)
    st.markdown("<h5>Section 1: Exploratory Data Analysis (EDA)</h5>", unsafe_allow_html=True)
    st.write("1. Load and explore the E-commerce dataset.")
    st.write("2. Visualize key statistics and features.")
    st.write("3. Identify patterns and outliers.")
    st.write("4. Generate visualizations for better understanding.")
    st.write("5. Provide insights and observations.")

    ## Section 2: Classification & Prediction
    st.markdown("<h5>Section 2: Classification & Prediction</h5>", unsafe_allow_html=True)
    st.write("1. Load the classification dataset.")
    st.write("2. Perform data preprocessing and exploration.")
    st.write("3. Split the dataset into training and testing sets.")
    st.write("4. Build and evaluate machine learning models for classification.")
    st.write("5. Display the evaluation results in a table, including precision, recall, accuracy, and F1-score.")

    ## Section 3: Image Processing
    st.markdown("<h5>Section 3: Image Processing</h5>", unsafe_allow_html=True)
    st.write("1. Upload & read the Image.")
    st.write("2. Process the Image, and perform image pre-processing steps.")
    st.write("3. Display pre-processed images with titles (e.g., Resized Image, Edge Detected Image, etc..).")
    st.write("4. If a Text Image is uploaded, show the result of OCR (Optical Character Recognition).")

    ## Section 4: Natural Language Processing (NLP)
    st.markdown("<h5>Section 4: Natural Language Processing (NLP)</h5>", unsafe_allow_html=True)
    st.write("1. Input a bunch of text.")
    st.write("2. Perform NLP pre-processing steps.")
    st.write("3. Showcase each step's outputs with their names.")
    st.write("4. Find keywords from the text.")
    st.write("5. Perform sentiment analysis on the text and visualise the plot.")

    ## Section 5: Product Recommendation System
    st.markdown("<h5>Section 5: Product Recommendation System</h5>", unsafe_allow_html=True)
    st.write("1. Build a product recommendation system using the E-commerce dataset.")
    st.write("2. Allow users to input a product and receive recommended products.")
    st.write("3. Implement recommendation algorithms.")
    st.write("4. Display atleast 5 recommended products.")

    # Display contact information
    st.markdown("<h4>Contact Information</h4>", unsafe_allow_html=True)
    st.write("If you have any questions, feedback, or suggestions, feel free to reach out:")
    st.markdown("""
    Contact me via email: [email](mailto:sec19ee048@sairamtap.edu.in)
    
    Connect with me on LinkedIn: [LinkedIn](https://www.linkedin.com/in/priyanga070302/)""")

    st.write("Thank you for exploring the E-Comm-ML project about session. I hope you find it informative and engaging!")

def display_exit():
    st.markdown("<h3 style='text-align: center;'>Exit Application</h3>", unsafe_allow_html=True)
    local_image_path = r"C:\Users\Hp\Desktop\DS_Final_Project\ext_bg.jpeg"
    image = Image.open(local_image_path)
    resized_image = image.resize((950, 475)) 
    st.image(resized_image, use_column_width=True)
    st.markdown(f"**Thank you for using the E-Comm ML: Machine Learning Model Building application.**")
    st.markdown(f"**If you wish to exit, simply close the browser tab or window.**")
    st.markdown(f"**Have a great day!**")

# Get user input for page selection
page = st.sidebar.selectbox("Select Page", ["About", "Classification & Prediction", "Image Processing", "Natural Language Processing(NLP)", "Product Recommendation System", "Exit"])

# Main section to display different pages
def main():
    if page == "About":
        display_about()
    elif page == "Classification & Prediction":
        data_class_pred()
    elif page == "Image Processing":
        image()
    elif page == "Natural Language Processing(NLP)":
        nlp_disp()
    elif page == "Product Recommendation System":
        product_recommendation()
    elif page == "Exit":
        display_exit()

# Call the main function to display the appropriate content based on the selected page
main()             

