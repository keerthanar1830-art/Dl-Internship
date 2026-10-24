"""
Fashion Item Recognition - Streamlit Web Application
=====================================================
An interactive web interface for classifying fashion items
using a trained CNN model on Fashion-MNIST data.
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import FashionCNN, FashionFC
from src.predict import load_model, preprocess_image, predict

# Class names
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def main():
    st.set_page_config(page_title='Fashion Item Recognition', page_icon='👗', layout='centered')
    
    st.title('👗 Fashion Item Recognition')
    st.markdown('''
    Upload an image of a clothing or fashion item, and the trained CNN model will classify it 
    into one of 10 Fashion-MNIST categories.
    
    > **Note:** This model is trained on small 28×28 grayscale Fashion-MNIST images. 
    > Predictions on real-world photographs may be less accurate due to differences 
    > in resolution, color, background, and complexity.
    ''')
    
    # Load model
    checkpoint_path = os.path.join(project_root, 'models', 'fashion_cnn.pth')
    
    if not os.path.exists(checkpoint_path):
        st.error('❌ Model checkpoint not found! Please train the model first by running the notebook.')
        st.info('Run the Jupyter notebook at `notebooks/fashion_item_recognition.ipynb` to train and save the model.')
        return
    
    try:
        model, class_names, device = load_model(checkpoint_path)
        st.success(f'✅ Model loaded successfully! Running on: {device}')
    except Exception as e:
        st.error(f'❌ Error loading model: {e}')
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        'Upload a fashion item image',
        type=['png', 'jpg', 'jpeg'],
        help='Upload a PNG or JPEG image of a clothing/fashion item'
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Uploaded Image')
            st.image(image, use_container_width=True)
        
        # Preprocess and predict
        try:
            image_tensor = preprocess_image(image)
            result = predict(model, image_tensor, class_names, device)
            
            with col2:
                st.subheader('Prediction Result')
                st.metric('Predicted Category', result['predicted_class'])
                st.metric('Confidence', f"{result['confidence']:.1%}")
            
            # Show probability chart
            st.subheader('Class Probabilities')
            
            # Create bar chart data
            probs = result['probabilities']
            prob_values = list(probs.values())
            prob_names = list(probs.keys())
            
            # Sort by probability
            sorted_indices = np.argsort(prob_values)[::-1]
            sorted_names = [prob_names[i] for i in sorted_indices]
            sorted_values = [prob_values[i] for i in sorted_indices]
            
            # Use streamlit bar chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#2ecc71' if name == result['predicted_class'] else '#3498db' for name in sorted_names]
            bars = ax.barh(range(len(sorted_names)), sorted_values, color=colors)
            ax.set_yticks(range(len(sorted_names)))
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Probability')
            ax.set_title('Prediction Probabilities for All Classes')
            ax.invert_yaxis()
            
            for bar, val in zip(bars, sorted_values):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.1%}', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f'❌ Error during prediction: {e}')
            st.info('Make sure the uploaded image is a valid image file.')
    
    # Footer
    st.markdown('---')
    st.markdown('**SKILL NEXIS Week 3** — Fashion Item Recognition using CNNs')

if __name__ == '__main__':
    main()
