import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# Set custom background image using CSS
page_bg_img = '''
<style>
body {
background-image: url("https://github.com/AkarshanGupta/sms-spam-detection/blob/main/bg.jpg?raw=true");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('Potato Leaf Disease Prediction')

# Display image below the title
image_url = "https://github.com/AkarshanGupta/sms-spam-detection/blob/main/bg.jpg?raw=true"  # Replace with your image URL
st.image(image_url, caption='Potato Leaf', use_column_width=True)

# Introduction text
st.write("""
This website predicts potato leaf diseases.
""")

# Upload file button
file_uploaded = st.file_uploader('Choose an image...', type='jpg', key='file_uploader')

def main():
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result, confidence = predict_class(image)
        st.write('Prediction : {}'.format(result))
        st.write('Confidence : {}%'.format(confidence))

def predict_class(image):
    with st.spinner('Loading Model...'):
        model = keras.models.load_model('potato.h5', compile=False)

    shape = (256, 256, 3)
    test_image = image.resize((256, 256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']

    prediction = model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]
    return final_pred, confidence

if __name__ == '__main__':
    main()
