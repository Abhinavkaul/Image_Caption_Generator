import streamlit as st
from transformers import VisionEncoderDecoderModel,ViTImageProcessor, AutoTokenizer
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.to("cpu")

def predict(image):
    images = []
    for i in image:
        i = Image.open(i)
        images.append(i)
        
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values 
    output = model.generate(pixel_values.to("cpu"), num_beams = 5)
    preds = tokenizer.batch_decode(output, skip_special_tokens=True)
    return preds[0]

st.title(":blue[Caption Generator]")
uploaded_img = st.file_uploader("Choose a file",type=['png','jpg','jpeg'],accept_multiple_files=False)

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    st.image(image, caption='Processed image', use_column_width=True)
    response = predict([uploaded_img])
    st.header(":blue[Caption: ]")
    st.subheader(response)
