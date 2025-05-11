import random
from math import radians, sin, cos, sqrt, atan2
import requests
import torch
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup
from datasets import load_dataset
from flask import Flask, jsonify, request  # type: ignore
from flask_cors import CORS
from PIL import Image  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
# from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import openai
app = Flask(__name__)
CORS(app)

# ✅ Static dermatologist data for Laguna, PH
static_doctors_laguna = [
    {
        "name": "Dr. Raul Jr O. Ojeda",
        "qualifications": "MD, FPDS",
        "specializations": "General Dermatology",
        "experience": "San Pablo City, Laguna",
        "clinics": ["San Pablo City, Laguna"],
        "link": "https://pds.org.ph/search-dermatologist/?loc=laguna"
    },
    {
        "name": "Dr. Luella Joy A. Escueta-Alcos",
        "qualifications": "MD, FPDS",
        "specializations": "Dermatologic Surgery",
        "experience": "Los Baños, Laguna",
        "clinics": ["Los Baños, Laguna"],
        "link": "https://pds.org.ph/search-dermatologist/?loc=laguna"
    },
    {
        "name": "Dr. Kathleen May E. Alpapara",
        "qualifications": "MD",
        "specializations": "Pediatric Dermatology",
        "experience": "Calamba, Laguna",
        "clinics": ["Calamba, Laguna"],
        "link": "https://pds.org.ph/search-dermatologist/?loc=laguna"
    },
    {
        "name": "Dr. Joan Joy Patricio",
        "qualifications": "MD, DPDS",
        "specializations": "Cosmetic Dermatology",
        "experience": "San Pedro, Laguna",
        "clinics": ["San Pedro, Laguna"],
        "link": "https://pds.org.ph/search-dermatologist/?loc=laguna"
    },
    {
        "name": "Dr. Andrea Bernales Mendoza",
        "qualifications": "MD, MMHOA, FPDS, FPDS-PDS",
        "specializations": "Aesthetic and Medical Dermatology",
        "experience": "Cabuyao, Laguna",
        "clinics": ["Cabuyao, Laguna"],
        "link": "https://pds.org.ph/search-dermatologist/?loc=laguna"
    }
]

def format_doctor_list(doctors):
    return "\n".join([
        f"- {d['name']} ({d.get('specializations', '')}) — {d['clinics'][0]} — Website: {d.get('link', '')}"
        for d in doctors
    ])


dataset = load_dataset("Mostafijur/Skin_disease_classify_data")
dataset1 = load_dataset("brucewayne0459/Skin_diseases_and_care")
# device = torch.device('cpu')
classes = [
    'Tinea Faciei (Ringworm)', 'Lupus (Discoid)', 'Actinic Keratosis',
    'Contact Dermatitis', 'Acne Vulgaris', 'Psoriasis', 'Basal Cell Carcinoma',
    'Melasma', 'Seborrheic Dermatitis', 'Warts', 'Normal Skin',
    'Rosacea', 'Perioral Dermatitis'
]

tokenizer1 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-detection")
model1 = AutoModel.from_pretrained("Unmeshraj/skin-disease-detection")
tokenizer2 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-treatment-plan")
model2 = AutoModel.from_pretrained("Unmeshraj/skin-disease-treatment-plan")
# image_model = models.resnet18(pretrained=False)
# image_model.fc = torch.nn.Linear(image_model.fc.in_features, len(classes))
# image_model.load_state_dict(torch.load("./model.pth", map_location=device))
image_model = load_model("./facial_skin_model_mobilenet_MINIMAL_FINAL.h5")
# image_model.eval()

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def predict_image(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = image_model.predict(img_array)
    predicted_index = np.argmax(prediction)
    return classes[predicted_index]


queries, diseases, embeddings = [], [], []
for example in dataset['train']:
    query = example['Skin_disease_classification']['query']
    disease = example['Skin_disease_classification']['disease']
    queries.append(query)
    diseases.append(disease)
    query_embedding = embed_text(query, tokenizer1, model1)
    embeddings.append(query_embedding)

topics, information, topic_embeddings = [], [], []
for example in dataset1['train']:
    topic = example['Topic']
    info = example['Information']
    topics.append(topic)
    information.append(info)
    topic_embedding = embed_text(topic, tokenizer2, model2)
    topic_embeddings.append(topic_embedding)

def find_similar_disease(input_query):
    input_embedding = embed_text(input_query, tokenizer1, model1)
    similarities = [
        cosine_similarity(input_embedding.detach().numpy(), emb.detach().numpy())[0][0] 
        for emb in embeddings
    ]
    
    max_similarity = max(similarities)
    if max_similarity > 0.5:
        return diseases[similarities.index(max_similarity)]
    else:
        return None  

def find_treatment_plan(disease_name):
    disease_embedding = embed_text(disease_name, tokenizer2, model2)
    similarities = [
        cosine_similarity(disease_embedding.detach().numpy(), topic_emb.detach().numpy())[0][0] 
        for topic_emb in topic_embeddings
    ]
    return information[similarities.index(max(similarities))]

# def predict_image(img):
#     img_tensor = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = image_model(img_tensor)
#         _, predicted = torch.max(outputs, 1)
#     return classes[predicted.item()]

def get_relevant_doctors(query):
    query_terms = query.lower().split()
    filtered = []

    for doctor in static_doctors_laguna:
        doctor_text = f"{doctor['name']} {doctor.get('qualifications', '')} {doctor.get('clinic', '')} {doctor.get('location', '')}".lower()
        if any(term in doctor_text for term in query_terms):
            filtered.append(doctor)

    return filtered[:3] if filtered else static_doctors_laguna[:3]

def talk_to_chatBot(query):
    client = openai.OpenAI(
        base_url="https://mistral-7b.lepton.run/api/v1/",
        api_key="BTmCPY2Xbr1vZ9jhRAqafqzLLjR3KzTL"
    )

    response = client.chat.completions.create(
        model="mistral-7b",
        messages=[{
            "role": "user",
            "content": f"""You are a helpful assistant named Visage Skin AI. You help users identify facial skin problems like acne, eczema, melasma, rosacea, and fungal infections.

If the user's message is a simple greeting like "hi", "hello", or "good morning", reply with:  
**"Hello! This is Visage Skin AI. How can I help you today?"**

If the user describes symptoms, ask for more details, or say "Let me analyze that for you."

If they ask where to consult a doctor, just say:  
**"Recommended dermatologists will appear below this message based on your location."**

❌ DO NOT mention or invent names like Dr. Smith or Dr. Brown.  
❌ DO NOT recommend doctors yourself — the system will show doctor cards only when a condition is found.

Now respond to the user’s message: “{query}”
"""
        }],
        max_tokens=1000,
        stream=True
    )

    formatted_text = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            formatted_text += chunk.choices[0].delta.content

    return formatted_text.strip()


def formatDesc(disease):
    client = openai.OpenAI(
        base_url="https://mistral-7b.lepton.run/api/v1/",
        api_key="BTmCPY2Xbr1vZ9jhRAqafqzLLjR3KzTL"
    )

    doctors_text = format_doctor_list(static_doctors_laguna)

    response = client.chat.completions.create(
        model="mistral-7b",
        messages=[{
            "role": "user",
            "content": f"""You are a highly skilled AI medical assistant. You only respond to questions about facial skin conditions such as acne, melasma, rosacea, eczema, and fungal infections on the face.

            Ignore unrelated questions and kindly explain that this assistant is only for facial skin health.

            Based on the disease: {disease}, generate a structured explanation that includes:

            1. Disease Overview  
            2. Symptoms  
            3. Treatment Plan  
            4. Precautions  
            5. Prognosis  

            End your message by suggesting the user consult a dermatologist. **Do NOT mention specific doctor names, samples like 'Dr. Smith', or include any list of doctors**. Doctor cards will appear below this message.
            """

        }],
        max_tokens=1000,
        stream=True
    )

    formatted_text = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            formatted_text += chunk.choices[0].delta.content
    return formatted_text.strip()

@app.route('/api/TextAi', methods=['POST'])
def GenResult():
    data = request.get_json()
    if 'inputText' not in data:
        return jsonify({'error': 'No input text provided'}), 400

    input_query = data['inputText']
    user_location = data.get('location', None)

    try:
        # Try to find a disease match
        similar_disease = find_similar_disease(input_query)

        # 🔁 General chatbot reply (no disease detected)
        if similar_disease is None:
            my_reply = talk_to_chatBot(input_query)

            # ✅ Check if user is asking for a doctor
            is_asking_for_doctor = any(keyword in input_query.lower() for keyword in [
                "doctor", "dermatologist", "clinic", "see someone", "consult", "skin specialist"
            ])

            return jsonify({
                'disease': "",
                'treatment': my_reply,
                'doctors': static_doctors_laguna[:3] if is_asking_for_doctor else []
            })

        # ✅ Disease matched
        treatment_plan = formatDesc(similar_disease)
        return jsonify({
            'disease': similar_disease,
            'treatment': treatment_plan,
            'doctors': static_doctors_laguna[:3]
        })

    except Exception as e:
        print(f"Error in GenResult: {str(e)}")
        return jsonify({
            'disease': "Could not determine disease",
            'treatment': "Please consult a dermatologist for proper diagnosis and treatment.",
            'doctors': []
        })


@app.route('/api/ImageAi', methods=['POST'])
def image_ai():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    predicted_disease = predict_image(image)
    
    # Get location data from form data
    location_str = request.form.get('location')
    if location_str:
        try:
            user_location = json.loads(location_str)
        except:
            user_location = None
    else:
        user_location = None
    
    # Fetch treatment plan for the predicted disease
    treatment = formatDesc(predicted_disease)
    
    # ✅ Use static dermatologist list
    doctor_info = static_doctors_laguna
    
    # Construct the response
    response = {
        'disease': predicted_disease,
        'treatment': treatment,
        'doctors': doctor_info
    }
    

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,port=5001)