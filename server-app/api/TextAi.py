import random
from math import radians, sin, cos, sqrt, atan2
import requests
import torch
from bs4 import BeautifulSoup
from datasets import load_dataset
from flask import Flask, jsonify, request  # type: ignore
from flask_cors import CORS
from PIL import Image  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer
import json
import openai
app = Flask(__name__)
CORS(app)


dataset = load_dataset("Mostafijur/Skin_disease_classify_data")
dataset1 = load_dataset("brucewayne0459/Skin_diseases_and_care")
device = torch.device('cpu')
classes = {0: 'Acne and Rosacea', 1: 'Actinic Keratosis Basal Cell Carcinoma', 
           2: 'Nail Fungus', 3: 'Psoriasis Lichen Planus', 4: 'Seborrheic Keratoses', 
           5: 'Tinea Ringworm Candidiasis', 6: 'Warts Molluscum'}

tokenizer1 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-detection")
model1 = AutoModel.from_pretrained("Unmeshraj/skin-disease-detection")
tokenizer2 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-treatment-plan")
model2 = AutoModel.from_pretrained("Unmeshraj/skin-disease-treatment-plan")
image_model = models.resnet18(pretrained=False)
image_model.fc = torch.nn.Linear(image_model.fc.in_features, len(classes))
image_model.load_state_dict(torch.load("./model.pth", map_location=device))
image_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

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

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return round(distance, 2)
def get_coordinates_from_address(address):
    """Get coordinates from address using Nominatim OpenStreetMap API"""
    try:
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": address,
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code == 200 and response.json():
            location = response.json()[0]
            return float(location["lat"]), float(location["lon"])
        return None, None
    except Exception as e:
        print(f"Error getting coordinates: {e}")
        return None, None
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

def predict_image(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = image_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]
def fetchDoctors(user_location, query, mode, backupQuery, backupMode, locality, city):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }

    def is_doctor_relevant(doctor_info, query):
        """Check if doctor is relevant to the query based on specializations and conditions treated"""
        query_terms = set(query.lower().split())
        
        # Check specializations
        specializations = doctor_info.get("specializations", "").lower()
        if any(term in specializations for term in query_terms):
            return True
            
        # Check conditions treated
        conditions = doctor_info.get("conditions_treated", "").lower()
        if any(term in conditions for term in query_terms):
            return True
            
        return False

    def fetch_from_url(location, search_query):
        """Helper function to fetch doctors from a specific location with query"""
        # Try disease-specific URL first
        search_url = f"https://www.practo.com/{location}/treatment-for-{search_query}"
        backup_url = f"https://www.practo.com/{location}/dermatologist"
        
        urls_to_try = [search_url, backup_url]
        doctor_data = []
        
        for url in urls_to_try:
            print(f"Attempting to fetch from: {url}")
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.text, "html.parser")
                
                for anchor in soup.find_all("a", href=True, class_=False):
                    if "/doctor/" in anchor["href"] and anchor.find("h2", class_="doctor-name"):
                        name = anchor.find("h2", class_="doctor-name").get_text(strip=True)
                        link = "https://www.practo.com" + anchor["href"]
                        doctor_data.append({"name": name, "link": link})
                
                if doctor_data:
                    break  # Stop if we found doctors
                    
            except Exception as e:
                print(f"Error fetching from {url}: {str(e)}")
                continue
                
        return doctor_data

    # Format query for URL
    formatted_query = query.lower().replace(" ", "-")
    
    # Try fetching doctors from city first
    doctor_data = fetch_from_url(city, formatted_query)
    
    # If no doctors found in city, try locality
    if not doctor_data and locality:
        print(f"No doctors found in {city}, trying locality: {locality}")
        doctor_data = fetch_from_url(locality, formatted_query)
    
    # If still no doctors found, return empty array
    if not doctor_data:
        print(f"No doctors found in either {city} or {locality}")
        return []

    doctors_info = []
    relevant_doctors = []
    
    # Process doctors and gather detailed information
    for doctor in doctor_data[:10]:  # Increased limit to find more relevant doctors
        if doctor["name"] == "Unknown":
            continue

        try:
            profile_response = requests.get(doctor["link"], headers=headers)
            if profile_response.status_code != 200:
                continue

            profile_soup = BeautifulSoup(profile_response.text, "html.parser")
            
            # Extract qualifications
            qualifications = profile_soup.find("p", class_="c-profile__details", 
                                            attrs={"data-qa-id": "doctor-qualifications"})
            
            # Extract specializations
            specializations_div = profile_soup.find("div", class_="c-profile__details", 
                                                  attrs={"data-qa-id": "doctor-specializations"})
            specializations = (
                ", ".join(
                    span.get_text(strip=True)
                    for span in specializations_div.find_all("span", class_="u-d-inlineblock u-spacer--right-v-thin")
                )
                if specializations_div else "Dermatologist"
            )
            
            # Extract experience
            experience_h2 = profile_soup.find("h2", string=lambda text: text and "Years Experience" in text)
            experience = (
                experience_h2.get_text(strip=True).replace("\xa0", " ")
                if experience_h2 else "Experience not available"
            )
            
            # Extract clinics
            clinics = profile_soup.find_all("p", class_="c-profile--clinic__address")
            
            # Extract conditions treated (useful for relevance checking)
            conditions_div = profile_soup.find("div", string=lambda text: text and "Conditions Treated" in text)
            conditions_treated = ""
            if conditions_div and conditions_div.find_next("div"):
                conditions_treated = conditions_div.find_next("div").get_text(strip=True)

            doctor_info = {
                "name": doctor['name'],
                "link": doctor['link'],
                "qualifications": qualifications.get_text(strip=True) if qualifications else "MBBS, Dermatologist",
                "specializations": specializations,
                "experience": experience,
                "conditions_treated": conditions_treated,
                "clinics": [clinic.get_text(strip=True) for clinic in clinics] if clinics else ["Address not available"]
            }

            # Check if doctor is relevant to the query
            if is_doctor_relevant(doctor_info, query):
                relevant_doctors.append(doctor_info)
            else:
                doctors_info.append(doctor_info)

        except Exception as e:
            print(f"Error processing doctor {doctor['name']}: {str(e)}")
            continue

    # Return relevant doctors first, then others, limited to 3 total
    return (relevant_doctors + doctors_info)[:3]
def talk_to_chatBot(query):
    client = openai.OpenAI(
        base_url="https://mistral-7b.lepton.run/api/v1/",
        api_key="BTmCPY2Xbr1vZ9jhRAqafqzLLjR3KzTL"
    )
        
    response = client.chat.completions.create(
        model="mistral-7b",
        messages=[{
            "role": "user",
            "content": f"""You are a highly skilled medical expert specializing in evidence-based treatment planning. Respond to this query: '{query}'. 
            Ask the patient only about  symptoms they may be experiencing. Do not ask about the duration, medical history, or other details—just focus on gathering more information about their symptoms in a brief and concise manner."""
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
    response = client.chat.completions.create(
  model="mistral-7b",
  messages=[{"role": "user", "content": f"""You are a highly skilled medical expert specializing in evidence-based treatment planning. Based on the provided disease name {disease}, generate a complete and structured response that includes the following:
    Disease Overview: Provide a brief overview of the disease, including its common causes, symptoms, and progression.
    Symptoms: List the most common and relevant symptoms associated with the disease.
    Treatment Plan: Create a step-by-step treatment plan, including:
    Medications (with dosage and duration, if applicable).
    Therapies (physical, occupational, psychological, etc.).
    Lifestyle modifications.
    Follow-up or monitoring recommendations.
    Precautions: Include specific precautions the patient should take to prevent complications or recurrence.
    Additional Notes: Add any relevant advice, warnings, or general information to enhance understanding and compliance.
    Prognosis: Provide an outlook or expected outcome for the condition with proper treatment and care. replace * with space """}],
  max_tokens=1000,
  stream=True ) 
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
        similar_disease = find_similar_disease(input_query)
        if similar_disease is None:
            my_reply = talk_to_chatBot(input_query)
            return jsonify({
                'disease': "",
                'treatment': my_reply
            })
        city = user_location.get('city', "Bangalore") if user_location else "Bengalore"
        print(city)
        treatment_plan = formatDesc(similar_disease)
        locality = user_location.get('locality', "Indiranagar") if user_location else "Indiranagar"
        print(locality)
        query = similar_disease.replace(" ", "%20")
        mode = "symptom"
        backupQuery = "dermatologist"
        backupMode = "service"

        doctor_info = fetchDoctors(user_location, query, mode, backupQuery, backupMode, locality,city)
        
        # Ensure we always return a valid response
        response = {
            'disease': similar_disease,
            'treatment': treatment_plan,
            'doctors': doctor_info if doctor_info else []  # Ensure doctors is always a list
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"Error in GenResult: {str(e)}")
        # Return a valid response even in case of error
        return jsonify({
            'disease': "Could not determine disease",
            'treatment': "Please consult a dermatologist for proper diagnosis and treatment.",
            'doctors': [{
                "name": "Dr. Default",
                "qualifications": "MBBS, MD Dermatology",
                "specializations": "Dermatologist",
                "clinics": ["Please try again or contact support for dermatologist recommendations"]
            }]
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
    # Use user's location if available, otherwise use default values
    city = user_location.get('city', "Bangalore") if user_location else "Bangalore"
    locality = user_location.get('locality', "Indiranagar") if user_location else "Indiranagar"
    
    # Construct query parameters
    query = predicted_disease.replace(" ", "%20")
    mode = "symptom"
    backupQuery = "dermatologist"
    backupMode = "service"
    
    # Fetch doctors based on the predicted disease and user's location
    doctor_info = fetchDoctors(user_location, query, mode, backupQuery, backupMode, locality, city)
    
    # Construct the response
    response = {
        'disease': predicted_disease,
        'treatment': treatment,
        'doctors': doctor_info if doctor_info else []
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,port=5001)