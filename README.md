# VisageCheckAI – Facial Skin Disease Identifier

![Project Banner](https://github.com/user-attachments/assets/ed6fbb89-02c2-4b7a-a661-1586c8df3c01)


## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage Instructions](#usage-instructions)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [Changelog](#changelog)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Introduction
**VisageCheckAI** is an AI-powered web application designed to help users identify facial skin diseases through natural language and image-based queries. It provides suggestions for remedies and recommends certified dermatologists in Laguna, Philippines.

---

## Project Overview
This project serves as a diagnostic support tool, offering users:
- Preliminary AI-based feedback on skin conditions based on facial image analysis or symptom input.
- Possible remedies and treatment plans.
- Links to local, verified dermatologists.

---

## Objectives
- Identify common facial skin conditions using AI.
- Provide user-friendly chat and image-based diagnosis.
- Recommend local dermatologists for further consultation.

---

## Features
- 🧠 AI chatbot for skin disease symptom discussion.
- 📸 Image upload to detect visible facial skin problems.
- 💊 Suggested treatments and home remedies.
- 👨‍⚕️ Recommended dermatologists in Laguna (with clickable profile links).

---

## Technologies Used
- **Frontend:** React.js, TailwindCSS
- **Backend:** Flask (Python)
- **Machine Learning:** PyTorch, Transformers, Mistral-7B API
- **Other Tools:** Git, GitHub, Git LFS (for model files)

---

## Setup and Installation

### Prerequisites:
- Node.js (LTS) – [Download](https://nodejs.org/)
- Python 3.8+ – [Download](https://www.python.org/)
- Git

---

### Installation Steps:

```bash
# 1. Clone the repository
git clone https://github.com/thebadsektor/tc3202-3a-9.git
cd tc3202-3a-9

# 2. Install React frontend dependencies
cd chat-app
npm install

# 3. Install backend dependencies 
cd server-app
npm install

# 4. Install root dependencies (including concurrently)
cd ../../
npm install

# If you get an error about 'concurrently' not found, run this:
npm install concurrently --save-dev
```

---

## Running the Application

```bash
# Starts both:
# - Flask backend (server-app/app.py)
# - React frontend (chat-app/)
npm run start
```

The app will automatically open in your browser at [http://localhost:3000](http://localhost:3000).

---

## Usage Instructions

- Use the chatbot to ask about facial skin symptoms.
- Upload an image to analyze facial skin conditions.
- View treatment advice and doctor recommendations with working profile links.

![Chat UI](https://github.com/user-attachments/assets/a87b0ab1-f4ff-4125-ab5d-9440533603ed)

![Image Upload UI](https://github.com/user-attachments/assets/3371d4e8-e378-48f5-aff3-81f216415579)

---

## Project Structure

```bash
.
├── chat-app/               # React frontend
│   ├── src/components/
│   └── package.json
├── server-app/             # Flask backend
│   ├── api/
│   ├── app.py
│  
├── README.md
```

---

## Contributors

- **Mishael Catignas** – 💻 Lead Developer / Full Stack & AI Integration  
- **Britney Capellan** – Moral Support 
- **Howard Candidato** – Moral Support
- **Rjay Dntcaree** – Moral Support 

---

## Changelog

### [v1.0.0] - 2025-05-01
- Added AI chatbot and image upload functionality
- Implemented facial skin disease detection and treatment suggestions
- Linked local dermatologist profiles via clickable URLs

---

## Acknowledgments

Special thanks to:
- [SkinDiseaseAI2](https://github.com/TheUnmeshRaj/SkinDiseaseAI2) – Original project inspiration.
- Mistral-7B via Lepton API – for lightweight open LLM integration.
- Philippine Dermatological Society – for doctor directory.

---

## License

This project is for academic demonstration purposes. Original templates and datasets used under fair educational use. No commercial intent.
