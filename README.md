# React Website - Setup and Usage

This README provides instructions for setting up and running the React website, its associated Python script, and the Node.js backend.

## Prerequisites

Make sure you have the following installed on your system:

1. **Node.js** (LTS version recommended)  
   Download: [https://nodejs.org/](https://nodejs.org/)

2. **Python** (version 3.8 or higher)  
   Download: [https://www.python.org/](https://www.python.org/)

3. **npm** (comes with Node.js)

4. Any necessary dependencies listed in `requirements.txt` (for Python) and `package.json` (for Node.js).

## Project Structure

The project is organized as follows:

```
project-root/
|-- api/               # Directory containing Python scripts
|   |-- TextAi.py      # Main Python script to be run
|
|-- backend/           # Directory containing Node.js backend
|   |-- package.json   # Dependencies for the backend
|
|-- frontend/          # Directory containing React code
    |-- package.json   # Dependencies for the React frontend
```

## Setup Instructions

### Step 1: Clone the Repository

Clone this repository to your local machine:
```bash
git clone <repository-url>
cd project-root
```

### Step 2: Install Dependencies

#### For Python (API):
Navigate to the `api` directory and install dependencies:
```bash
cd api
pip install requests torch torchvision torchaudio beautifulsoup4 datasets flask flask-cors pillow scikit-learn transformers openai
```

#### For Node.js Backend:
Navigate to the `backend` directory and install dependencies:
```bash
cd backend
npm install
```

#### For React Frontend:
Navigate to the `frontend` directory and install dependencies:
```bash
cd frontend
npm install
```

## Running the Application

### Step 1: Start the Python API
Navigate to the `api` directory and run the Python script:
```bash
cd api
python TextAi.py
```

### Step 2: Start the Node.js Backend
Navigate to the `backend` directory and start the backend:
```bash
cd backend
npm run
```

### Step 3: Start the React Frontend
Navigate to the `frontend` directory and start the React development server:
```bash
cd frontend
npm start
```

### Accessing the Application
Once all services are running:
- Open your browser and navigate to `http://localhost:3000` to view the React frontend.

## Notes
- Make sure the Python API and Node.js backend are running before starting the React frontend.
- If any service fails to start, check for errors in the terminal and resolve missing dependencies or conflicts.

## Troubleshooting
- **Python Issues:** Verify the Python version and ensure all dependencies are installed using `pip install -r requirements.txt`.
- **Node.js Issues:** Delete the `node_modules` folder and run `npm install` again.
- **React Issues:** Clear the cache by running `npm cache clean --force` and reinstall dependencies.

---


