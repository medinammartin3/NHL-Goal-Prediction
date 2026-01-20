# NHL Expected Goals (xG) Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![Flask](https://img.shields.io/badge/Backend-Flask-000000)
![GCP](https://img.shields.io/badge/Deployment-Google_Cloud_Run-4285F4)

A full-stack Machine Learning application that streams live NHL game data, processes events in real-time, and calculates the **Expected Goals (xG)** probability for every shot. The system is containerized using Docker and deployed on Google Cloud Run.


## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)


## Project Overview

This project implements a complete ML pipeline for hockey analytics. It connects to the **NHL public API** to fetch live play-by-play data, engineers features on the fly (such as shot distance, angle, speed, and rebound status), and queries a serving API to get predictions from a tuned **XGBoost** model.

The goal is to provide real-time insights into game momentum and shot quality that go beyond basic box scores.

The system currently operates using the best-performing **XGBoost** model.

### 🔗 [Try the Deployed App Here](https://streamlit-service-478523788975.us-central1.run.app)

## Project Structure

```text
NHL-Goal-Prediction/
│
├── figures/
│   └── Static visual assets.
│
├── serving/
│   └── Model inference service responsible for real-time xG prediction.
│       This component exposes a REST API that loads a trained ML model
│       and returns expected goal probabilities for incoming shot or
│       play-by-play events.
│
│       Typical responsibilities:
│       - Load trained model artifacts
│       - Validate and preprocess incoming requests
│       - Perform inference
│       - Return predictions in JSON format
│       - Log requests and predictions
│
├── src/
│   └── Core machine learning pipeline logic used across the project.
│
│       Contains:
│       - Data collection, cleaning, preprocessing and visualization
│       - Feature engineering
│       - Models implementation, training, and evaluation code
│         └── Implemented Models:
│             - Logistic Regression (baseline)
│             - XGBoost
│             - Catboost
│             - LightGBM
│             - MLP
│             - Stacking (MLP + Catboost + LightGBM)
│       - Metrics (validation, testing and analysis utilities)
│       - Shared helpers and configuration files
│
├── streamlit/
│   └── Interactive dashboard built with Streamlit for visualization
│       and real-time exploration of expected goals.
│
│       Responsibilities:
│       - User-facing UI
│       - Fetch predictions from the serving API
│       - Display shot locations, xG timelines, and game summaries
│       - Provide interactive controls for model and game selection
│
├── Dockerfile.serving
│   └── Docker image definition for the prediction service.
│       Builds an isolated environment to run the serving API.
│
├── Dockerfile.streamlit
│   └── Docker image definition for the Streamlit dashboard.
│       Runs the UI as a separate container.
│
├── docker-compose.yaml
│   └── Multi-container orchestration that runs:
│       - The ML inference service
│       - The Streamlit dashboard
│       Enables communication between services via internal networking.
│
├── requirements.txt
│   └── Python dependencies required for development, modeling,
│       serving, and visualization.
│
├── setup.py
│   └── Packaging configuration that makes the src/ directory
│       installable as a Python module.
│
└── README.md
    └── Project overview and documentation.
```


## System Architecture 

The application consists of two decoupled microservices:

1.  **Streamlit Frontend (`/streamlit`):**
    * User interface for selecting games and visualizing data.
    * Handles the game logic loop (fetching schedule, pinging events).
    * Displays Shot Maps (Plotly) and xG Evolution charts.
2.  **Flask Serving API (`/serving`):**
    * Loads the trained model artifact from **Weights & Biases (WandB)**.
    * Exposes a REST API (`/predict`, `/download_registry_model`) to serve predictions.
    * Handles feature alignment and validation.

**Data Flow:**
`NHL API` → `Game Client` → `Flask API (Inference)` → `Streamlit (Visualization)`

### Features
* **Schedule Explorer:** Select any date and pick specific matchups (e.g., *Canadiens vs. Bruins*).
* **Real-Time Simulation:** "Ping" the game to load events in batches, simulating a live feed.
* **Interactive Visualizations:**
    * **Shot Map:** Rink overlay showing shot locations, sized by goal probability.
    * **xG Evolution:** Cumulative xG line chart to track team dominance over time.
* **Advanced Metrics:** Displays calculated features (Distance, Angle, Speed, Rebound) alongside the raw event data.
* **Auto-Model Loading:** Automatically pulls the latest production-ready XGBoost model from the registry.
