# GCP/K8s Propensity Score Matching App (Python Edition)

## Project Overview
Originally conceived as an R Shiny app, the project now utilizes Python and Streamlit to provide a robust, scalable environment for propensity score matching on chemotherapy cohorts.

## Current Status: Phase 1 Complete
The application is fully Dockerized and can be run locally. This establishes the prerequisite for serverless deployment and Kubernetes orchestration.

## Technical Stack
* Language: Python 3.9
* Framework: Streamlit for rapid UI development
* Libraries: pandas, scikit-learn for propensity score matching, and plotly for data visualization
* Infrastructure: Docker

## Next Steps
1. Phase 2: Google Cloud Run. Deploy the container to a serverless environment using GCP free tier to learn IAM basics and container registries.
2. Phase 3: Kubernetes. Orchestrate the app using Minikube or Kind locally to master kubectl and manifest management before moving to GKE Autopilot.
3. Phase 4: Cloud Data Warehousing. Integrate BigQuery to handle analytical workloads, separating them from application logic.

## How to Run Locally
```bash
# Build the image
docker build -t propensity-matcher-python .

# Run the container
docker run -p 8501:8501 propensity-matcher-python
