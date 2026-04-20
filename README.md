# GCP/K8s Propensity Score Matching App (Python Edition)

## Project Overview
Originally conceived as an R Shiny app, the project now utilizes Python and Streamlit to provide a robust, scalable environment for propensity score matching on chemotherapy cohorts.

## Current Status: Phase 2 Complete
The application is fully Dockerized and deployed to Google Cloud Run. This establishes the foundation for serverless scalability and cloud resource management.

## Technical Stack
* Language: Python 3.9
* Framework: Streamlit for rapid UI development
* Libraries: pandas, scikit-learn for propensity score matching, and plotly for data visualization
* Infrastructure: Docker, Google Artifact Registry, Google Cloud Run

## Phase 2 Deployment Notes
To ensure environment parity between local ARM64 development (Mac) and Cloud Run (x86_64), the image must be built using the --platform linux/amd64 flag. Additionally, memory allocation was increased to 2Gi to handle the pre-processing of the MSK Chord 2024 dataset.

## Next Steps
1. Phase 3: Kubernetes. Orchestrate the app using Minikube or Kind locally to master kubectl and manifest management before moving to GKE Autopilot.
2. Phase 4: Cloud Data Warehousing. Integrate BigQuery to handle analytical workloads, separating them from application logic.

## How to Run Locally
```bash
# Build the image
docker build -t propensity-matcher-python .

# Run the container
docker run -p 8501:8501 propensity-matcher-python
