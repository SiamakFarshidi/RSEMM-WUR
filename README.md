# 📊 RSEMM: Research Software Evaluation & Maturity Monitor

**RSEMM** is an open-source web dashboard designed to evaluate the **maturity**, **FAIRness**, and **impact** of research software. It provides actionable insights for developers, maintainers, and researchers seeking to improve the quality and sustainability of their codebases.

> 🔗 Access the dashboard: [https://ai4rse.nl/RSEMM/](https://ai4rse.nl/RSEMM/)  
> 📂 Evaluation dataset: [Mendeley Data](https://doi.org/10.17632/t2dygzcsyt.2)  
> 💻 Source code: [GitHub Repository](https://github.com/ai4rse/RSEMM)

---

## 🌟 Key Features

- ✅ **FAIR4RS Compliance**  
  Automatically assesses Findability, Accessibility, Interoperability, and Reusability based on metadata and best practices.

- 🧪 **Software Engineering Metrics**  
  Evaluates testing, documentation, version control, CI/CD integration, licensing, and community activity.

- 🤖 **AI-Generated Code Detection**  
  Uses GPT-based zero-shot classification to estimate the presence of AI-generated code and flag potential risks.

- 🔍 **AI/ML Workflow Classification**  
  Identifies MLOps or AIOps components using commit history, documentation, and repository signals.

- 📈 **Citation Tracking**  
  Gathers scholarly impact data from OpenAlex and Europe PMC, including citation counts and author networks.

- 📊 **Maturity Scoring Dashboard**  
  Consolidates all results into an interactive report with a final classification:  
  - Early-stage  
  - Growing  
  - Mature

- 🧭 **Actionable Recommendations**  
  Suggests concrete next steps to improve reproducibility, sustainability, and FAIR compliance.

---

## 🛠️ System Overview

RSEMM is a modular, multi-layered system built with:

- **Frontend:** React  
- **Backend:** Python (FastAPI)  
- **Deployment:** Docker  
- **Data Sources:** GitHub, Zenodo, OpenAlex, Europe PMC  

![System Architecture](https://ai4rse.nl/assets/architecture.png)

---

## 🔎 How It Works

1. **Input a Repository or Zenodo DOI**  
   Evaluate a live project or explore previously analyzed repositories.

2. **Data Collection**  
   Fetches metadata from Zenodo and/or GitHub.

3. **Evaluation Engine**  
   Runs five modules to assess:
   - FAIR4RS principles
   - SE maturity
   - AI/ML workflow classification
   - AI code usage estimation
   - Citation analytics

4. **Scoring and Recommendations**  
   Outputs a visual report with scores, improvement areas, and suggested actions.

---

## 📊 Evaluation at Scale

- **1,519** open-source research software projects evaluated  
- All metadata, results, and scoring models are openly available:  
  🔗 [Mendeley Data DOI](https://doi.org/10.17632/t2dygzcsyt.2)

---

## 📥 Getting Started

### 🧰 Requirements

- Docker
- Python 3.9+
- Git

### 🚀 Run Locally

```bash
git clone https://github.com/ai4rse/RSEMM
cd RSEMM
docker-compose up --build
