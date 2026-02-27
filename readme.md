# Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

## 📋 Project Objective
This project demonstrates a complete MLOps workflow: transitioning from an experimental Jupyter Notebook to production-ready Python scripts, fine-tuning a transformer model using the Hugging Face ecosystem, containerizing the environment with Docker, and deploying the final artifacts to the Hugging Face Model Hub.

---

## 🚀 Quick Links
* **Hugging Face Model:** [Jaishankar02/distilbert-reviews-genres](https://huggingface.co/Jaishankar02/distilbert-reviews-genres)
* **GitHub Repository:** `[Insert Your GitHub Link Here]`

---

## 📁 Project Structure
The project has been refactored from a monolithic notebook into a modular Python package:

```text
├── src/
│   ├── data.py         # Data loading and preprocessing logic
│   ├── train.py        # Model training script using Hugging Face Trainer API
│   ├── eval.py         # Evaluation logic for local & remote models
│   └── utils.py        # Helper functions (logging, config)
├── docker/
│   ├── Dockerfile      # Production Docker image configuration
│   └── requirements.txt # Python dependencies
├── results/
│   └── eval_results.json # Saved evaluation metrics
├── main.py             # Main entry point for the pipeline
└── README.md           # Project documentation

---

### Chunk 2: Results and Docker Deployment
```markdown
---

## 🧪 Model Training & Evaluation

### Model Selection
I utilized the **DistilBERT** (`distilbert-base-uncased`) architecture.
* **Reasoning:** It provides 95% of BERT's performance while being 40% smaller and 60% faster, making it ideal for containerized deployment.

### Evaluation Results
The model was evaluated on **127 samples**. Results from the Hugging Face repository:

| Metric | Value |
| :--- | :--- |
| **Model ID** | `Jaishankar02/distilbert-reviews-genres` |
| **Accuracy** | 20.47% |
| **F1 Score** | 0.0727 |
| **Precision** | 0.0442 |
| **Recall** | 0.2047 |
| **Loss** | 4.0127 |

---

## 🐳 Docker Deployment (Production)

The final Docker image pulls the model directly from the Hugging Face Hub and executes evaluation on startup.

### Build and Run
```bash
# Build the production image
docker build -t jaishankar/eval-app:latest -f docker/Dockerfile .

# Run the container
docker run --rm jaishankar/eval-app:latest

---

### Chunk 3: Short Report and Links
```markdown
---

## 📝 Short Report

### 1. Training Summary
The process involved modularizing the instructor's notebook into separate scripts. Training used the Hugging Face `Trainer` API for efficient optimization. The fine-tuned model was pushed to `Jaishankar02/distilbert-reviews-genres`.

### 2. Evaluation Comparison
The re-evaluation inside the Docker container matched the local metrics perfectly, confirming that the model weights and tokenizer were correctly uploaded and accessible.

### 3. Challenges
* **Refactoring:** Converting global notebook variables into a modular script format.
* **Dockerization:** Optimizing image size for heavy ML libraries like `torch`.
* **Deployment:** Automating model pulling within the container startup sequence.

---

## 🔗 Submission Links
* **Hugging Face Model:** [Jaishankar02/distilbert-reviews-genres](https://huggingface.co/Jaishankar02/distilbert-reviews-genres)
