# Cognitive Hazard AI - Proactive Supply Chain Intelligence Engine

Cognitive Hazard AI is a proof-of-concept for a proactive supply chain risk analysis system. It leverages a Large Language Model (LLM) to read, understand, and connect disparate global events, predicting potential disruptions to key suppliers before they manifest in traditional logistics dashboards.

---

### The Core Business Problem

In the fast-paced fashion industry, supply chain disruptions are a critical threat. A two-week delay in sourcing materials from a global supplier can lead to:
-   **Missed Seasons:** A seasonal collection (e.g., a "Diwali Collection") arriving late results in massive inventory write-downs.
-   **Lost Revenue:** Inability to restock a trending item means direct revenue loss to competitors.
-   **Increased Costs:** Forced reliance on expensive last-minute air freight to mitigate delays erodes profit margins.

Traditional logistics software is **reactive**. It can tell you when a shipment is already delayed, but not *why* it might be delayed next week.

### The Solution: Proactive Cognitive Intelligence

This project demonstrates a system that moves from reaction to **proactive prediction**. By ingesting and analyzing unstructured, multi-lingual data from a simulated feed of global events (news, weather, labor disputes, etc.), the AI can identify non-obvious, second-order risks.

The engine connects a local weather event in Bangladesh with a separate labor dispute and a global shipping container shortage to predict a high-risk scenario for a specific garment factory in Dhaka, providing actionable intelligence to the business.

---

### Key Features

-   **Supplier Risk Dashboard:** Select a key supplier from a list to perform a real-time risk analysis.
-   **Dynamic Event Analysis:** The system identifies the most relevant global events for a given supplier's location and operational context.
-   **LLM-Powered Risk Assessment:** Uses Google's Gemini Pro to generate a clear, qualitative risk analysis with a defined level (Low, Medium, High, Critical).
-   **Actionable Recommendations:** The AI provides a concrete, recommended next step for the logistics team to mitigate the identified risk.

### Technical Architecture & Tech Stack

This project is built as a proof-of-concept using a **Retrieval-Augmented Generation (RAG)** architecture.

1.  **Data Ingestion:** Supplier and event data are loaded from CSV files, simulating a connection to company databases and real-time intelligence feeds.
2.  **Semantic Retrieval (The "RAG" part):**
    -   A **Sentence Transformer** model (`all-MiniLM-L6-v2`) converts all event headlines into numerical vector embeddings.
    -   These embeddings are stored in a **FAISS** index, a high-performance vector search library that acts as our local vector database.
    -   When a supplier is selected, its `location_tags` are used to perform a semantic search against the FAISS index to retrieve the most contextually relevant events.
3.  **Generative Reasoning:**
    -   The retrieved events and the supplier profile are dynamically compiled into a detailed prompt.
    -   This prompt is sent to the **Google Gemini Pro** LLM, which is tasked with performing the causal reasoning, risk assessment, and generating the final analysis in a structured JSON format.
4.  **Frontend:**
    -   The entire application is built and served using **Streamlit**, a Python framework for building interactive data science web apps.

**Tech Stack:**
-   **Language:** Python
-   **Web Framework:** Streamlit
-   **LLM:** Google Gemini Pro
-   **Vector Embeddings:** `sentence-transformers`
-   **Vector Search:** `faiss-cpu` (Facebook AI Similarity Search)
-   **Core Libraries:** `pandas`, `numpy`

---

### Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/cognitive_hazard_ai.git](https://github.com/your-username/cognitive_hazard_ai.git)
    cd cognitive_hazard_ai
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    The `requirements.txt` file is configured to use a PyTorch backend to avoid dependency conflicts.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Key:**
    -   Create a file at `.streamlit/secrets.toml`.
    -   Add your Google Gemini API key to the file:
        ```toml
        GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
        ```

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
