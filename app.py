#importing libraries
import streamlit as st
import pandas as pd
import json
import os
from llama_cpp import Llama 
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#paging configuration
st.set_page_config(
    page_title="Cognitive Hazard AI",
    page_icon="🌍",
    layout="wide"
)

#model loading & caching
# We use @st.cache_resource to load the LLM only once for the entire session.
@st.cache_resource
def load_llm_model():
    """Loads the TinyLlama GGUF model from the local filesystem."""
    try:
        model_path = r"D:\Documents\cognitive_hazard_ai\tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
        llm = Llama(
            model_path=model_path,
            n_ctx=2048, 
            n_gpu_layers=0, 
            verbose=False 
        )
        return llm
    except FileNotFoundError:
        st.error("Model file not found. Please place 'tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf' in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the LLM: {e}")
        st.stop()

#data loading & caching
@st.cache_data
def load_data():
    """Loads supplier and global event data from CSV files."""
    try:
        suppliers_df = pd.read_csv("suppliers.csv")
        events_df = pd.read_csv("global_events.csv")
        return suppliers_df, events_df
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Make sure the CSV files are in the same directory.")
        return None, None

#semantic search engine(vector store) set up
@st.cache_resource
def create_event_retriever(events_df):
    """Creates a FAISS index for fast semantic search of events."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    event_embeddings = model.encode(events_df['headline'].tolist(), convert_to_tensor=True)
    index = faiss.IndexFlatL2(event_embeddings.shape[1])
    index.add(event_embeddings.cpu().numpy())
    return index, model

#core AI analysis function
def generate_risk_analysis(llm, supplier_info, relevant_events):
    """Generates a supply chain risk analysis using the TinyLlama model."""
    
    prompt = f"""\
    <|system|>
    You are an expert Supply Chain Intelligence Analyst working for Myntra.
    Your mission is to identify and analyze potential disruptions for a key supplier.

    **SUPPLIER PROFILE:**
    - Name: {supplier_info['supplier_name']}
    - Location: {supplier_info['city']}, {supplier_info['country']}
    - Products: {supplier_info['products']}

    **RELEVANT GLOBAL INTELLIGENCE (RECENT EVENTS):**
    {relevant_events.to_string(index=False)}

    **YOUR TASK:**
    1.  **Analyze Connections:** Carefully read the intelligence reports and identify any events that could directly or indirectly impact the specified supplier.
    2.  **Assess Risk Level:** Based on the severity and likelihood of the impact, determine an overall risk level. The levels are: 'Low', 'Medium', 'High', or 'Critical'.
    3.  **Provide Reasoning:** Write a concise, clear paragraph explaining *why* you've assigned this risk level. Connect the specific events to potential impacts on the supplier (e.g., "The predicted cyclone could flood roads, delaying transport from the factory to the port.").
    4.  **Recommend Action:** Suggest a concrete, actionable step that the Myntra logistics team should take immediately.

    **OUTPUT FORMAT:**
    Respond ONLY with a valid JSON object. Do not include any other text or markdown formatting outside of the JSON.
    Example:
    {{
      "risk_level": "High",
      "reasoning": "The combination of a looming port strike and a worsening container shortage presents a significant risk of shipping delays of 1-2 weeks for all suppliers in this region.",
      "recommendation": "Immediately contact the supplier to confirm their current production status and explore alternative air freight options for the most time-sensitive products."
    }}
    <|user|>
    Generate the risk analysis JSON for the provided data.\
    """

    try:
        # We call the local LLM to generate the content
        output = llm(prompt, max_tokens=512, stop=["<|endoftext|>", "<|assistant|>"], echo=False)
        response_text = output["choices"][0]["text"]
        
        # Clean up the response to make sure it's a valid JSON
        json_response = json.loads(response_text.strip())
        return json_response
    except Exception as e:
        st.error(f"🚨 An error occurred while communicating with the AI: {e}")
        st.error(f"LLM Response Text: {response_text if 'response_text' in locals() else 'No response'}")
        return None

#streamlit user interface
suppliers_df, events_df = load_data()
if suppliers_df is not None and events_df is not None:
    event_retriever, retriever_model = create_event_retriever(events_df)
    llm = load_llm_model() 
else:
    st.stop()

st.title("🌍 Cognitive Hazard AI")
st.markdown("A Proactive Supply Chain Intelligence Engine for **Myntra**")

st.sidebar.title("Supplier Selection")
selected_supplier_name = st.sidebar.selectbox(
    "Select a supplier to analyze:",
    suppliers_df['supplier_name']
)

analyze_button = st.button(f"Analyze Risk for {selected_supplier_name}", type="primary")

st.subheader("Global Intelligence Feed")
st.dataframe(events_df, use_container_width=True)

if analyze_button:
    supplier_info = suppliers_df[suppliers_df['supplier_name'] == selected_supplier_name].iloc[0]

    with st.spinner(f"Running cognitive analysis for {selected_supplier_name}..."):
        search_query = supplier_info['location_tags']
        query_embedding = retriever_model.encode([search_query])
        
        D, I = event_retriever.search(np.array(query_embedding).astype('float32'), 5)
        relevant_events_df = events_df.iloc[I[0]]
        
        analysis_result = generate_risk_analysis(llm, supplier_info, relevant_events_df)
        
        if analysis_result:
            st.subheader(f"Risk Analysis for: {supplier_info['supplier_name']}")
            
            risk_level = analysis_result.get("risk_level", "Unknown").upper()
            
            if risk_level == "LOW":
                st.success(f"**Risk Level: {risk_level}**")
            elif risk_level == "MEDIUM":
                st.warning(f"**Risk Level: {risk_level}**")
            else:
                st.error(f"**Risk Level: {risk_level}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Reasoning:**\\n\\n{analysis_result.get('reasoning', 'No reasoning provided.')}")
            with col2:
                st.warning(f"**Recommended Action:**\\n\\n{analysis_result.get('recommendation', 'No recommendation provided.')}")

            st.markdown("---")
            st.write("This analysis was based on the following intelligence signals:")
            st.dataframe(relevant_events_df)