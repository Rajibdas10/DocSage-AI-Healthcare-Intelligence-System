import streamlit as st
import os
import traceback
import logging
from datetime import datetime
from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qna_hybrid.qna_pipeline import HealthcareRAGPipeline, create_rag_pipeline
from clinical_summarizer.entity_extractor import MedicalEntityExtractor, create_entity_extractor
from clinical_summarizer.summarizer import ClinicalSummarizer, create_clinical_summarizer

# Import new features
from recommendation_engine.recommender import HealthcareRecommender, create_recommender
from sentiment_analysis.sentiment_classifier import SentimentAnalyzer, create_sentiment_analyzer
from sentiment_analysis.sentiment_chatbot import SentimentChatbot

from utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_excel,
    extract_text_from_csv,
    extract_text_from_json
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🩺 DocSage - AI Healthcare Intelligence System",
    page_icon="🏥",
    layout="wide"
)

def setup_sidebar():
    """Setup sidebar with configuration options"""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Enter your Groq API key or set GROQ_API_KEY environment variable"
        )

        # Model selection
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "bge-small-en-v1.5", "bge-base-en-v1.5"],
            help="Choose the embedding model for document processing"
        )

        llm_model = st.selectbox(
            "LLM Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant","openai/gpt-oss-20b"],
            help="Choose the Groq LLM model for response generation"
        )

        # Chunking parameters
        st.subheader("📄 Chunking Settings")
        chunk_size = st.slider("Chunk Size", 256, 1024, 512, 64, 
                              help="Size of text chunks in tokens")
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10,
                                help="Overlap between consecutive chunks")

        # Retrieval parameters
        st.subheader("🔍 Retrieval Settings")
        similarity_top_k = st.slider("Top-K Results", 1, 10, 5, 1,
                                   help="Number of most similar chunks to retrieve")

        return {
            "groq_api_key": groq_api_key,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "similarity_top_k": similarity_top_k
        }

def check_config_changed(new_config):
    """Check if configuration has changed compared to stored config"""
    if "last_config" not in st.session_state:
        st.session_state.last_config = None
        return True
    
    last_config = st.session_state.last_config
    if not last_config:
        return True
    
    # Compare relevant parameters (exclude API key for security)
    config_keys = ["embedding_model", "llm_model", "chunk_size", "chunk_overlap"]
    for key in config_keys:
        if last_config.get(key) != new_config.get(key):
            return True
    
    return False

def reset_pipeline_if_config_changed(config):
    """Reset pipeline if configuration has changed"""
    if check_config_changed(config):
        st.session_state.pipeline = None
        st.session_state.document_processed = False
        st.session_state.store_dir = None
        st.session_state.last_config = config.copy()
        
        # Show notification if config changed (not first load)
        if st.session_state.get("last_config") is not None:
            st.info("🔄 Configuration changed. Please reprocess your document with the new settings.")

def initialize_session_state():
    """Initialize all session state variables"""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "entity_extractor" not in st.session_state:
        st.session_state.entity_extractor = None
    if "recommender" not in st.session_state:
        st.session_state.recommender = None
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = None
    if "store_dir" not in st.session_state:
        st.session_state.store_dir = None
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "processing_log" not in st.session_state:
        st.session_state.processing_log = []
    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = None
    if "extracted_entities" not in st.session_state:
        st.session_state.extracted_entities = None
    if "recommendation_data" not in st.session_state:
        st.session_state.recommendation_data = None
    if "sentiment_results" not in st.session_state:
        st.session_state.sentiment_results = None

def main():
    st.title("🩺 DocSage - AI Healthcare Intelligence System")
    st.markdown("### Powered by Groq LLaMA-3 and Advanced Embedding Models")

    # Setup sidebar
    config = setup_sidebar()

    # Validate API key
    if not config["groq_api_key"]:
        st.error("🔑 Please provide a Groq API key in the sidebar")
        st.info("Get your free API key from: https://console.groq.com/keys")
        st.stop()

    # Initialize session state
    initialize_session_state()

    # Check if configuration changed and reset pipeline if needed
    reset_pipeline_if_config_changed(config)

    # Initialize components if not exists
    if st.session_state.entity_extractor is None:
        try:
            st.session_state.entity_extractor = create_entity_extractor(
                groq_api_key=config["groq_api_key"],
                llm_model=config["llm_model"]
            )
        except Exception as e:
            st.error(f"Error initializing entity extractor: {e}")

    # Initialize recommendation engine
    if st.session_state.recommender is None:
        try:
            st.session_state.recommender = create_recommender(
                groq_api_key=config["groq_api_key"],
                llm_model=config["llm_model"]
            )
        except Exception as e:
            st.error(f"Error initializing recommender: {e}")

    # Initialize sentiment analyzer
    if st.session_state.sentiment_analyzer is None:
        try:
            st.session_state.sentiment_analyzer = create_sentiment_analyzer(
                groq_api_key=config["groq_api_key"],
                llm_model=config["llm_model"]
            )
        except Exception as e:
            st.error(f"Error initializing sentiment analyzer: {e}")

    # Display current configuration
    with st.expander("🔧 Current Configuration", expanded=False):
        st.write(f"**Embedding Model:** {config['embedding_model']}")
        st.write(f"**LLM Model:** {config['llm_model']}")
        st.write(f"**Chunk Size:** {config['chunk_size']}")
        st.write(f"**Chunk Overlap:** {config['chunk_overlap']}")
        st.write(f"**Top-K Results:** {config['similarity_top_k']}")

    # Create tabs for all four features
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Q&A System", 
        "📋 Clinical Summary", 
        "🎯 Recommendation Engine", 
        "😊 Sentiment Analysis"
    ])

    with tab1:
        qna_tab(config)

    with tab2:
        clinical_summary_tab(config)

    with tab3:
        recommendation_engine_tab(config)

    with tab4:
        sentiment_analysis_tab(config)

    # Display processing log
    if st.session_state.processing_log:
        with st.expander("📋 Processing Log", expanded=False):
            for log_entry in st.session_state.processing_log:
                st.text(f"{log_entry['time']} - {log_entry['message']}")

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using **Streamlit**, **LlamaIndex**, **Groq**, and **ChromaDB**")

def qna_tab(config):
    """Q&A tab content - your existing functionality"""
    st.subheader("💬 Question & Answer System")
    st.markdown("Upload healthcare documents and ask questions about their content.")
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Document Upload")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload healthcare documents",
            type=["pdf", "docx", "xlsx", "csv", "json"],
            help="Supported formats: PDF, Word, Excel, CSV, JSON",
            key="qna_uploader"
        )

        if uploaded_file is not None:
            # Create uploads directory
            os.makedirs("uploads", exist_ok=True)

            try:
                # Save uploaded file
                file_path = os.path.join("uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                st.success(f"✅ File uploaded: {uploaded_file.name}")

                # Processing button
                if st.button("🚀 Process Document", type="primary", key="process_qna"):
                    process_document(file_path, config)

            except Exception as e:
                st.error(f"Error uploading file: {e}")

    with col2:
        st.subheader("💬 Ask Questions")

        if st.session_state.document_processed and st.session_state.pipeline:
            # Query input
            query = st.text_area(
                "Ask a question about your document:",
                height=100,
                placeholder="What are the key findings in this healthcare document?",
                key="qna_query"
            )

            if query and st.button("🔍 Get Answer", key="get_answer"):
                get_answer(query, config)

        else:
            st.info("👆 Please upload and process a document first")

def clinical_summary_tab(config):
    """Clinical Summary tab content - existing functionality"""
    st.subheader("📋 Clinical Summary & Entity Extraction")
    st.markdown("Extract and analyze medical entities from clinical documents.")

    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Document Upload")
        
        # File upload for clinical summary
        uploaded_file = st.file_uploader(
            "Upload clinical documents",
            type=["pdf", "docx", "xlsx", "csv", "json"],
            help="Upload clinical documents for entity extraction and summary",
            key="clinical_uploader"
        )

        if uploaded_file is not None:
            # Create uploads directory
            os.makedirs("uploads", exist_ok=True)

            try:
                # Save uploaded file
                file_path = os.path.join("uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                st.success(f"✅ File uploaded: {uploaded_file.name}")

                # Processing button for clinical summary
                if st.button("🔬 Extract Clinical Entities", type="primary", key="process_clinical"):
                    extract_clinical_entities(file_path, config)

            except Exception as e:
                st.error(f"Error uploading file: {e}")

        # Manual text input option
        st.subheader("✏️ Or Enter Clinical Text")
        manual_text = st.text_area(
            "Enter clinical text manually:",
            height=200,
            placeholder="Enter clinical notes, discharge summaries, or other medical text...",
            key="manual_clinical_text"
        )

        if manual_text and st.button("🔬 Extract from Text", key="extract_from_text"):
            extract_entities_from_text(manual_text, config)

    with col2:
        st.subheader("📊 Clinical Summary Results")

        if st.session_state.extracted_entities:
            display_clinical_summary(st.session_state.extracted_entities)
        else:
            st.info("👈 Please upload a clinical document or enter text to see the summary")

def recommendation_engine_tab(config):
    """Recommendation Engine tab content - third feature"""
    st.subheader("🎯 Personalized Healthcare Recommendations")
    st.markdown("Generate personalized recommendations for diet, yoga, medication, and lifestyle based on patient data.")

    # Create recommender instance
    recommender = create_recommender(config["groq_api_key"])

    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Patient Data Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "Upload Document"],
            key="rec_input_method"
        )
        
        if input_method == "Manual Entry":
            # Manual patient data entry
            st.markdown("**Patient Information:**")
            
            age = st.number_input("Age", min_value=0, max_value=120, value=30, key="rec_age")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="rec_gender")
            
            conditions = st.text_area(
                "Medical Conditions:",
                height=100,
                placeholder="Enter known medical conditions, allergies, current medications...",
                key="rec_conditions"
            )
            
            st.markdown("**Lifestyle Preferences:**")
            activity_level = st.selectbox(
                "Activity Level", 
                ["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                key="rec_activity"
            )
            
            dietary_restrictions = st.multiselect(
                "Dietary Restrictions",
                ["Vegetarian", "Vegan", "Gluten-Free", "Diabetic", "Low-Sodium", "Heart-Healthy"],
                key="rec_diet"
            )
            
            if st.button("🎯 Generate Recommendations", type="primary", key="generate_recommendations_manual"):
                patient_data = {
                    "age": age,
                    "gender": gender,
                    "conditions": conditions,
                    "activity_level": activity_level,
                    "dietary_restrictions": dietary_restrictions
                }
                st.session_state.recommendation_data = recommender.generate_recommendations(patient_data)
        
        else:
            # Document upload for recommendation
            uploaded_file = st.file_uploader(
                "Upload patient document",
                type=["pdf", "docx", "xlsx", "csv", "json"],
                help="Upload patient documents to extract information for recommendations",
                key="recommendation_uploader"
            )

            if uploaded_file is not None:
                os.makedirs("uploads", exist_ok=True)

                try:
                    file_path = os.path.join("uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())

                    st.success(f"✅ File uploaded: {uploaded_file.name}")

                    if st.button("🎯 Extract Info & Generate Recommendations", type="primary", key="generate_recommendations_doc"):
                        extracted_info = recommender.analyze_patient_document(open(file_path, "r", errors="ignore").read())
                        st.session_state.recommendation_data = recommender.generate_recommendations(extracted_info)

                except Exception as e:
                    st.error(f"Error uploading file: {e}")

    with col2:
        st.subheader("💡 Personalized Recommendations")
        
        if st.session_state.recommendation_data:
            display_recommendations(st.session_state.recommendation_data)
        else:
            st.info("👈 Please provide patient information to generate recommendations")


def sentiment_analysis_tab(config):
    st.subheader("💬 Sentiment-Aware Chatbot")
    
    # Check if API key is provided
    if not config.get("groq_api_key"):
        st.error("❌ Please enter your Groq API Key in the sidebar to use the chatbot.")
        return
    
    if "chatbot" not in st.session_state:
        # FIXED: Use config API key instead of environment variable
        st.session_state.chatbot = SentimentChatbot(groq_api_key=config["groq_api_key"])
    
    # Also handle config changes - recreate chatbot if API key changes
    if "last_chatbot_api_key" not in st.session_state:
        st.session_state.last_chatbot_api_key = None
    
    if st.session_state.last_chatbot_api_key != config["groq_api_key"]:
        st.session_state.chatbot = SentimentChatbot(groq_api_key=config["groq_api_key"])
        st.session_state.last_chatbot_api_key = config["groq_api_key"]
        st.session_state.chat_history = []  # Reset chat history when API key changes
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    user_message = st.chat_input("Type your message here...")
    
    if user_message:
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        result = st.session_state.chatbot.generate_response(user_message, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": result["chatbot_response"]})
    
    # Display messages
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])


def generate_recommendations(age, gender, conditions, activity_level, dietary_restrictions, config):
    """Generate health recommendations using the engine"""
    try:
        with st.spinner("Generating personalized recommendations..."):
            
            # Prepare patient data in the format expected by your engine
            patient_data = {
                "age": age,
                "gender": gender,
                "conditions": conditions,
                "activity_level": activity_level,
                "dietary_restrictions": dietary_restrictions
            }
            
            # Initialize recommender engine
            recommender = HealthcareRecommender(config)
            
            # Generate recommendations (engine returns basic + enhanced)
            recommendations = recommender.generate_recommendations(patient_data)
            
            # Save into session state
            st.session_state.recommendation_data = {
                "patient_info": patient_data,
                "recommendations": recommendations
            }
            
            st.rerun()
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")


def analyze_sentiment_from_text(text, config):
    """Analyze sentiment from text input (placeholder implementation)"""
    with st.spinner("Analyzing sentiment..."):
        try:
            # TODO: Implement actual sentiment analysis logic
            
            add_to_log("Analyzing text sentiment")
            
            # Simulate sentiment analysis
            import time
            time.sleep(1)  # Simulate processing time
            
            # Create sample sentiment results
            sentiment_results = {
                "overall_sentiment": {
                    "label": "Positive",
                    "score": 0.72,
                    "confidence": 0.85
                },
                "emotions": {
                    "joy": 0.3,
                    "trust": 0.4,
                    "fear": 0.1,
                    "sadness": 0.05,
                    "anger": 0.02,
                    "surprise": 0.08,
                    "anticipation": 0.25
                },
                "urgency_level": "Medium",
                "patient_satisfaction": "Satisfied",
                "key_phrases": ["feeling better", "concerned about", "grateful for care"],
                "text_length": len(text),
                "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.sentiment_results = sentiment_results
            
            add_to_log("Sentiment analysis completed")
            st.success("🎉 Sentiment analysis completed!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error analyzing sentiment: {str(e)}")
            add_to_log(f"Sentiment analysis error: {str(e)}")

def analyze_sentiment_from_document(file_path, config):
    """Analyze sentiment from uploaded document (placeholder implementation)"""
    with st.spinner("Extracting text and analyzing sentiment..."):
        try:
            # Extract text from document
            add_to_log("Extracting text for sentiment analysis")
            
            if file_path.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file_path)
            elif file_path.endswith(".docx"):
                extracted_text = extract_text_from_docx(file_path)
            elif file_path.endswith((".xlsx", ".xls")):
                extracted_text = extract_text_from_excel(file_path, save_json=False)
            elif file_path.endswith(".csv"):
                extracted_text = extract_text_from_csv(file_path)
            elif file_path.endswith(".json"):
                extracted_text = extract_text_from_json(file_path)
            elif file_path.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            else:
                raise ValueError("Unsupported file format for sentiment analysis")

            if "Error" in extracted_text:
                raise ValueError(extracted_text)
            
            # Analyze sentiment from extracted text
            analyze_sentiment_from_text(extracted_text, config)
            
            # Clean up file
            try:
                os.remove(file_path)
            except:
                pass

        except Exception as e:
            st.error(f"❌ Error analyzing document sentiment: {str(e)}")
            add_to_log(f"Document sentiment analysis error: {str(e)}")

def display_recommendations(recommendation_data):
    """Display the generated recommendations"""
    recommendations = recommendation_data["recommendations"]
    patient_info = recommendation_data["patient_profile"]   
    
    # Patient Summary
    st.markdown("### 👤 Patient Profile")
    st.write(f"**Age:** {patient_info['age']} | **Gender:** {patient_info['gender']}")
    st.write(f"**Activity Level:** {patient_info['activity_level']}")
    if patient_info['dietary_restrictions']:
        st.write(f"**Dietary Restrictions:** {', '.join(patient_info['dietary_restrictions'])}")
    
    st.markdown("---")
    
    # Create recommendation tabs
    rec_tabs = st.tabs(["🥗 Diet", "🏃 Exercise", "💊 Medication", "🌱 Lifestyle"])
    
    categories = ["dietary", "exercise", "medication", "lifestyle"]
    icons = ["🥗", "🏃", "💊", "🌱"]
    titles = ["Dietary Recommendations", "Exercise Recommendations", "Medication Guidelines", "Lifestyle Recommendations"]
    
    for tab, cat, icon, title in zip(rec_tabs, categories, icons, titles):
        with tab:
            st.subheader(f"{icon} {title}")
            
            # Basic Recommendations (expanded by default)
            with st.expander("📋 Basic Recommendations", expanded=True):
                if recommendations[cat]["basic"]:
                    for rec in recommendations[cat]["basic"]:
                        st.write(f"- {rec}")
                else:
                    st.write("_No basic recommendations available._")
            
            # Enhanced Recommendations (collapsed by default)
            with st.expander("✨ Enhanced Recommendations", expanded=False):
                if recommendations[cat]["enhanced"]:
                    for rec in recommendations[cat]["enhanced"]:
                        st.write(f"- {rec}")
                else:
                    st.write("_No enhanced recommendations available._")
    
    # Export functionality
    st.markdown("---")
    if st.button("📋 Export Recommendations", key="export_recommendations"):
        json_str = json.dumps(recommendation_data, indent=2)
        st.code(json_str, language="json")
        st.info("Recommendations data displayed above - you can copy it manually")

def display_sentiment_results(sentiment_results, chatbot_response=None):
    """Display sentiment analysis results + chatbot response"""

    # Overall sentiment
    sentiment = sentiment_results["overall_sentiment"]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Overall Sentiment", sentiment["label"], f"Score: {sentiment['score']:.2f}")
    with col2:
        st.metric("Confidence", f"{sentiment['confidence']:.1%}")
    with col3:
        st.metric("Urgency Level", sentiment_results["urgency_level"])

    # Emotion breakdown
    st.subheader("🎭 Emotion Analysis")
    emotions = sentiment_results["emotions"]

    for emotion, score in emotions.items():
        if score > 0.01:  # Only show emotions with meaningful scores
            st.write(f"**{emotion.title()}:** {score:.1%}")
            st.progress(score)

    # Key insights
    st.subheader("🔍 Key Insights")
    st.write(f"**Patient Satisfaction:** {sentiment_results['patient_satisfaction']}")

    if sentiment_results["key_phrases"]:
        st.write("**Key Phrases:**")
        for phrase in sentiment_results["key_phrases"]:
            st.write(f"• '{phrase}'")

    # 💬 Chatbot response section
    if chatbot_response:
        st.markdown("---")
        st.subheader("🤖 Sentiment-Aware Chatbot Response")
        st.info(chatbot_response)

    # Export functionality
    st.markdown("---")
    if st.button("📋 Export Sentiment Analysis", key="export_sentiment"):
        json_str = json.dumps(sentiment_results, indent=2)
        st.code(json_str, language="json")
        st.info("Sentiment analysis data displayed above - you can copy it manually")

def extract_clinical_entities(file_path: str, config: dict):
    """Extract clinical entities from uploaded file"""
    with st.spinner("Extracting clinical entities..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Extract text
            status_text.text("📄 Extracting text from document...")
            progress_bar.progress(25)
            add_to_log("Starting clinical text extraction")

            if file_path.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file_path)
            elif file_path.endswith(".docx"):
                extracted_text = extract_text_from_docx(file_path)
            elif file_path.endswith(".xlsx"):
                extracted_text = extract_text_from_excel(file_path, save_json=True)
            elif file_path.endswith(".csv"):
                extracted_text = extract_text_from_csv(file_path)
            elif file_path.endswith(".json"):
                extracted_text = extract_text_from_json(file_path)
            else:
                raise ValueError("Unsupported file format")

            if "Error" in extracted_text:
                raise ValueError(extracted_text)

            progress_bar.progress(50)

            # Step 2: Extract entities
            status_text.text("🔬 Extracting medical entities...")
            add_to_log("Extracting medical entities using LLM")

            entities_result = st.session_state.entity_extractor.extract_entities(extracted_text)
            
            progress_bar.progress(75)

            # Step 3: Store results
            st.session_state.extracted_text = extracted_text
            st.session_state.extracted_entities = entities_result

            progress_bar.progress(100)
            status_text.text("✅ Clinical entity extraction complete!")

            add_to_log("Clinical entity extraction completed successfully")

            # Clean up
            try:
                os.remove(file_path)
            except:
                pass

            st.success("🎉 Clinical entities extracted successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error extracting clinical entities: {str(e)}")
            add_to_log(f"Clinical extraction error: {str(e)}")
            logger.error(f"Clinical entity extraction error: {e}")

            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

def extract_entities_from_text(text: str, config: dict):
    """Extract entities from manually entered text"""
    with st.spinner("Extracting clinical entities from text..."):
        try:
            entities_result = st.session_state.entity_extractor.extract_entities(text)
            
            st.session_state.extracted_text = text
            st.session_state.extracted_entities = entities_result

            add_to_log("Clinical entities extracted from manual text")
            st.success("🎉 Clinical entities extracted successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error extracting entities: {str(e)}")
            add_to_log(f"Manual text extraction error: {str(e)}")

def display_clinical_summary(entities_result):
    """Display the extracted clinical entities in an organized format"""
    entities = entities_result["entities"]
    metadata = entities_result["metadata"]

    # Display metadata
    st.info(f"📊 Extracted {metadata['total_entities']} entities from {metadata['text_length']} characters")

    # Create tabs for different entity types
    entity_tabs = st.tabs([
        "💊 Medications", "🏥 Conditions", "⚕️ Procedures", 
        "📈 Vital Signs", "🔬 Lab Results", "⚠️ Allergies",
        "📋 Summary"
    ])

    with entity_tabs[0]:  # Medications
        st.subheader("💊 Medications")
        if entities.get("medications"):
            for i, med in enumerate(entities["medications"], 1):
                with st.expander(f"Medication {i}: {med.get('name', 'Unknown')}"):
                    st.write(f"**Name:** {med.get('name', 'N/A')}")
                    st.write(f"**Dosage:** {med.get('dosage', 'N/A')}")
                    st.write(f"**Frequency:** {med.get('frequency', 'N/A')}")
                    st.write(f"**Route:** {med.get('route', 'N/A')}")
        else:
            st.info("No medications found")

    with entity_tabs[1]:  # Conditions
        st.subheader("🏥 Medical Conditions")
        if entities.get("conditions"):
            for i, condition in enumerate(entities["conditions"], 1):
                with st.expander(f"Condition {i}: {condition.get('name', 'Unknown')}"):
                    st.write(f"**Name:** {condition.get('name', 'N/A')}")
                    st.write(f"**Status:** {condition.get('status', 'N/A')}")
                    st.write(f"**Severity:** {condition.get('severity', 'N/A')}")
        else:
            st.info("No medical conditions found")

    with entity_tabs[2]:  # Procedures
        st.subheader("⚕️ Medical Procedures")
        if entities.get("procedures"):
            for i, procedure in enumerate(entities["procedures"], 1):
                with st.expander(f"Procedure {i}: {procedure.get('name', 'Unknown')}"):
                    st.write(f"**Name:** {procedure.get('name', 'N/A')}")
                    st.write(f"**Date:** {procedure.get('date', 'N/A')}")
                    st.write(f"**Location:** {procedure.get('location', 'N/A')}")
        else:
            st.info("No procedures found")

    with entity_tabs[3]:  # Vital Signs
        st.subheader("📈 Vital Signs")
        if entities.get("vital_signs"):
            for i, vital in enumerate(entities["vital_signs"], 1):
                with st.expander(f"Vital Sign {i}: {vital.get('type', 'Unknown')}"):
                    st.write(f"**Type:** {vital.get('type', 'N/A')}")
                    st.write(f"**Value:** {vital.get('value', 'N/A')} {vital.get('unit', '')}")
                    st.write(f"**Date:** {vital.get('date', 'N/A')}")
        else:
            st.info("No vital signs found")

    with entity_tabs[4]:  # Lab Results
        st.subheader("🔬 Laboratory Results")
        if entities.get("lab_results"):
            for i, lab in enumerate(entities["lab_results"], 1):
                with st.expander(f"Lab Test {i}: {lab.get('test_name', 'Unknown')}"):
                    st.write(f"**Test:** {lab.get('test_name', 'N/A')}")
                    st.write(f"**Value:** {lab.get('value', 'N/A')} {lab.get('unit', '')}")
                    st.write(f"**Reference Range:** {lab.get('reference_range', 'N/A')}")
                    st.write(f"**Status:** {lab.get('status', 'N/A')}")
        else:
            st.info("No lab results found")

    with entity_tabs[5]:  # Allergies
        st.subheader("⚠️ Allergies")
        if entities.get("allergies"):
            for i, allergy in enumerate(entities["allergies"], 1):
                with st.expander(f"Allergy {i}: {allergy.get('allergen', 'Unknown')}"):
                    st.write(f"**Allergen:** {allergy.get('allergen', 'N/A')}")
                    st.write(f"**Reaction:** {allergy.get('reaction', 'N/A')}")
                    st.write(f"**Severity:** {allergy.get('severity', 'N/A')}")
        else:
            st.info("No allergies found")

    with entity_tabs[6]:  # Summary
        st.subheader("📋 Entity Summary")
        
        # Create summary statistics
        summary_data = []
        for entity_type, entity_list in entities.items():
            if isinstance(entity_list, list):
                count = len(entity_list)
                if count > 0:
                    summary_data.append({
                        "Entity Type": entity_type.replace("_", " ").title(),
                        "Count": count
                    })

        if summary_data:
            import pandas as pd
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No entities found to summarize")

        # Display other entity types
        other_entities = ["body_parts", "symptoms", "medical_devices", "healthcare_providers"]
        for entity_type in other_entities:
            if entities.get(entity_type):
                st.subheader(f"📌 {entity_type.replace('_', ' ').title()}")
                if isinstance(entities[entity_type], list):
                    if entities[entity_type] and isinstance(entities[entity_type][0], dict):
                        for item in entities[entity_type]:
                            st.write(f"• {item}")
                    else:
                        for item in entities[entity_type]:
                            st.write(f"• {item}")

        # Export functionality
        st.subheader("📤 Export Results")
        if st.button("📋 Copy JSON to Clipboard"):
            json_str = json.dumps(entities_result, indent=2)
            st.code(json_str, language="json")
            st.info("JSON data displayed above - you can copy it manually")

def add_to_log(message: str):
    """Add message to processing log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_log.append({
        "time": timestamp,
        "message": message
    })

def process_document(file_path: str, config: dict):
    """Process uploaded document through the RAG pipeline"""

    with st.spinner("Processing document..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Extract text
            status_text.text("📄 Extracting text from document...")
            progress_bar.progress(20)
            add_to_log("Starting text extraction")

            if file_path.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file_path)
            elif file_path.endswith(".docx"):
                extracted_text = extract_text_from_docx(file_path)
            elif file_path.endswith(".xlsx"):
                extracted_text = extract_text_from_excel(file_path, save_json=True)
            elif file_path.endswith(".csv"):
                extracted_text = extract_text_from_csv(file_path)
            elif file_path.endswith(".json"):
                extracted_text = extract_text_from_json(file_path)
            else:
                raise ValueError("Unsupported file format")

            if "Error" in extracted_text:
                raise ValueError(extracted_text)

            add_to_log(f"Extracted {len(extracted_text)} characters")
            progress_bar.progress(40)

            # Step 2: Initialize RAG pipeline with current config
            status_text.text("🤖 Initializing RAG pipeline...")
            add_to_log(f"Setting up models: {config['embedding_model']} + {config['llm_model']}")

            # Create NEW pipeline with current configuration
            pipeline = create_rag_pipeline(
                groq_api_key=config["groq_api_key"],
                embedding_model=config["embedding_model"],
                llm_model=config["llm_model"],
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"]
            )

            progress_bar.progress(60)

            # Step 3: Process document
            status_text.text("⚙️ Processing document (chunking + embedding)...")
            add_to_log("Creating chunks and generating embeddings")

            store_dir = pipeline.process_document(extracted_text)

            progress_bar.progress(80)

            # Step 4: Finalize
            status_text.text("✅ Document processing complete!")

            # Store pipeline and config in session state
            st.session_state.pipeline = pipeline
            st.session_state.store_dir = store_dir
            st.session_state.document_processed = True
            st.session_state.last_config = config.copy()  # Save current config

            progress_bar.progress(100)
            add_to_log("Document processing completed successfully")

            # Clean up
            try:
                os.remove(file_path)
            except:
                pass

            st.success("🎉 Document processed successfully! You can now ask questions.")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            add_to_log(f"Error: {str(e)}")
            logger.error(f"Document processing error: {e}")

            # Show detailed error in expander
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

def get_answer(query: str, config: dict):
    """Get answer for user query"""

    with st.spinner("Finding answer..."):
        try:
            # Use the pipeline stored in session state (which has correct config)
            result = st.session_state.pipeline.query_documents(
                query=query,
                store_dir=st.session_state.store_dir
            )

            # Display answer
            st.subheader("💡 Answer")
            st.write(result["answer"])

            # Display metadata
            if result.get("source_nodes", 0) > 0:
                st.info(f"📊 Answer based on {result['source_nodes']} relevant document chunks")

            # Show which models were used
            st.info(f"🤖 Generated using: {config['llm_model']} (LLM) + {config['embedding_model']} (Embeddings)")

            add_to_log(f"Answered query: {query[:50]}...")

        except Exception as e:
            st.error(f"❌ Error getting answer: {str(e)}")
            add_to_log(f"Query error: {str(e)}")

            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()