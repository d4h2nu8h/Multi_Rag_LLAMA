import os
from dotenv import load_dotenv
import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langchain_community.llms import Ollama

# Load environment variables from .env file
load_dotenv()

# Initialize the local Llama3.1 model
@st.cache_resource
def init_ollama():
    try:
        llm = Ollama(model="llama3.1")
        # Test connection by trying to invoke with a simple prompt
        test_response = llm.invoke("test")
        return llm
    except Exception as e:
        st.warning(f"⚠️ Ollama is not available: {str(e)}. Please start Ollama and ensure the 'llama3.1' model is installed.")
        return None

llm = init_ollama()

# Neo4j connection details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Initialize Neo4j driver
_neo4j_available = None

@st.cache_resource
def init_neo4j_driver():
    global _neo4j_available
    if _neo4j_available is False:
        return None
    
    # Check if environment variables are set
    if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
        st.warning("⚠️ Neo4j environment variables are not set. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your .env file.")
        _neo4j_available = False
        return None
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        # Test connection
        driver.verify_connectivity()
        _neo4j_available = True
        return driver
    except Exception as e:
        st.warning(f"⚠️ Failed to connect to Neo4j: {str(e)}. Please ensure Neo4j is running and the connection details are correct.")
        _neo4j_available = False
        return None

# Initialize sentence transformer model for semantic search
@st.cache_resource
def init_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_pdf_content():
    """Load the content of the PDF from the text file."""
    try:
        with open('extracted_text.txt', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        st.error("Error: 'extracted_text.txt' file not found. Please ensure the file exists in the same directory as this script.")
        return None

@st.cache_data
def preprocess_and_index(_pdf_content):
    """Preprocess the PDF content, compute embeddings, and create a FAISS index."""
    content_chunks = _pdf_content.split('\n\n')  # Split into paragraphs
    
    # Compute embeddings
    model = init_sentence_transformer()
    embeddings = model.encode(content_chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return index, content_chunks

@st.cache_data
def load_or_create_index(_pdf_content):
    """Load existing index or create a new one if it doesn't exist."""
    return preprocess_and_index(_pdf_content)

def semantic_search(user_query, index, content_chunks, top_k=3):
    """Perform semantic search using FAISS."""
    model = init_sentence_transformer()
    query_vector = model.encode([user_query]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return [content_chunks[i] for i in indices[0]]

def get_neo4j_status():
    """Check Neo4j connection status and return knowledge graph statistics."""
    driver = init_neo4j_driver()
    if driver is None:
        return False, None, None, None
    
    try:
        with driver.session() as session:
            # Get node count
            node_result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            node_count = node_result.single()['count']
            
            # Get relationship count
            rel_result = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count")
            rel_count = rel_result.single()['count']
            
            # Get sample entities
            sample_result = session.run("MATCH (e:Entity) RETURN e.name as name LIMIT 10")
            sample_entities = [record['name'] for record in sample_result]
            
            # Get sample relationships
            sample_rel_result = session.run(
                "MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity) "
                "RETURN e1.name as from, e2.name as to LIMIT 5"
            )
            sample_relationships = [(record['from'], record['to']) for record in sample_rel_result]
            
            return True, node_count, rel_count, (sample_entities, sample_relationships)
    except Exception as e:
        return False, None, None, None

def query_knowledge_graph(user_query):
    """Query the knowledge graph for relevant information."""
    driver = init_neo4j_driver()
    if driver is None:
        return []  # Return empty list if Neo4j is not available
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)-[r:RELATED_TO]->(related:Entity)
                WHERE e.name CONTAINS $user_query OR related.name CONTAINS $user_query
                RETURN e.name AS entity, type(r) AS relation, related.name AS related_entity
                LIMIT 5
                """,
                user_query=user_query
            )
            return [f"{record['entity']} {record['relation']} {record['related_entity']}" 
                    for record in result]
    except Exception as e:
        st.warning(f"Error querying knowledge graph: {str(e)}")
        return []  # Return empty list on error

def generate_response(user_query, pdf_excerpts, kg_info):
    """Generate a response using the local Llama3.1 model based on the query, PDF excerpts, and knowledge graph info."""
    if llm is None:
        # If Ollama is not available, return a summary based on the retrieved excerpts
        context_summary = "\n\n".join(pdf_excerpts[:3])  # Use top 3 excerpts
        return f"Based on the retrieved information:\n\n{context_summary}\n\n(Note: Ollama LLM is not available. This is a direct excerpt from the document. Please start Ollama to get AI-generated responses.)"
    
    # Prepare context
    context = "\n".join(pdf_excerpts)
    if kg_info:
        context += "\n\nKnowledge Graph Information:\n" + "\n".join(kg_info)
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"

    # Generate response using Llama3.1 model
    try:
        # Use invoke() method for LangChain Ollama (newer versions)
        if hasattr(llm, 'invoke'):
            response = llm.invoke(prompt)
        elif hasattr(llm, 'predict'):
            response = llm.predict(prompt)
        else:
            # Fallback: try calling directly (older versions)
            response = llm(prompt)
        
        # Ensure response is a string
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    except Exception as e:
        # If generation fails, return the relevant excerpts as a fallback
        context_summary = "\n\n".join(pdf_excerpts[:3])
        return f"Based on the retrieved information:\n\n{context_summary}\n\n(Note: LLM generation failed: {str(e)})"

def main():
    st.title("RAG Chatbot with Knowledge Graph")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load PDF content and create index (with loading indicator)
    with st.spinner("Initializing chatbot..."):
        pdf_content = load_pdf_content()
        if pdf_content is None:
            st.error("Failed to load PDF content. Please check the 'extracted_text.txt' file.")
            return
        index, content_chunks = load_or_create_index(pdf_content)
        
        # Check Neo4j availability and get knowledge graph stats
        is_connected, node_count, rel_count, samples = get_neo4j_status()
        
        # Display Neo4j status in sidebar
        with st.sidebar:
            st.header("🔗 System Status")
            
            # Neo4j Status
            if is_connected:
                st.success("✅ Neo4j: Connected")
                st.write(f"**Knowledge Graph Location:**")
                st.write(f"Database: {NEO4J_URI}")
                st.write(f"*The knowledge graph is stored in the Neo4j database*")
                st.write(f"**Knowledge Graph Statistics:**")
                st.write(f"- Nodes (Entities): {node_count}")
                st.write(f"- Relationships: {rel_count}")
                
                if samples and samples[0]:
                    with st.expander("📊 View Sample Entities"):
                        for entity in samples[0]:
                            st.write(f"- {entity}")
                    
                    if samples[1]:
                        with st.expander("🔗 View Sample Relationships"):
                            for from_ent, to_ent in samples[1]:
                                st.write(f"- {from_ent} → {to_ent}")
            else:
                st.error("❌ Neo4j: Not Connected")
                st.write("**To enable knowledge graph:**")
                st.write("1. Start Neo4j database")
                st.write("2. Set environment variables in `.env`:")
                st.code("""
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
                """)
                st.write("3. Run `python graphcreation.py` to populate the graph")
                st.write("**Note:** The knowledge graph is stored in the Neo4j database, not in files.")
            
            # Ollama Status
            if llm is None:
                st.warning("⚠️ Ollama: Not Available")
            else:
                st.success("✅ Ollama: Available")
        
        # Check Neo4j availability (main area)
        if not is_connected:
            st.info("ℹ️ Neo4j is not available. The app will work with semantic search only. To enable knowledge graph features, please start Neo4j and run graphcreation.py.")
        
        # Check Ollama availability
        if llm is None:
            st.info("ℹ️ Ollama is not running. To enable AI-generated responses:\n1. Install Ollama from https://ollama.ai\n2. Start Ollama service\n3. Pull the model: `ollama pull llama3.1`")

    # Chat interface
    user_input = st.text_input("You:", key="user_input")

    if user_input:
        with st.spinner("Generating response..."):
            relevant_excerpts = semantic_search(user_input, index, content_chunks)
            kg_info = query_knowledge_graph(user_input)
            response = generate_response(user_input, relevant_excerpts, kg_info)

        # Add user input and bot response to chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Bot:** {message}")

if __name__ == "__main__":
    main()

