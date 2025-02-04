import streamlit as st
import pandas as pd
import os
from groq import Groq
# from dotenv import load_dotenv  # Remove dotenv import
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import re

# Remove loading environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for API key if it doesn't exist
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Set page config
st.set_page_config(
    page_title="Enterprise Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Add custom CSS for light theme
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F0F0F0;
        border-right: 1px solid #CCCCCC;
        padding: 1rem;
    }
    
    /* Headers */
    h1 {
        color: #000000 !important;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600 !important;
        font-size: 2rem !important;
    }
    
    h2, h3 {
        color: #000000 !important;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 500 !important;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #E0E0E0;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .assistant-message {
        background-color: #F0F0F0;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #F8F8F8 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
        padding: 1rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #E0E0E0 !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
    }
    
    .stButton > button:hover {
        background-color: #D0D0D0 !important;
        border: 1px solid #BBBBBB !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #E0E0E0 !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
    }
    
    /* Success/Info/Warning messages */
    .stSuccess, .stInfo, .stWarning {
        background-color: #E0E0E0 !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Status messages */
    .stStatus {
        background-color: #E0E0E0 !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
    }
    
    /* Plotly chart background */
    .js-plotly-plot {
        background-color: #FFFFFF !important;
    }
    .plotly {
        background-color: #FFFFFF !important;
    }
    
    /* Markdown text */
    .markdown-text-container {
        color: #000000 !important;
    }
    
    /* Links */
    a {
        color: #1E90FF !important;
    }
    
    /* Divider */
    hr {
        border-color: #CCCCCC !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file):
    """Load and cache the uploaded CSV data"""
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        # Fallback to latin1 encoding if UTF-8 fails
        return pd.read_csv(uploaded_file, encoding='latin1')

def extract_code_block(text):
    """Extract code block from LLM response"""
    import re
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def get_code_from_groq(meta, sample_data, user_query, groq_api_key):
    """Get Python code from Groq API based on user's question"""
    client = Groq(api_key=groq_api_key)
    
    # Check if visualization is requested
    needs_viz = any(word in user_query.lower() for word in ['visualize', 'plot', 'graph', 'chart', 'show', 'display'])
    
    system_message = {
        "role": "system",
        "content": (
            "You are an expert data analyst and Python programmer. "
            "Generate only the executable Python code to answer the given question. "
            "The code should use pandas and print clear, formatted results. "
            "The DataFrame is already loaded and available as 'df'. "
            "For visualization requests, use Plotly (imported as px and go) and return the figure object. "
            "For non-visualization requests, print the results. "
            "Do not include explanations or file loading code. "
            "If creating a visualization:\n"
            "1. Use appropriate chart types based on the data and question\n"
            "2. Include proper titles, labels, and formatting\n"
            "3. Use a color scheme that works well with light theme\n"
            "4. Store the figure in a variable named 'fig'\n"
        )
    }
    
    user_message = {
        "role": "user",
        "content": (
            f"Here is the CSV metadata and a few sample rows:\n"
            f"Metadata: {meta}\n"
            f"Sample Data: {sample_data}\n\n"
            f"Question: {user_query}\n\n"
            "Please generate only the executable Python code that answers the question. "
            "The DataFrame is already loaded as 'df'. "
            f"{'Create a visualization using Plotly.' if needs_viz else 'Print the results clearly.'}"
        )
    }
    
    messages = [system_message, user_message]
    
    completion = client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=messages,
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=False,
        stop=None,
    )
    
    response = completion.choices[0].message.content
    return extract_code_block(response), needs_viz

def extract_meta_and_sample(df, num_samples=5):
    """Extract metadata and sample rows from DataFrame"""
    meta = {
        "columns": list(df.columns),
        "total_rows": len(df)
    }
    sample_data = df.head(num_samples).to_dict('records')
    return meta, sample_data

def execute_code(code, df):
    """Execute the generated code and capture its output"""
    import sys
    from io import StringIO
    import contextlib

    # Create string buffer to capture output
    output = StringIO()
    
    # Redirect stdout to our buffer
    with contextlib.redirect_stdout(output):
        try:
            # Create a globals dict with the DataFrame and visualization libraries
            globals_dict = {
                'pd': pd,
                'df': df,
                'px': px,
                'go': go
            }
            exec(code, globals_dict)
            
            # Check if a figure was created
            fig = globals_dict.get('fig')
            return output.getvalue(), None, fig
        except Exception as e:
            return None, str(e), None

def display_chat_history():
    """Display the chat history with proper formatting"""
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f'<div class="user-message"><strong>💬 You</strong><br>{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>🤖 AI Assistant</strong><br>{content}</div>', unsafe_allow_html=True)
            
            # Display code, results, and visualizations if present
            if "code" in message:
                with st.expander("View Analysis Code"):
                    st.code(message["code"], language="python")
            if "result" in message and message["result"]:
                st.text(message["result"])
            if "figure" in message and message["figure"]:
                st.plotly_chart(message["figure"], use_container_width=True)

def get_corrected_code_from_groq(error_message, original_code, user_query, meta, sample_data, groq_api_key):
    """Get corrected code from Groq API based on the error"""
    client = Groq(api_key=groq_api_key)
    
    system_message = {
        "role": "system",
        "content": (
            "You are an expert Python programmer and data analyst. "
            "The previous code generated an error. Please fix the code to handle the error. "
            "Generate only the corrected executable Python code. "
            "The DataFrame is already loaded as 'df'. "
            "For visualization requests, use Plotly (imported as px and go) and return the figure object. "
            "Do not include explanations or file loading code."
        )
    }
    
    user_message = {
        "role": "user",
        "content": (
            f"Original question: {user_query}\n\n"
            f"Original code:\n```python\n{original_code}\n```\n\n"
            f"Error message:\n{error_message}\n\n"
            f"DataFrame metadata: {meta}\n"
            f"Sample data: {sample_data}\n\n"
            "Please provide the corrected code that fixes this error."
        )
    }
    
    messages = [system_message, user_message]
    
    completion = client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=messages,
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=False,
        stop=None,
    )
    
    response = completion.choices[0].message.content
    return extract_code_block(response)

def try_execute_with_retries(code, df, user_query, meta, sample_data, groq_api_key, max_retries=3):
    """Try to execute code with multiple retries on error"""
    attempt = 1
    last_error = None
    last_code = code
    
    while attempt <= max_retries:
        # Execute the current code
        result, error, fig = execute_code(last_code, df)
        
        if not error:
            return result, None, fig, last_code, attempt
            
        if attempt < max_retries:
            # Add error message to chat
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Attempt {attempt}: I encountered an error. Let me try again...\nError: {error}",
                "code": last_code
            })
            
            # Get corrected code
            last_code = get_corrected_code_from_groq(
                error, last_code, user_query, meta, sample_data, groq_api_key
            )
            
        last_error = error
        attempt += 1
    
    return None, f"After {max_retries} attempts, I still couldn't fix the error: {last_error}", None, last_code, attempt - 1

def main():
    # Simple header without image
    st.title("Enterprise Data Analyzer")
    
    # Sidebar for API key, example questions, and tips
    with st.sidebar:
        st.markdown("## Configuration")
        groq_api_key = st.text_input(
            "Enter your Groq API Key", 
            type="password",
            help="Obtain your Groq API key by signing up at [Groq](https://console.groq.com/keys)",
            key="api_key_input",
            placeholder="Enter your API key here"
        )
        
        # Add model selection dropdown
        selected_model = st.selectbox(
            "Select Groq Model",
            options=[
                "deepseek-r1-distill-llama-70b",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768",
                "llama-3.3-70b-versatile",
                "llama-3.3-70b-specdec", 
                "llama-3.1-8b-instant",
                "llama-guard-3-8b",
                "gemma2-9b-it"
            ],
            index=0,
            help="Choose which LLM model to use for analysis",
            key="selected_model"
        )
        
        st.markdown("### Follow these steps:")
        st.markdown("""
        1. [Create your own API key](https://console.groq.com/keys)  
        2. Enter your API key above  
        3. Select the model you want to use  
        4. Upload your CSV file  
        5. Check the data preview  
        6. Ask a question about your data  
        7. See the analysis!  
        """)
        
        with st.expander("Tips",expanded=True):
            st.markdown("""
            - Use natural language to ask questions
            - Request visualizations using "show", "plot", "visualize"
            - Click charts to zoom or pan
            - Hover over data points for details
            """)
    
    # File uploader in main section
    st.markdown("### File Source")
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type="csv",
        help="Limit 200MB per file • CSV"
    )
    
    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)
        st.success(f"Loaded {len(df):,} rows")
        
        # Data preview
        with st.expander("Preview Data", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
        
        # Clear chat button
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("---")
    
    # Main chat interface
    if uploaded_file is not None and groq_api_key:
        try:
            # Display chat history
            display_chat_history()
            
            # Get metadata and sample data
            meta, sample_data = extract_meta_and_sample(df)
            
            # Chat input
            user_query = st.chat_input("Ask anything about your data...")
            
            if user_query:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_query
                })
                
                # Display user query immediately
                st.markdown(f'<div class="user-message"><strong>💬 You</strong><br>{user_query}</div>', unsafe_allow_html=True)
                
                # Analysis progress using spinner
                with st.spinner("Analyzing data..."):
                    # Get initial code from Groq
                    code, needs_viz = get_code_from_groq(meta, sample_data, user_query, groq_api_key)
                    
                    # Try to execute with retries
                    result, error, fig, final_code, attempts = try_execute_with_retries(
                        code, df, user_query, meta, sample_data, groq_api_key
                    )
                
                if error:
                    st.error("Analysis failed")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error,
                        "code": final_code
                    })
                else:
                    st.success("Analysis complete")
                    let_response = "Here's what I found:"
                    if attempts > 1:
                        let_response = f"After {attempts} attempts, here's what I found:"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": let_response,
                        "code": final_code,
                        "result": result,
                        "figure": fig
                    })
                
                # Rerun to update chat display
                st.rerun()
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Add subtle footer
    st.markdown("""
    <div style='position: fixed; bottom: 10px; right: 10px; opacity: 0.5; font-size: 0.8em;'>
        Built by Joseph Thekkekara <i>(or by an overworked AI? 😉)</i>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
