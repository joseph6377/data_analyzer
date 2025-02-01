# Enterprise Data Analyzer

Enterprise Data Analyzer is a Streamlit-based application designed to help you explore and analyze CSV data in an interactive way. With a built-in chat interface powered by Groq's API, you can ask natural language questions about your data and get executable Python code along with visualizations using Plotly.

## Features

- **Interactive Data Analysis:** Upload your CSV file and ask questions about your data.
- **Natural Language Queries:** Use plain English commands to generate Python code for data analysis.
- **Visualization Support:** Automatically create visualizations (like charts and graphs) using Plotly for visual queries.
- **Dark Theme:** Enjoy a custom dark theme compatible with modern interfaces.
- **Error Handling:** Automated retries and code corrections ensure robust analysis.

## Prerequisites

- Python 3.8+
- A valid Groq API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/enterprise-data-analyzer.git
   cd enterprise-data-analyzer
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

To start the application, simply run:
```bash
streamlit run sales_analyzer_app.py
```

Once the app is running:

1. Upload your CSV file using the sidebar.
2. Interact with the chat interface by asking natural language questions about your data.
3. View the generated analysis code, textual output, and interactive Plotly visualizations.
4. Use the "Clear History" button to reset the chat if needed.

## Example CSV Format

```
Product Name,Quantity Sold,Total Revenue,Date
Product A,100,5000,2024-01-01
Product B,150,7500,2024-01-01
```

## Project Structure

- **sales_analyzer_app.py:** Main Streamlit application.
- **requirements.txt:** Python dependencies.
- **.env:** Environment variables file (not committed to version control).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.