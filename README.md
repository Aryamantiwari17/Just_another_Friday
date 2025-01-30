# Just Another Friday

A sophisticated customer feedback analysis system built with Python, leveraging AI to process, analyze, and respond to customer feedback automatically.

## Features

- Sentiment analysis of customer feedback
- Automatic response generation based on feedback content
- FAQ integration and similarity search
- Support for multiple product categories
- Web interface for easy interaction
- Detailed aspect-based sentiment breakdown
- Real-time feedback processing

## Technologies Used

- Flask web framework
- Groq LLM API for natural language processing
- Langchain for LLM integration
- HuggingFace embeddings for semantic search
- ChromaDB for vector storage
- Pydantic for data validation
- Bootstrap for frontend styling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yaryamantiwari17/just-another-friday.git
cd just-another-friday
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add:
```
GROQ_API_KEY=your_groq_api_key
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Navigate to `http://localhost:5000` in your web browser

3. Initialize the system:
   - Enter your company name
   - Choose FAQ input method (direct input or JSON file)
   - Enter product categories
   - Click "Initialize System"

4. Start analyzing feedback:
   - Enter customer feedback in the text area
   - Click "Analyze Feedback"
   - View detailed analysis results and generated responses

## Project Structure

```
├── app.py                 # Flask application entry point
├── utils/
│   └── feedback_utils.py  # Core functionality
├── templates/
│   ├── base.html         # Base template
│   ├── index.html        # System initialization
│   ├── feedback.html     # Feedback input
│   └── results.html      # Analysis results
├── static/
│   └── styles.css        # Custom styles
└── requirements.txt      # Project dependencies
```

## Features in Detail

### Sentiment Analysis
- Overall sentiment detection
- Aspect-based sentiment breakdown
- Mixed sentiment detection
- Score-based sentiment intensity

### Response Generation
- Context-aware responses
- Professional and empathetic tone
- Solution-focused suggestions
- Integration with FAQ knowledge base

### FAQ Integration
- Semantic similarity search
- Relevant FAQ matching
- Support for direct input and JSON file loading

## Output

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Groq's LLM API
- Powered by Langchain framework
- Uses HuggingFace's powerful embeddings
- ChromaDB for efficient vector storage
