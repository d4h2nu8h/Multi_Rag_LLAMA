import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Load environment variables from .env (if needed for API keys)
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMA_PARSE_API_KEY")
if not api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY or LLAMA_PARSE_API_KEY environment variable is required. Please set it in your .env file or environment.")

# Set up the parser with LlamaParse
parser = LlamaParse(result_type="text", api_key=api_key)  # Extract plain text

# Use SimpleDirectoryReader to parse the Pdf file
file_extractor = {".pdf": parser}
pdf_path = 'Dhanush_Resume_Core.pdf'  # Replace with your actual PDF path
documents = SimpleDirectoryReader(input_files=[pdf_path], file_extractor=file_extractor).load_data()

# Combine all extracted text from documents
extracted_text = "\n".join([doc.text for doc in documents])

# Save the extracted text into extracted_text.txt
output_path = 'extracted_text.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(extracted_text)

print(f"Text and tables have been saved to {output_path}")
