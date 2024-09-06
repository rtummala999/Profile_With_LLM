from langchain.vectorstores import Chroma
import os
import json
from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from functools import wraps
from werkzeug.utils import secure_filename
import requests
import openai
from openai import OpenAI
from dotenv import load_dotenv
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Set a secret key for sessions

# Load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv('OPENAI_API_KEY'),
)
# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load documents and initialize the vector database
def load_documents_and_initialize_db():
    try:
        # Path to the directory containing the documents
        docs_path = 'docs'
        documents = []

        # Iterate over all PDF files in the directory
        for filename in os.listdir(docs_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(docs_path, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        # Create embeddings and initialize the vector database
        embeddings = OpenAIEmbeddings()
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        return db
    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
        return None

# Initialize the vector database
db = load_documents_and_initialize_db()

# Check if database was successfully initialized
if db is None:
    raise ValueError("Failed to initialize the vector database. Please check the document path and format.")

# Configure the language model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Custom prompt template
# Create a conversational retrieval chain with memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    memory=memory
)

# File to store conversations
CONVERSATION_FILE = 'conversations.json'

# Save conversation to a file
def save_conversation(question, answer):
    try:
        if os.path.exists(CONVERSATION_FILE):
            with open(CONVERSATION_FILE, 'r') as f:
                conversations = json.load(f)
        else:
            conversations = []

        conversations.append({"question": question, "answer": answer})

        with open(CONVERSATION_FILE, 'w') as f:
            json.dump(conversations, f, indent=4)
    except Exception as e:
        print(f"Failed to save conversation: {e}")

# Admin login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function



def scrape_jina_ai(url: str) -> str:
    response = requests.get("https://r.jina.ai/" + url)
    return response.text

def get_llm_response(content, prompt):
    """
    Uses OpenAI's LLM to generate a response in JSON format.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant please respond if they call you ram or ramgopal or tummala  "},
            {"role": "user", "content": f"consider you are ramgopal tummala. Please provide the answer to the following prompt based on the given content: {prompt}\n\nContent: {content}  please respond frendly  and try to give small answers in 2 lines   "}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    try:
        custom_prompt = f"You are an agent, take this prompt: '{question}' and give results in 2 lines."
        result = qa_chain({"question": custom_prompt})

        answer = result['answer']

        # Initialize token usage variables
        total_tokens = 0
        completion_tokens = 0
        prompt_tokens = 0
        cost = 0

        # Check if the API response contains token usage details
        if 'completion_tokens' in result and 'prompt_tokens' in result:
            completion_tokens = result['completion_tokens']
            prompt_tokens = result['prompt_tokens']
            total_tokens = completion_tokens + prompt_tokens

            # Calculate the cost based on token usage
            COST_PER_1000_TOKENS = 0.03  # Adjust based on your model
            cost = (len(total_tokens) / 1000) * COST_PER_1000_TOKENS

        save_conversation(question, answer)  # Save the conversation
    except Exception as e:
        answer = f"An error occurred: {e}"
        cost = 0  # Set cost to 0 if there's an error
        total_tokens = 10

    return jsonify({'answer': answer, 'cost': f"${cost:.4f}", 'tokens_used': total_tokens})


# Admin login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Simple authentication (replace with a more secure method)
        if username == 'admin' and password == 'adminpassword':
            session['admin_logged_in'] = True
            return redirect(url_for('admin'))
        else:
            return "Invalid credentials"
    return render_template('login.html')

# Admin route to view conversations and upload documents
@app.route('/admin')
@login_required
def admin():
    try:
        if os.path.exists(CONVERSATION_FILE):
            with open(CONVERSATION_FILE, 'r') as f:
                conversations = json.load(f)
        else:
            conversations = []
    except Exception as e:
        conversations = []
        print(f"Failed to load conversations: {e}")
    return render_template('admin.html', conversations=conversations)


@app.route('/json_bot', methods=['GET', 'POST'])
def json_bot():
    if request.method == 'POST':
        prompt = request.form['prompt']
        url = request.form['url']

        # Scrape the content from the given URL
        content = scrape_jina_ai(url)

        try:
            # Get the LLM response based on the content and prompt
            final_content = get_llm_response(content, prompt)

            # Format the JSON output for display
            final_content = final_content.replace("```json", "").replace("```", "")
            data = json.loads(final_content)
            formatted_json = json.dumps(data, indent=4)
        except json.JSONDecodeError:
            formatted_json = "The response is not in valid JSON format."

        return jsonify({'json_response': formatted_json})
    else:
        # If it's a GET request, you can render the form page
        return render_template('json_bot.html')


# Route for uploading a document
@app.route('/upload_document', methods=['POST'])
@login_required
def upload_document():
    if 'document' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
    file = request.files['document']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('docs', filename)
        file.save(file_path)
        try:
            # Load the new document and update the vector database
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)
            # Here we add the documents directly to the vector store
            db.add_documents(docs)
            return jsonify({"success": True, "message": "File uploaded successfully", "document": filename})
        except Exception as e:
            return jsonify({"success": False, "message": f"Failed to process the document: {e}"}), 500
    else:
        return jsonify({"success": False, "message": "File not allowed"}), 400




@app.route('/view_documents')
@login_required
def view_documents():
    try:
        docs_path = 'docs'
        documents = [f for f in os.listdir(docs_path) if os.path.isfile(os.path.join(docs_path, f))]
    except Exception as e:
        documents = []
        print(f"Failed to load documents: {e}")
    return render_template('view_documents.html', documents=documents)


@app.route('/delete_document/<filename>', methods=['POST'])
@login_required
def delete_document(filename):
    try:
        # Delete the file from the filesystem
        file_path = os.path.join('docs', filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            return jsonify({"success": False, "message": f"File not found: {filename}"}), 404

        # Rebuild the vector database excluding the deleted file
        documents = []
        for fname in os.listdir('docs'):
            if fname != filename and fname.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join('docs', fname))
                documents.extend(loader.load())

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        # Recreate the vector database with remaining documents
        global db
        embeddings = OpenAIEmbeddings()
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)

        # Reinitialize the QA chain with the updated vector store
        global qa_chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(),
            memory=memory
        )

        return jsonify({"success": True, "message": "Document deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {e}"}), 500


# Admin logout route
@app.route('/logout')
@login_required
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
