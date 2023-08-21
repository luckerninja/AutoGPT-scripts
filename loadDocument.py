from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os
import getpass
import re
import requests
import openai
from langchain.docstore.document import Document


openai_key = getpass.getpass("OpenAI API Key:")
os.environ["OPENAI_API_KEY"] = openai_key
openai.api_key = openai_key

def load_document_text(url: str) -> str:
    # Extract the document ID from the URL
    match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('Invalid Google Docs URL')
    doc_id = match_.group(1)

    # Download the document as plain text
    response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
    response.raise_for_status()
    text = response.text

    return text

system = load_document_text()  # system message

database= load_document_text()  

source_chunks = []
splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0)

for chunk in splitter.split_text(database):
    source_chunks.append(Document(page_content=chunk, metadata={}))


# Initialize the embeddings model
embeddings = OpenAIEmbeddings()

# Create an index database from segmented text chunks
db = FAISS.from_documents(source_chunks, embeddings)

def insert_newlines(text: str, max_len: int = 170) -> str:
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) > max_len:
            lines.append(current_line)
            current_line = ""
        current_line += " " + word
    lines.append(current_line)
    return " ".join(lines)

def answer_index(system, topic, search_index, temp=1, verbose=0):

    # Search for relevant segments from the knowledge base
    docs = search_index.similarity_search(topic, k=4)
    if verbose: print('\n ===========================================: ')
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nDocument Excerpt #{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    if verbose: print('message_content :\n ======================================== \n', message_content)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Document with information for the response to the client: {message_content}\n\nClient's question: \n{topic}"}
    ]

    if verbose: print('\n ===========================================: ')

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temp
    )
    answer = insert_newlines(completion.choices[0].message.content)
    return answer  # returns the answer

temperature = 1        # enter the desired value here
verbose = 1            # enter the desired value here

# user's question
topic = " "            # enter the desired value here

ans = answer_index(system, topic, db)
print(ans)