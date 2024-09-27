# import libraries
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # add your OpenAI API Key
# add your document path below
DOC_PATH = "../alphabet_10K_2022.pdf"
CHROMA_PATH = "your_db_name" 

# ----- Data Indexing Process -----

# load your pdf doc
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

# split the doc into smaller chunks i.e. chunk_size=500
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

# get OpenAI Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# embed the chunks as vectors and load them into the database
db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

# ----- Retrieval and Generation Process -----

# this is an example of a user question (query)
query = 'what are the top risks mentioned in the document?'

# retrieve context - top 5 most relevant (closests) chunks to the query vector 
# (by default Langchain is using cosine distance metric)
docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

# generate an answer based on given user query and retrieved context information
context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

# you can use a prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

# load retrieved context and user query in the prompt template
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query)

# call LLM model to generate the answer based on the given context and query
model = ChatOpenAI()
response_text = model.predict(prompt)

"""
Generated response:

The top risks mentioned in the provided context are:
1. Decrease in investment value
2. Low adoption of products and services
3. Disruptions from factors like changes, terrorist attacks, or natural disasters
4. Exposure of trade secrets and potential legal or financial issues
5. Reputational, financial, and regulatory risks
6. Platform abuse and misuse of user data
7. Errors or weaknesses causing service outages or failures
8. Risks tied to international operations.
"""
Ra