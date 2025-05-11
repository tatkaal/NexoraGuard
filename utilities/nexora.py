#!pip install langchain langchain-chroma langchain-ollama chromadb ollama spacy pandas 
#!python -m spacy download en_core_web_md

# ## 1  Imports & Global Config
import json, re, pandas as pd, csv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

import spacy
nlp = spacy.load('en_core_web_md')

# configure ollama model & embeddings
MODEL = 'qwen3:1.7b'
embeddings = OllamaEmbeddings(model=MODEL)

# ## 2  Fix the faqs.csv dataset since there are questions separated by comma and not enclosed in quotes
input_path = Path("data/faqs.csv")
output_path = Path("data/faqs_cleaned.csv")

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

    header = next(reader)
    writer.writerow(header)

    for row in reader:
        if len(row) > 3:
            # Fix rows with a comma in the question (merge columns until we have 3)
            question_parts = row[:-2]  # Everything except last 2 fields
            question = ",".join(question_parts).strip()
            answer = row[-2].strip()
            category = row[-1].strip()
            writer.writerow([question, answer, category])
        else:
            writer.writerow(row)

# ## 2  Load & Transform Knowledge‑Base Documents
DATA_DIR = Path('data')  # adjust if different
products_path = DATA_DIR / 'products_occupation.json'
faqs_path = DATA_DIR / 'faqs_cleaned.csv'

# Load products
with open(products_path, 'r', encoding='utf-8') as f:
    products_data = json.load(f)['products']

# Flatten products into plain‑text docs
product_docs = []
for p in products_data:
    text = f"""
    Product Name: {p['name']}
    Description: {p['description']}
    Target Industries: {', '.join(p['target_industries'])}
    Coverage Options: {', '.join(p['coverage_options'])}
    Premium Range: {p['premium_range']['min']}-{p['premium_range']['max']} {p['premium_range']['currency']}
    Excess Range: {p['excess_range']['min']}-{p['excess_range']['max']} {p['excess_range']['currency']}
    Key Features: {', '.join(p['key_features'])}
    Exclusions: {', '.join(p['exclusions'])}
    Unique Selling Points: {', '.join(p['unique_selling_points'])}
    Required Documents: {', '.join(p['required_documents'])}
    """.strip()
    product_docs.append(Document(page_content=text, metadata={'type': 'product', 'name': p['name']}))


# Load FAQs
faqs_df = pd.read_csv(faqs_path, quotechar='"')

# ✅ Validate the structure of the CSV
assert set(faqs_df.columns) == {'question', 'answer', 'category'}, "Unexpected CSV format"

faq_docs = [
    Document(
        page_content=f"Question: {row.question}\nAnswer: {row.answer}",
        metadata={'type': 'faq', 'category': row.category}
    )
    for _, row in faqs_df.iterrows()
]

# Combine all documents
documents = product_docs + faq_docs
print(f'Total documents: {len(documents)}')

# ## 3  Create / Reload Chroma Vector Store
# Set up vector store
VECTOR_DIR = 'nexora_chroma'

if Path(VECTOR_DIR).exists():
    vector_store = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
    print('Loaded existing vector store.')
else:
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )
    print('Built new vector store →', VECTOR_DIR)

# ## 4  Lightweight Intent Recognition & Entity Extraction
INTENT_PATTERNS = {
    'claims': r'\b(claim|lodg(e|ing)|damage)\b',
    'coverage': r'\b(cover(ed|age)|policy limit|exclusion|excess)\b',
    'products': r'\bwhat is [A-Za-z ]+ insurance|types? of insurance\b',
    'pricing': r'\b(cost|price|premium|fee)\b',
    'account': r'\b(login|account|certificate of currency|policy documents|amend)\b',
}

PRODUCT_NAMES = [p['name'].lower() for p in products_data]

def classify_intent(query:str):
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, query, re.IGNORECASE):
            return intent
    return 'general'

def extract_entities(query:str):
    doc = nlp(query)
    products = [p for p in PRODUCT_NAMES if p in query.lower()]
    industries = [ent.text for ent in doc.ents if ent.label_=='ORG' or ent.label_=='NORP']
    return {'products': products, 'industries': industries}

print(classify_intent('How do I lodge a claim?'))
print(extract_entities('Do you cover Architecture firms for Professional Indemnity?'))

# ## 5  Build Retrieval‑Augmented QA Chain
llm = ChatOllama(model=MODEL, temperature=0, num_ctx=10000)

prompt_template = (
    "You are NexoraBot, a helpful and knowledgeable customer-service assistant for an Australian SME insurance broker. "
    "Use ONLY the provided context to answer. If unsure, say you don't know but can escalate to a human agent. "
    "Answer in a friendly, concise manner and, where relevant, suggest next steps inside Nexora's platform.\n\n"
    "### Context\n{context}\n\n### Question\n{question}\n\n### Answer\n"
)

prompt = ChatPromptTemplate.from_template(prompt_template)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={'k':4}),
    chain_type_kwargs={'prompt': prompt}
)

# ### Quick Test
question = 'What does Professional Indemnity cover?'
print(rag_chain.run(question))
print(rag_chain.invoke({'query': question}))

