import json
import re
import pandas as pd
import csv
from pathlib import Path
import shutil # For cleaning up vector store if needed

# LangChain components
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # For streaming
from langchain.callbacks.manager import CallbackManager

# NLP
import spacy
from spacy.matcher import Matcher, PhraseMatcher

# Streamlit (for the web app part)
import streamlit as st

# --- Global Configurations ---
NEXORA_MODEL_NAME = 'qwen3:1.7b'

EMBEDDING_MODEL_FOR_OLLAMA = NEXORA_MODEL_NAME

DATA_DIR = Path('data') # Create a dedicated directory for this project's data
VECTOR_STORE_DIR = Path('nexora_chroma_vector_store')
PRODUCTS_FILE = DATA_DIR / 'products_occupation.json'
FAQS_FILE = DATA_DIR / 'faqs.csv'
FAQS_CLEANED_FILE = DATA_DIR / 'faqs_cleaned.csv'
CHAT_CONVERSATIONS_FILE = DATA_DIR / 'chat_conversations.json'

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize SpaCy
nlp = spacy.load('en_core_web_md')


# Ollama settings
llm = ChatOllama(
    model=NEXORA_MODEL_NAME,
    temperature=0.0,  # For factual, consistent responses
    num_ctx=4096,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), # See model output streaming
)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_FOR_OLLAMA)


##utility functions
def test_embedding_model():
    print("→ Testing embedding connectivity…")
    try:
        sample = embeddings.embed_documents(["hello world"])
        print("Embedding OK:", sample[:5])
    except Exception as e:
        print("Embedding error:", e)
        raise

# ## 2. Data Ingestion & Preprocessing: The Knowledge Foundation
# 
# A RAG system is only as good as its knowledge base. We'll meticulously prepare Nexora's data.
# 
# ### 2.1. Product Data (`products_occupation.json`)
# This file contains detailed information about Nexora's insurance products. We'll transform this structured JSON into a set of text `Document` objects, making each product's details searchable.

with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
    try:
        products_full_data = json.load(f)
        # The sample data has "products" as the key. We check if the loaded data is a dict with "products" key or directly a list.
        if isinstance(products_full_data, dict) and 'products' in products_full_data:
            products_data = products_full_data['products']
        elif isinstance(products_full_data, list): # If the root is already the list of products
             products_data = products_full_data
        else:
            print("Warning: 'products_occupation.json' doesn't have the expected top-level 'products' key or is not a list. Attempting to process as is if it's a list of product-like objects.")
            products_data = [] # Or handle error more gracefully
            if isinstance(products_full_data, list) and len(products_full_data) > 0 and 'name' in products_full_data[0]:
                 products_data = products_full_data

        if not products_data: # If after checks, products_data is empty
            raise ValueError("No product data found or products_data is empty. Please check the format of products_occupation.json.")

    except json.JSONDecodeError as e:
        print(f"Error decoding {PRODUCTS_FILE}: {e}")
        print("Please ensure the JSON is valid.")
        products_data = []
    except ValueError as e:
        print(e)
        products_data = []


product_docs = []
if products_data: # Proceed only if product_data was loaded successfully
    for p in products_data:
        # Construct a comprehensive text description for each product
        # This text will be embedded and stored in the vector database
        # Using f-strings and careful formatting for readability and LLM parsing
        details_list = []
        details_list.append(f"Product Name: {p.get('name', 'N/A')}")
        details_list.append(f"Description: {p.get('description', 'N/A')}")
        
        target_industries = p.get('target_industries', [])
        if target_industries: # Check if the list is not empty
            details_list.append(f"Target Industries: {', '.join(target_industries)}")
        
        coverage_options = p.get('coverage_options', [])
        if coverage_options:
            details_list.append(f"Coverage Options: {', '.join(map(str,coverage_options))}") # Ensure all are strings

        premium_range = p.get('premium_range', {})
        if premium_range: # Check if dict is not empty
            details_list.append(f"Premium Range: {premium_range.get('min', 'N/A')}-{premium_range.get('max', 'N/A')} {premium_range.get('currency', 'AUD')}")

        excess_range = p.get('excess_range', {})
        if excess_range:
            details_list.append(f"Excess Range: {excess_range.get('min', 'N/A')}-{excess_range.get('max', 'N/A')} {excess_range.get('currency', 'AUD')}")

        key_features = p.get('key_features', [])
        if key_features:
            details_list.append(f"Key Features: {'; '.join(key_features)}") # Semicolon for lists within text

        exclusions = p.get('exclusions', [])
        if exclusions:
            details_list.append(f"Exclusions: {'; '.join(exclusions)}")

        usps = p.get('unique_selling_points', [])
        if usps:
            details_list.append(f"Unique Selling Points: {'; '.join(usps)}")
        
        required_docs = p.get('required_documents', [])
        if required_docs:
             details_list.append(f"Required Documents for Quote/Application: {'; '.join(required_docs)}")

        text = "\n".join(details_list)
        
        product_docs.append(Document(
            page_content=text.strip(),
            metadata={
                'source_type': 'product_data',
                'product_name': p.get('name', 'Unknown Product'),
                'product_id': p.get('product_id', 'N/A')
            }
        ))
    print(f"Successfully processed {len(product_docs)} products into documents.")
    if product_docs:
      print("\nSample Product Document Content:")
      print(product_docs[0].page_content)
      print(f"\nSample Product Document Metadata: {product_docs[0].metadata}")
else:
    print("No product data loaded to process.")

# ### 2.2. FAQs Data (`faqs.csv`)
# CSV Cleaning (as provided, robust for unquoted commas in question field)
if FAQS_FILE.exists():
    with open(FAQS_FILE, "r", encoding="utf-8") as infile, \
         open(FAQS_CLEANED_FILE, "w", newline="", encoding="utf-8") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

        try:
            header = next(reader)
            writer.writerow(header)

            for row in reader:
                if not row: continue # Skip empty rows
                if len(row) > 3: # Question field likely contains unquoted commas
                    question_parts = row[:-2]
                    question = ",".join(question_parts).strip('" ') # Join and strip quotes/spaces
                    answer = row[-2].strip('" ')
                    category = row[-1].strip('" ')
                    writer.writerow([question, answer, category])
                elif len(row) == 3:
                     writer.writerow([col.strip('" ') for col in row])
                else:
                    print(f"Skipping malformed FAQ row: {row}") # Log malformed rows
        except StopIteration:
            print(f"Warning: {FAQS_FILE} might be empty or just has a header.")
            # Create an empty cleaned file with header if original was empty/header-only
            if 'header' not in locals() or not header: header = ['question', 'answer', 'category']
            writer.writerow(header)


    faqs_df = pd.read_csv(FAQS_CLEANED_FILE)
    # Validate structure
    expected_cols = {'question', 'answer', 'category'}
    if not expected_cols.issubset(faqs_df.columns):
        print(f"Error: Cleaned FAQ CSV {FAQS_CLEANED_FILE} has unexpected columns: {faqs_df.columns}. Expected: {expected_cols}")
        faq_docs = []
    else:
        faq_docs = [
            Document(
                page_content=f"Question: {row.question}\nAnswer: {row.answer}",
                metadata={
                    'source_type': 'faq',
                    'category': row.category,
                    'question': row.question # Adding original question to metadata for potential exact matching
                }
            )
            for _, row in faqs_df.iterrows() if pd.notna(row.question) and pd.notna(row.answer) # Handle NaN values
        ]
    print(f"Successfully processed {len(faq_docs)} FAQs into documents.")
    if faq_docs:
        print("\nSample FAQ Document Content:")
        print(faq_docs[0].page_content)
        print(f"\nSample FAQ Document Metadata: {faq_docs[0].metadata}")
else:
    print(f"{FAQS_FILE} not found. Skipping FAQ processing.")
    faq_docs = []

# ### 2.3. Chat Conversations Data (`chat_conversations.json`) - For Analysis & Future Fine-Tuning
# This dataset is primarily earmarked for a future fine-tuning strategy (discussed later). However, analyzing it can provide insights into common user intents, entities, and phrasing, which can inform our current RAG prompt engineering and intent classification.
# 
# For now, we will load it and outline its potential. We could, for example, extract Q&A pairs from successful conversations to augment our knowledge base if gaps are found.

chat_conversation_docs = [] # Not directly adding to KB for RAG now, but for analysis
if CHAT_CONVERSATIONS_FILE.exists():
    with open(CHAT_CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        try:
            chat_data = json.load(f)
            print(f"Loaded {len(chat_data)} chat conversations from {CHAT_CONVERSATIONS_FILE}.")
            # Example: Extracting simple Q&A patterns for potential KB augmentation or intent examples
            # This is a conceptual step; full processing would require more sophisticated logic.
            potential_new_qas = []
            for conv in chat_data:
                messages = conv.get('messages', [])
                for i in range(len(messages) - 1):
                    if messages[i]['role'] == 'customer' and messages[i+1]['role'] == 'agent':
                        # Simple heuristic: if customer asks and agent answers immediately
                        q = messages[i]['text']
                        a = messages[i+1]['text']
                        # Further filtering could be applied (e.g., length, keywords)
                        if len(q) > 10 and len(a) > 10 and "?" in q: # Basic filter
                             potential_new_qas.append({"question": q, "answer": a, "source_conversation_id": conv.get("id")})
            
            print(f"Identified {len(potential_new_qas)} potential Q&A pairs from chat logs for review.")
            if potential_new_qas:
                print("Example potential Q&A:", potential_new_qas[0])
            
            # We are NOT adding these to `documents` for now to keep the KB focused on curated content (products, FAQs).
            # These would be candidates for manual review and addition to FAQs, or for fine-tuning datasets.

        except json.JSONDecodeError as e:
            print(f"Error decoding {CHAT_CONVERSATIONS_FILE}: {e}")
            chat_data = []
        except Exception as e:
            print(f"An error occurred while processing {CHAT_CONVERSATIONS_FILE}: {e}")
            chat_data = []
else:
    print(f"{CHAT_CONVERSATIONS_FILE} not found. Skipping chat conversation analysis.")
    chat_data = []


# ### 2.4. Consolidating the Knowledge Base
# Combine all processed documents into a single list for vector store creation.

documents = product_docs + faq_docs
if documents:
    print(f"\nTotal documents for RAG knowledge base: {len(documents)}")
    print("Sample of combined documents (first product and first FAQ):")
    if product_docs: print(f"Product Doc Example Metadata: {product_docs[0].metadata}")
    if faq_docs: print(f"FAQ Doc Example Metadata: {faq_docs[0].metadata}")
else:
    print("No documents were processed. The knowledge base is empty. Chatbot functionality will be severely limited.")


# ## 3. Vector Store Creation: Indexing Nexora's Knowledge
# 
# We'll use ChromaDB to store vector embeddings of our documents. This allows for fast semantic search, finding the most relevant information for a user's query.
# 
# **Process:**
# 1.  Each document's content is converted into a numerical vector (embedding) using `OllamaEmbeddings` (powered by the specified `qwen3:1.7b` model or its embedding counterpart).
# 2.  These vectors are stored in ChromaDB, indexed for efficient similarity lookup.

FORCE_REBUILD_VECTOR_STORE = False # Set to True to always rebuild, False to load if exists

if FORCE_REBUILD_VECTOR_STORE and VECTOR_STORE_DIR.exists():
    print(f"Force rebuilding: Deleting existing vector store at {VECTOR_STORE_DIR}...")
    shutil.rmtree(VECTOR_STORE_DIR) # Remove the directory and its contents

if not documents:
    print("Critical: No documents available to build the vector store. Please check data loading steps.")
    vector_store = None # Or handle this case as appropriate for your application flow
else:
    if VECTOR_STORE_DIR.exists():
        print(f"Loading existing vector store from {VECTOR_STORE_DIR}...")
        vector_store = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings
        )
        print("Vector store loaded successfully.")
    else:
        print(f"Building new vector store at {VECTOR_STORE_DIR}...")
        if not VECTOR_STORE_DIR.exists(): # Ensure directory exists before Chroma tries to write to it
             VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        
        test_embedding_model()

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(VECTOR_STORE_DIR) # Chroma expects a string path
        )
        print(f"New vector store built and persisted at {VECTOR_STORE_DIR}.")

    # # Test the vector store with a sample query (if it was successfully created/loaded)
    # if vector_store:
    #     print("\nTesting vector store retrieval with 'professional indemnity details':")
    #     sample_retrieved_docs = vector_store.similarity_search("professional indemnity details", k=1)
    #     if sample_retrieved_docs:
    #         print(f"Retrieved {len(sample_retrieved_docs)} document(s).")
    #         print("Content of top retrieved document:")
    #         print(sample_retrieved_docs[0].page_content[:300] + "...") # Print snippet
    #         print(f"Metadata: {sample_retrieved_docs[0].metadata}")
    #     else:
    #         print("No documents retrieved. The vector store might be empty or the query too dissimilar.")

# ## 4. Enhanced Intent Recognition & Entity Extraction
# 
# Understanding the user's *intent* and key *entities* in their query is crucial for guiding the RAG pipeline effectively. We'll enhance the provided basic system.
# 
# **Approach:**
# *   **Intent Classification:** Using a combination of improved regex and SpaCy's `Matcher` for more robust pattern identification.
# *   **Entity Extraction:** Leveraging SpaCy's pre-trained NER and extending it with `PhraseMatcher` for domain-specific terms (like product names).
# 
# ### 4.1. Defining Intents and Entities
# We'll define key intents relevant to Nexora's customer service.

# --- Intent & Entity Configuration ---
INTENT_PATTERNS_REGEX = {
    'get_quote': r'\b(get a quote|quote for|how much for|pricing on)\b',
    'product_info': r'\b(what is|tell me about|details on|info on)\s+([A-Za-z\s]+insurance)\b|\b(compare|difference between)\b',
    'coverage_query': r'\b(cover(ed|s|age)|include|policy limit|exclusion|excess)\b',
    'claim_query': r'\b(claim|lodge|report an incident|damage|incident)\b',
    'account_management': r'\b(my account|login|password|policy document|certificate of currency|update details|amend policy)\b',
    'general_greeting': r'\b(hi|hello|hey|good morning|good afternoon)\b',
    'general_farewell': r'\b(bye|goodbye|thanks|thank you)\b',
    'ask_agent': r'\b(human|agent|person|talk to someone)\b'
}

# Extract all product names for entity matching
ALL_PRODUCT_NAMES = []
if products_data:
 ALL_PRODUCT_NAMES = [p.get('name',"").lower() for p in products_data if p.get('name')]
else:
 print("Warning: products_data not available for ALL_PRODUCT_NAMES list.")


# Initialize SpaCy Matchers
matcher = Matcher(nlp.vocab)
phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# Add product name patterns to PhraseMatcher
product_patterns = [nlp.make_doc(name) for name in ALL_PRODUCT_NAMES]
if product_patterns: # only add if list is not empty
    phrase_matcher.add("INSURANCE_PRODUCT", product_patterns)
else:
    print("No product patterns to add to PhraseMatcher.")


# Example: More complex intent patterns using SpaCy's rule-based Matcher
# Pattern for asking about exclusions for a specific product
# e.g., "What are the exclusions for Professional Indemnity?"
# [{"LOWER": "what"}, {"LOWER": "are"}, {"LOWER": "the"}, {"LOWER": "exclusions"}, {"LOWER": "for"}, {"ENT_TYPE": "INSURANCE_PRODUCT", "OP": "?"}, {"POS": "PROPN", "OP": "*"}, {"LOWER": "insurance", "OP": "?"}]
# This is illustrative; more patterns needs be added. For now, we'll focus on using the extracted product entities.

# ### 4.2. Intent and Entity Extraction Functions
# These functions will process the user's query.

def classify_intent_and_extract_entities(query: str):
    query_lower = query.lower()
    doc = nlp(query) # Process with SpaCy once

    # --- Intent Classification ---
    detected_intent = 'general_inquiry' # Default intent
    
    # 1. Regex-based (broad strokes)
    for intent_name, pattern in INTENT_PATTERNS_REGEX.items():
        if re.search(pattern, query_lower):
            detected_intent = intent_name
            # For 'product_info' via regex, try to capture the product name
            if intent_name == 'product_info':
                 match = re.search(pattern, query_lower)
                 if match and len(match.groups()) > 1 and match.group(2): # group(2) is the ([A-Za-z\s]+insurance) part
                     # Basic cleaning for the regex-captured product name
                     potential_product = match.group(2).replace("insurance", "").strip()
                     # Check if this roughly matches any known product names
                     for known_prod in ALL_PRODUCT_NAMES:
                         if potential_product in known_prod:
                             # If a specific product is mentioned with "what is...", it's likely product_info
                             break # Keep product_info intent
                 break # Stop after first regex match for simplicity here, can be prioritised.


    # --- Entity Extraction ---
    entities = {
        "products": [],
        "occupations_mentioned": [],
        "locations": [],
        "organizations": [],
        "generic_terms": [] # for terms like 'coverage', 'premium' etc.
    }

    # 1. SpaCy PhraseMatcher for Product Names
    # We apply phrase_matcher to the original casing `doc` because `nlp.make_doc` preserves casing, 
    # but the matching attribute is 'LOWER'
    phrase_matches = phrase_matcher(doc) 
    for match_id, start, end in phrase_matches:
        span = doc[start:end]
        entities["products"].append(span.text)
    
    # Remove duplicates if any product got matched multiple ways
    entities["products"] = list(set(entities["products"]))

    # 2. SpaCy pre-trained NER for other entities
    for ent in doc.ents:
        if ent.label_ == "ORG": # Organization might be relevant (e.g. competitor name, or business name context)
            entities["organizations"].append(ent.text)
        elif ent.label_ == "GPE" or ent.label_ == "LOC": # Geopolitical Entity or Location
            entities["locations"].append(ent.text)
        # Add more entity types as per the dataset

    # 3. Keyword-based for generic insurance terms (can be refined)
    # Example: extracting "coverage", "premium" even if not a formal "intent"
    if "coverage" in query_lower: entities["generic_terms"].append("coverage")
    if "premium" in query_lower: entities["generic_terms"].append("premium")
    if "excess" in query_lower: entities["generic_terms"].append("excess")
    if "claim" in query_lower: entities["generic_terms"].append("claim")
    
    # If intent is 'product_info' but no product entity found yet,
    # check if any part of the query is a product name
    # This handles cases like "Professional Indemnity" as a query directly
    if detected_intent == 'product_info' and not entities["products"]:
        for prod_name_full in ALL_PRODUCT_NAMES: # ALL_PRODUCT_NAMES are already lowercased
            if prod_name_full in query_lower:
                # Find the proper-cased version from original products_data
                for p_data in products_data:
                    if p_data.get('name',"").lower() == prod_name_full:
                        entities["products"].append(p_data.get('name'))
                        break
                if entities["products"]: break # Found one
    
    # Basic occupation identification (can be improved with a gazetteer or NER training)
    # For now, if user mentions an occupation present in "target_industries" lists (as proxy)
    # This is a simplified example; a real system might use a list of known occupations.
    if products_data: # check if products_data is available
        for product_entry in products_data:
            for industry in product_entry.get("target_industries", []):
                if industry.lower() in query_lower:
                    entities["occupations_mentioned"].append(industry)
        entities["occupations_mentioned"] = list(set(entities["occupations_mentioned"]))


    return detected_intent, entities

# # Test the enhanced functions
# test_queries = [
#     "Hi there, what is Professional Indemnity insurance?",
#     "Tell me about coverage for my consulting business.",
#     "How much does Public Liability cost for a retail shop?",
#     "I need to lodge a claim for property damage.",
#     "How do I get my certificate of currency?",
#     "What are the exclusions for Cyber Insurance?",
#     "Do you cover accountants for professional indemnity?",
#     "Thanks, bye!"
# ]

# print("--- Intent & Entity Extraction Tests ---")
# for q in test_queries:
#     intent, found_entities = classify_intent_and_extract_entities(q)
#     print(f"Query: '{q}'\n  -> Intent: {intent}\n  -> Entities: {found_entities}\n")

# ## 5. Advanced RAG Pipeline Construction
# 
# Now, we assemble the RAG chain. Key enhancements include:
# *   **Dynamic Prompting:** The prompt will be subtly adjusted or informed by the detected intent and entities.
# *   **Source Attribution:** The chain will return the source documents used to generate the answer, enabling transparency.
# *   **Context-Aware Retrieval (Conceptual):** While basic retrieval uses similarity, future enhancements could involve filtering search results based on extracted entities (e.g., specific product names) before they are sent to the LLM. For this iteration, the entities will primarily guide prompt formulation and response validation.
# 
# ### 5.1. Crafting a Persona-Driven, Dynamic Prompt
# The prompt guides the LLM's behavior. It's crucial for accuracy, tone, and staying within bounds.

BASE_PROMPT_TEMPLATE = """
You are NexoraGuard, an AI assistant for Nexora Pty Ltd, an Australian SME insurance broker.
Your primary goal is to provide accurate, helpful, and concise information based **ONLY** on the context provided below.
Do NOT use any external knowledge or make assumptions beyond this context.
/no_think

**Context from Nexora's Knowledge Base:**
{context}

**User's Question:** {question}

**Instructions for Answering:**
1.  Analyze the user's question carefully.
2.  If the provided context contains relevant information to answer the question, synthesize a response.
3.  If the context does NOT contain enough information, clearly state that you cannot answer the question with the current information and suggest they contact a human agent at Nexora for more specialized advice (e.g., "I don't have specific details on that. For more specialized advice, please contact a Nexora agent at support@nexora.com.au or call us.")
4.  Be friendly, professional, and use clear language.
5.  If the query is about a specific product and details are found, summarize them.
6.  If the query is a simple greeting, respond politely and ask how you can help.
7.  If the query is a farewell, respond politely.
8.  If the user asks to speak to a human, provide contact details for Nexora customer support.

**Answer:**
"""

# We will make this prompt slightly more dynamic in the QA function later.
# For now, setting up the RetrievalQA chain with this base.

RAG_PROMPT = PromptTemplate(
    template=BASE_PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# ### 5.2. Building the `RetrievalQA` Chain
# This chain combines retrieval from `ChromaDB` with generation by the `ChatOllama` LLM.
if vector_store: # Ensure vector_store is initialized
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" puts all retrieved docs into the context
        retriever=vector_store.as_retriever(
            search_type="similarity", # "similarity", "mmr", "similarity_score_threshold"
            search_kwargs={
                'k': 5, # Retrieve top 5 relevant documents. Tune this based on context window & desired detail.
                # 'score_threshold': 0.6 # Example: filter by relevance score (Chroma supports this with specific embedding types)
            }
        ),
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True # Crucial for transparency and debugging
    )
    print("RAG chain initialized successfully.")
else:
    print("Error: Vector store not available. RAG chain cannot be initialized.")
    rag_chain = None


# ### 5.3. Intelligent Query Processing Function
# This function will take a user query, get intent/entities, and then invoke the RAG chain.
# It can also implement pre/post-processing logic.

def ask_nexora_guard(query: str):
    if not rag_chain:
        return {"answer": "I'm sorry, but my knowledge systems are currently unavailable. Please try again later.", "sources": [], "intent": "error", "entities": {}}

    print(f"\nUser Query: {query}")
    intent, entities = classify_intent_and_extract_entities(query)
    print(f"🤖 Intent: {intent}, Entities: {entities}")

    # --- Pre-computation / Heuristics based on intent ---
    if intent == 'general_greeting':
        return {"answer": "Hello! I'm NexoraGuard, your AI assistant for Nexora insurance. How can I help you today?", "sources": [], "intent": intent, "entities": entities}
    if intent == 'general_farewell':
        return {"answer": "You're welcome! If you have more questions later, feel free to ask. Have a great day!", "sources": [], "intent": intent, "entities": entities}
    if intent == 'ask_agent':
        return {"answer": "If you'd like to speak with a human agent, you can contact Nexora support at support@nexora.com.au or call us at [Nexora Phone Number - Placeholder].", "sources": [], "intent": intent, "entities": entities}
    
    # --- Modify query or context for RAG based on intent/entities (Example) ---
    # For instance, if a specific product is mentioned, we could try to make the query more explicit
    # or add a filter to the retriever if the retriever supports metadata filtering directly.
    # For now, the prompt is generic but the LLM will use the full context of retrieved docs.
    # The current `vector_store.as_retriever` doesn't easily take dynamic metadata filters
    # in a simple way here without custom retriever logic. So, entities are mainly for prompt conditioning
    # and for potential display in UI.
    
    # If specific product is identified, add it to the question to guide LLM focus even more
    # This is a simple form of "stuffing" the query for better context to LLM.
    refined_query = query
    if entities.get("products"):
        product_names = ", ".join(entities["products"])
        # A simple refinement to ensure product name is in the question for the LLM
        if not any(prod_name.lower() in query.lower() for prod_name in entities["products"]):
             refined_query = f"{query} (specifically regarding {product_names})"
             print(f"🤖 Refined query for LLM: {refined_query}")
    
    result = rag_chain.invoke({'query': refined_query}) # Standard invocation
    
    answer = result.get('result', "I'm sorry, I encountered an issue processing your request.")
    source_docs = result.get('source_documents', [])

    # --- Post-processing (Example: Filter sources shown to user) ---
    # For now, we'll return all retrieved sources. In a prod system, you might filter them based on true relevance.
    
    # Adding a check if the answer seems generic or like "I don't know"
    # This is a simple heuristic
    if "don't know" in answer.lower() or "cannot answer" in answer.lower() or "don't have specific details" in answer.lower() or "unable to find information" in answer.lower():
        if not source_docs: # If LLM says I don't know AND no docs were relevant enough
             answer += " It seems I couldn't find relevant information in my current knowledge base. For further assistance, please contact a Nexora agent."
        else: # LLM couldn't synthesize from retrieved docs
             answer += " While I found some documents, I couldn't extract a specific answer for your query. You might want to contact a Nexora agent."
    
    return {
        "answer": answer.strip(),
        "sources": source_docs,
        "intent": intent,
        "entities": entities
    }

# ### 5.4. Quick Test of the Full Pipeline

if rag_chain: # Only test if RAG chain is set up
    print("\n--- Testing the ask_nexora_guard function ---")
    
    # Test Case 1: Product Information
    response1 = ask_nexora_guard("What is Professional Indemnity insurance?")
    # print(f"\n🤖 NexoraGuard:\n{response1['answer']}")
    print(f"Source Documents Retrieved: {len(response1['sources'])}")
    if response1['sources']:
        print(f"  Top source: {response1['sources'][0].metadata.get('source_type', 'N/A')} - {response1['sources'][0].metadata.get('product_name', response1['sources'][0].metadata.get('question', 'N/A'))}")

    # # Test Case 2: FAQ Style Question
    # response2 = ask_nexora_guard("How do I get a copy of my policy documents?")
    # print(f"\n🤖 NexoraGuard:\n{response2['answer']}")
    # print(f"Source Documents Retrieved: {len(response2['sources'])}")
    # if response2['sources']:
    #     print(f"  Top source: {response2['sources'][0].metadata.get('source_type', 'N/A')} - {response2['sources'][0].metadata.get('category', 'N/A')}")

    # # Test Case 3: Question likely not in KB
    # response3 = ask_nexora_guard("What's the weather like in Sydney for an outdoor event?")
    # print(f"\n🤖 NexoraGuard:\n{response3['answer']}")
    # print(f"Source Documents Retrieved: {len(response3['sources'])}")
    
    # # Test Case 4: More complex query potentially needing info from product docs
    # response4 = ask_nexora_guard("Tell me about exclusions for Business Insurance")
    # print(f"\n🤖 NexoraGuard:\n{response4['answer']}")
    # print(f"Source Documents Retrieved: {len(response4['sources'])}")
    # if response4['sources']:
    #     print(f"  Top source: {response4['sources'][0].metadata.get('source_type', 'N/A')} - {response4['sources'][0].metadata.get('product_name', 'N/A')}")
        
else:
    print("RAG chain is not initialized, skipping tests for ask_nexora_guard.")

# ## 6. Interactive Chatbot Demonstration (Streamlit Web App)

def run_streamlit_app():
    st.set_page_config(page_title="NexoraGuard AI Chatbot", layout="wide")

    # Sidebar for Toggles and Info
    st.sidebar.title("NexoraGuard Controls")
    st.sidebar.info(
        "This is a demo of the Nexora RAG Customer Service Chatbot. "
        "It uses a local LLM (`qwen3:1.7b` via Ollama) and a knowledge base "
        "built from Nexora's product and FAQ data."
    )
    
    show_debug_info = st.sidebar.checkbox("Show Debug Info (Intent, Entities, Sources)", value=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About NexoraGuard")
    st.sidebar.markdown(
        "NexoraGuard is designed to assist with queries regarding Nexora's insurance products, "
        "coverage, claims, and account management."
    )

    # Main Chat Interface
    st.title("🤖 NexoraGuard Insurance Assistant")
    st.caption("Your AI-powered guide to Nexora's insurance solutions.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm NexoraGuard. How can I help you with your Nexora insurance needs today?"}]

    # Display chat messages
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"] and show_debug_info:
                with st.expander("View Sources Used", expanded=False):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}: Type - {source.metadata.get('source_type', 'N/A')}**")
                        
                        # Display product name or FAQ question from metadata
                        if source.metadata.get('source_type') == 'product_data':
                            st.caption(f"Product: {source.metadata.get('product_name', 'N/A')}")
                        elif source.metadata.get('source_type') == 'faq':
                            st.caption(f"FAQ Category: {source.metadata.get('category', 'N/A')}")
                            st.caption(f"Q: {source.metadata.get('question', 'N/A')}")

                        st.text_area(f"Content Snippet {i+1}", value=source.page_content[:500]+"...", height=100, disabled=True, key=f"source_{message['role']}_{i}_{source.metadata.get('product_id', source.metadata.get('question',''))}")


    # Chat input
    if prompt := st.chat_input("Ask about Nexora products, coverage, claims..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.spinner("NexoraGuard is thinking..."):
            if not rag_chain or not vector_store: # Check if core components are initialized
                response_data = {
                    "answer": "I'm sorry, the AI system is not fully initialized. Please ensure data is loaded and the RAG chain is built.",
                    "sources": [], "intent": "error", "entities": {}
                }
            else:
                response_data = ask_nexora_guard(prompt)
        
        assistant_response_content = response_data["answer"]
        
        # Add assistant response to chat history
        assistant_message = {"role": "assistant", "content": assistant_response_content, "sources": response_data["sources"]}
        st.session_state.messages.append(assistant_message)

        # Display assistant response and debug info
        with st.chat_message("assistant"):
            st.markdown(assistant_response_content)
            if show_debug_info:
                st.caption(f"Detected Intent: {response_data['intent']}")
                st.caption(f"Extracted Entities: {response_data['entities']}")
                if response_data["sources"]:
                    with st.expander("View Sources Used", expanded=False):
                        for i, source in enumerate(response_data["sources"]):
                            st.markdown(f"**Source {i+1}: Type - {source.metadata.get('source_type', 'N/A')}**")
                            if source.metadata.get('source_type') == 'product_data':
                                st.caption(f"Product: {source.metadata.get('product_name', 'N/A')}")
                            elif source.metadata.get('source_type') == 'faq':
                                st.caption(f"FAQ Category: {source.metadata.get('category', 'N/A')}")
                                st.caption(f"Q: {source.metadata.get('question', 'N/A')}")
                            st.text_area(f"Content Snippet {i+1}", value=source.page_content[:500]+"...", height=100, disabled=True, key=f"source_streamlit_{i}_{source.metadata.get('product_id', source.metadata.get('question',''))}")
                else:
                    st.caption("No specific source documents were heavily relied upon for this response (or it's a canned response).")

# # To signal where the "main" execution for streamlit would start if this notebook were a script:
if __name__ == '__main__':
    if 'rag_chain' in globals() and rag_chain is not None:
        run_streamlit_app()
    else:
        print("Core RAG components not initialized. Cannot start Streamlit app.")

# ## 7. Implementation & Deployment Strategy (Outline)
# 
# Successfully developing NexoraGuard is the first step. Deploying and maintaining it robustly is paramount.
# 
# **A. Development & Iteration:**
# 1.  **Local Development:** As demonstrated (Python, Ollama, ChromaDB).
# 2.  **Version Control:** Git for all code, configurations, and potentially DVC for large data/models.
# 3.  **Modular Design:** Separating concerns (data processing, embedding, retrieval, generation, UI) allows for easier updates and scaling. Current notebook structure follows this.
# 
# **B. Packaging & Deployment Options:**
# 1.  **Containerization (Docker):** Package the chatbot application (Python backend + Streamlit frontend, or a FastAPI backend with a separate frontend) into a Docker container. This ensures consistency across environments.
#     *   Include Ollama setup or instructions for connecting to a shared Ollama instance if not embedding it directly for scalability.
# 2.  **Deployment Platforms:**
#     *   **Cloud VMs (AWS EC2, Azure VM, GCP Compute Engine):** Basic deployment, good control. Requires manual setup of environment or Docker runtime.
#     *   **Managed Container Services (AWS ECS/EKS, Azure AKS, GCP GKE/Cloud Run):** Recommended for scalability, resilience, and easier management. Cloud Run is excellent for stateless applications like Streamlit apps or FastAPI backends.
#     *   **Serverless Functions (AWS Lambda, Azure Functions, GCP Cloud Functions) for API Backend:** If the core RAG logic is exposed via an API (e.g., FastAPI), serverless can be cost-effective and auto-scaling. The LLM component might need a more persistent host, or use serverless instances with larger memory/GPU if available for model serving.
# 3.  **Vector Database:**
#     *   **ChromaDB (Self-hosted/Managed):** Can be run in a Docker container alongside the app or as a separate service. For production, consider its scaling limits or managed vector DB alternatives (e.g., Pinecone, Weaviate, Qdrant Cloud, Vertex AI Vector Search).
# 
# **C. Key Considerations for Production:**
# 1.  **Scalability:**
#     *   LLM serving: Ollama locally is fine for one user. For many, consider dedicated model serving solutions (e.g., vLLM, TGI, Seldon Core) or managed LLM APIs if moving to larger models.
#     *   Vector DB: Choose one that scales with data and query load.
#     *   Application: Horizontal scaling of app instances (e.g., multiple Docker containers behind a load balancer).
# 2.  **Reliability & Monitoring:**
#     *   Logging: Comprehensive logging of requests, responses, errors, and retrieved context.
#     *   Monitoring: Track latency, error rates, resource usage (CPU, memory, GPU if used).
#     *   Health Checks: For application instances and dependent services.
# 3.  **Security:**
#     *   Input sanitization.
#     *   Authentication/Authorization if accessing sensitive data or features.
#     *   Secure API endpoints (HTTPS).
#     *   Protecting knowledge base integrity.
# 4.  **Maintainability:**
#     *   Automated testing (unit, integration, E2E).
#     *   CI/CD pipelines for automated build, test, and deployment.
#     *   Regular updates to the knowledge base and model retraining/fine-tuning cycles.
# 
# **D. Phased Rollout:**
# 1.  **Internal Alpha/Beta:** Test with Nexora employees.
# 2.  **Limited External Beta:** Roll out to a small segment of customers.
# 3.  **General Availability:** Gradual rollout to all users, monitoring performance and feedback closely.

# ## 8. Conceptual Tasks: Advanced Horizons
# 
# ### 8.1. MLOps Pipeline: Engineering AI for Durability
# 
# An end-to-end MLOps pipeline is critical for the long-term success, scalability, and reliability of NexoraGuard.
# 
# **Conceptual MLOps Pipeline for NexoraGuard:**
# 
# ```
# [Data Sources (Products, FAQs, Chat Logs, User Feedback)]
#      |
#      v
# [1. Data Ingestion & Preprocessing] --- (Versioning: DVC/Git-LFS)
#      | (Scheduled or Triggered)
#      v
# [2. Knowledge Base Construction & Vectorization]
#      |  - Embeddings Model (Ollama/SentenceTransformers)
#      |  - Vector Store (ChromaDB -> Production Vector DB)
#      |  - Output: Updated Vector Index
#      |
#      v
# [3. (Optional) Intent/Entity Model Training/Update] --- (Model Registry: MLflow)
#      |  - Based on new chat data, annotated examples
#      |  - Output: Updated Intent/Entity Model
#      |
#      v
# [4. (Optional) LLM Fine-Tuning Pipeline] --- (Model Registry: MLflow)
#      |  - Base LLM (e.g., qwen3:1.7b -> Larger OSS model)
#      |  - Fine-tuning dataset (from Chat Conversations)
#      |  - Techniques: LoRA/QLoRA
#      |  - Evaluation: Metrics + Human-in-the-loop
#      |  - Output: Fine-tuned LLM
#      |
#      v
# [5. Chatbot Application Build & Packaging (Docker)]
#      |  - Incorporates:
#      |      - Latest Vector Index
#      |      - Latest Intent/Entity Model (if used)
#      |      - Base or Fine-tuned LLM
#      |      - RAG Orchestration Logic (LangChain)
#      |      - API (FastAPI) / UI (Streamlit)
#      |
#      v
# [6. CI/CD Pipeline (Jenkins, GitLab CI, GitHub Actions)]
#      |  - Automated Testing (Unit, Integration, RAG Evals)
#      |  - Security Scans
#      |  - Build Docker Image
#      |  - Push to Container Registry
#      |
#      v
# [7. Deployment (Staging -> Production)]
#      |  - Deployment Strategy: Blue/Green, Canary
#      |  - Infrastructure: Kubernetes, Cloud Run, etc.
#      |  - Configuration Management
#      |
#      v
# [8. Monitoring & Observability Dashboard]
#      |  - Metrics: Latency, Throughput, Error Rates
#      |  - RAG Quality: Retrieval Relevance, Answer Faithfulness, Helpfulness (User Feedback)
#      |  - Data Drift / Concept Drift
#      |  - Cost Monitoring
#      |
#      v
# [9. Feedback Loop & Iteration Management]
#      |  - Collect user ratings, difficult questions, hallucinations
#      |  - Annotate data for KB updates or fine-tuning
#      |  - Trigger retraining/updates (-> Back to Step 1, 3, or 4)
#      --------------------------------------------------------------------
# ```
# 
# **Key Components & Technologies (Examples):**
# *   **Workflow Orchestration:** Apache Airflow, Kubeflow Pipelines, Prefect.
# *   **Data Versioning:** DVC, Git LFS.
# *   **Model Registry & Experiment Tracking:** MLflow, Weights & Biases, Vertex AI Model Registry.
# *   **CI/CD:** Jenkins, GitLab CI, GitHub Actions.
# *   **Infrastructure as Code (IaC):** Terraform, CloudFormation.
# *   **Monitoring:** Prometheus, Grafana, ELK Stack, Cloud-specific monitoring tools.
# 
# **Application Lifecycle Management:**
# *   **Agile Development:** Sprints, iterative improvements.
# *   **Change Management:** Controlled updates to KB, models, and application code.
# *   **Rollback Strategies:** Essential for mitigating issues in production.

# ### 8.2. Fine-Tuning Strategy: Elevating LLM Performance
# 
# While RAG significantly improves factual grounding, fine-tuning the LLM (e.g., `qwen3:1.7b` or a larger base model) on Nexora-specific data can yield substantial benefits. The `chat_conversations.json` dataset is ideal for this.
# 
# **High-Level Fine-Tuning Plan:**
# 
# 1.  **Goal Definition:**
#     *   Improve stylistic alignment (Nexora's brand voice, tone).
#     *   Enhance understanding of nuanced, domain-specific queries.
#     *   Reduce boilerplate and improve conciseness where appropriate.
#     *   Potentially improve summarization capabilities for retrieved context.
#     *   Better handling of multi-turn conversational nuances (if the fine-tuning data supports it).
# 
# 2.  **Data Preparation (`chat_conversations.json`):**
#     *   **Filtering:** Select high-quality, successful conversations where the agent's response is exemplary. Filter out incomplete, erroneous, or irrelevant exchanges.
#     *   **Formatting:** Convert conversations into an instruction-following format suitable for the chosen LLM. Example:
#         ```json
#         {
#           "instruction": "A customer is asking about the coverage of Public Liability insurance.",
#           "input": "What is Public Liability insurance?", // User's query
#           "output": "Public Liability insurance covers organizations against legal liability for bodily injury, property damage, and advertising injury claims from third parties. It's essential for businesses interacting with the public." // Ideal agent response
#         }
#         ```
#         Or, for models like Llama/Mistral/Qwen (often fine-tuned with chat templates):
#         ```
#         <s>[INST] What is Public Liability insurance? [/INST] Public Liability insurance covers organizations against legal liability for bodily injury, property damage, and advertising injury claims from third parties. It's essential for businesses interacting with the public.</s>
#         ```
#     *   **Data Augmentation (Optional):** Paraphrase existing Q&A pairs, generate variations to increase dataset size and diversity.
#     *   **Splitting:** Divide into training, validation, and test sets.
# 
# 3.  **Model Selection & Baseline:**
#     *   **Base Model:** Start with the same model used for RAG (e.g., `qwen3:1.7b`) or a slightly larger, capable open-source model (e.g., `Mistral-7B`, `Llama-3-8B-Instruct` when available and feasible).
#     *   **Baseline Performance:** Evaluate the (Base LLM + RAG) system on a curated test set before fine-tuning.
# 
# 4.  **Fine-Tuning Process:**
#     *   **Technique:** Parameter-Efficient Fine-Tuning (PEFT) like LoRA (Low-Rank Adaptation) or QLoRA (Quantized LoRA) is highly recommended. This drastically reduces computational requirements and allows fine-tuning even large models on modest hardware.
#     *   **Hyperparameter Tuning:** Learning rate, batch size, number of epochs, LoRA rank (`r`), alpha.
#     *   **Training Infrastructure:** Local GPU (if powerful enough), or cloud GPU instances (AWS SageMaker, Azure ML, GCP Vertex AI Training). Tools like `Axolotl`, `LLaMA-Factory`, or Hugging Face `Trainer` simplify this.
# 
# 5.  **Evaluation:**
#     *   **Quantitative Metrics:**
#         *   ROUGE, BLEU (for similarity to reference answers).
#         *   Perplexity (on validation set).
#         *   Task-specific metrics (e.g., accuracy for classification-like intents if fine-tuning for that).
#     *   **Qualitative Metrics (Human Evaluation):** Crucial for chatbots. Assess:
#         *   **Helpfulness:** Does the answer satisfy the user's need?
#         *   **Faithfulness/Accuracy:** Is the answer consistent with provided context (if used with RAG post-fine-tuning) and factual?
#         *   **Clarity & Conciseness.**
#         *   **Tone & Style Alignment.**
#     *   **A/B Testing:** Compare the fine-tuned LLM (+RAG) against the base LLM (+RAG) in a live or simulated environment.
# 
# 6.  **Integration with RAG:**
#     *   After fine-tuning, the improved LLM replaces the base LLM in the RAG pipeline.
#     *   The system still benefits from RAG's ability to inject up-to-date, factual context, while the LLM is better at *utilizing* that context in a Nexora-specific way.
# 
# **Potential Benefits of Fine-Tuning over Base LLM with RAG:**
# *   **Improved Domain Acclimatization:** The LLM learns Nexora's specific jargon, common query patterns, and desired response styles more deeply than RAG prompting alone can achieve.
# *   **Enhanced Nuance Understanding:** Better at handling ambiguous queries or those requiring implicit knowledge gleaned from the fine-tuning data.
# *   **More Natural Interactions:** Responses can sound less generic and more like a seasoned Nexora agent.
# *   **Reduced Hallucinations (Potentially):** While RAG targets this, a fine-tuned model might be less prone to confabulate details *within* the domain it's trained on, even when RAG context is slightly imperfect.
# *   **Efficiency:** A fine-tuned model might sometimes achieve desired responses with less verbose prompting or fewer retrieved documents if it has internalized certain common knowledge patterns.
# 
# **Challenges:**
# *   **Data Quality & Quantity:** Requires sufficient high-quality conversational data.
# *   **Catastrophic Forgetting:** Risk that the LLM forgets general capabilities if not fine-tuned carefully. PEFT methods help mitigate this.
# *   **Cost & Effort:** Fine-tuning, even PEFT, requires resources and expertise.
# 
# A hybrid (Fine-Tuned LLM + RAG) approach often offers the best of both worlds: a model adept at the domain's specifics, grounded by fresh, external knowledge.

# ## 9. Conclusion & Vision for Nexora's AI Future
# 
# This notebook has demonstrated the foundational yet powerful capabilities of a RAG-based AI assistant, NexoraGuard, tailored for Nexora Pty Ltd. We've moved beyond basic implementation by:
# 
# *   **Ingesting and structuring diverse data sources.**
# *   **Implementing an enhanced intent and entity recognition system.**
# *   **Building a transparent RAG pipeline with source attribution.**
# *   **Providing a clear, interactive demonstration via Streamlit.**
# *   **Outlining comprehensive strategies for MLOps and future LLM fine-tuning.**
# 
# **NexoraGuard is not just a chatbot; it's a stepping stone towards a more intelligent, efficient, and customer-centric Nexora.**
# 
# **Future Enhancements (Beyond Current Scope):**
# 1.  **Multi-Turn Conversation Management:** Enabling NexoraGuard to remember context from previous turns in a conversation for more natural dialogue.
# 2.  **Advanced Retrieval Strategies:** Hybrid search (semantic + keyword), re-ranking of retrieved documents for optimal relevance.
# 3.  **Proactive Assistance:** Identifying potential user needs or offering relevant information before explicitly asked.
# 4.  **Integration with Nexora Systems:** For actions like initiating a quote, retrieving policy details directly, or escalating to specific human agent queues.
# 5.  **Continuous Learning Loop:** Actively using user feedback and interaction data to refine the knowledge base, intent models, and fine-tuned LLMs.
# 
# As an aspiring Lead AI Engineer at Nexora, I am excited by the prospect of leading such transformative AI initiatives. My approach emphasizes robust engineering, creative problem-solving, and a keen understanding of business objectives to deliver AI solutions that provide tangible value.
# 
# Thank you for this opportunity. I am confident that with a strategic vision and dedicated execution, Nexora can significantly enhance its customer service and operational efficiency through AI.

# This is the end of the notebook.
# To run the Streamlit app:
# 1. Make sure Ollama is running (e.g., `ollama serve` in a terminal if not already).
# 2. Make sure you have pulled the model: `ollama pull qwen3:1.7b` (or the chosen model).
# 3. Save this notebook as `nexora_lead_ai_solution.py`.
# 4. Open your terminal, navigate to the directory containing the .py file and the 'data_nexora_bot' folder.
# 5. Run: `streamlit run nexora_lead_ai_solution.py`