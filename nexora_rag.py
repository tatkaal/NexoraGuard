# Streamlit (for the web app part)
import streamlit as st

@st.cache_resource(show_spinner=False)
def init_nexora_rag():
    import json
    import re
    import pandas as pd
    import csv
    from pathlib import Path
    import shutil # For cleaning up vector store if needed
    import time

    # LangChain components
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # For streaming
    from langchain.callbacks.manager import CallbackManager

    # NLP
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher

    # --- Global Configurations ---
    NEXORA_MODEL_NAME = 'gemma3:4b'

    EMBEDDING_MODEL_FOR_OLLAMA = "mxbai-embed-large"

    DATA_DIR = Path('data') # Create a dedicated directory for this project's data
    VECTOR_STORE_DIR = Path('nexora_chroma_vector_store')
    PRODUCTS_FILE = DATA_DIR / 'products_occupation.json'
    FAQS_CLEANED_FILE = DATA_DIR / 'faqs_cleaned.csv'

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
        # if product_docs:
        #     print("\nSample Product Document Content:")
        #     print(product_docs[0].page_content)
        #     print(f"\nSample Product Document Metadata: {product_docs[0].metadata}")
    else:
        print("No product data loaded to process.")

    # ### 2.2. FAQs Data (`faqs.csv`)
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
        # if faq_docs:
        #     print("\nSample FAQ Document Content:")
        #     print(faq_docs[0].page_content)
        #     print(f"\nSample FAQ Document Metadata: {faq_docs[0].metadata}")

    # ### 2.4. Consolidating the Knowledge Base
    # Combine all processed documents into a single list for vector store creation.

    documents = product_docs + faq_docs
    if documents:
        print(f"\nTotal documents for RAG knowledge base: {len(documents)}")
        # print("Sample of combined documents (first product and first FAQ):")
        # if product_docs: print(f"Product Doc Example Metadata: {product_docs[0].metadata}")
        # if faq_docs: print(f"FAQ Doc Example Metadata: {faq_docs[0].metadata}")
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

            start = time.time()
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=str(VECTOR_STORE_DIR) # Chroma expects a string path
            )
            elapsed = time.time() - start
            print(f'Built new vector store → {VECTOR_STORE_DIR} in {elapsed:.2f}s')

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

    # ## 4. Intent Recognition & Entity Extraction
    # 
    # Understanding the user's *intent* and key *entities* in their query is crucial for guiding the RAG pipeline effectively. We'll enhance the provided basic system.
    # 
    # **Approach:**
    # *   **Intent Classification:** Using a combination of improved regex and SpaCy's `Matcher` for more robust pattern identification.
    # *   **Entity Extraction:** Leveraging SpaCy's pre-trained NER and extending it with `PhraseMatcher` for domain-specific terms (like product names).
    # 
    # ### 4.1. Defining Intents and Entities
    # We'll define key intents relevant to Nexora's customer service.

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
    INTENT_PATTERNS_REGEX = {
    'get_quote': r'\b(get a quote|quote for|how much for|pricing on)\b',
    'product_info': r'\b(what is|tell me about|details on|info on)\s+([A-Za-z\s]+insurance)\b|\b(compare|difference between)\b',
    'coverage_query': r'\b(cover(ed|s|age)|include|policy limit|exclusion|excess)\b',
    'claim_query': r'\b(claim|lodge|report an incident|damage|incident)\b',
    'account_management': r'\b(my account|login|password|policy document|certificate of currency|update details|amend policy|payment)\b',
    'general_greeting': r'\b(hi|hello|hey|good morning|good afternoon)\b',
    'general_farewell': r'\b(bye|goodbye|thanks|thank you)\b',
    'ask_agent': r'\b(human|agent|person|talk to someone)\b'
    }
        
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
    9.  Answer to the point and in concise manner without excessive unrelated information.

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

    # # ### Quick Test
    # def ask_and_debug(chain, question: str):
    #     # 1) retrieve the top-k docs
    #     docs = chain.retriever.get_relevant_documents(question)
    #     # 2) build exactly the same context string
    #     context = "\n\n".join(doc.page_content for doc in docs)
    #     print("Actual question: ", question)
    #     print('-'*50)
    #     print("───── CONTEXT SENT TO LLM ─────\n")
    #     print(context)
    #     print("\n──────── END CONTEXT ────────\n")
    #     # 3) finally ask the chain
    #     return chain.invoke({'query': question})

    # # now use our helper instead of rag_chain.invoke:
    # question = 'What does Professional Indemnity cover?'
    # print(ask_and_debug(rag_chain, question))

    # question = 'How do I change my password?'
    # print(ask_and_debug(rag_chain, question))

    # question = 'How do I update my personal details?'
    # print(ask_and_debug(rag_chain, question))

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
    
    return ask_nexora_guard, rag_chain, vector_store

# ### 5.4. Quick Test of the Full Pipeline

# if rag_chain: # Only test if RAG chain is set up
    # print("\n--- Testing the ask_nexora_guard function ---")
    
    # # Test Case 1: Product Information
    # response1 = ask_nexora_guard("What is Professional Indemnity insurance?")
    # # print(f"\n🤖 NexoraGuard:\n{response1['answer']}")
    # print(f"Source Documents Retrieved: {len(response1['sources'])}")
    # if response1['sources']:
    #     print(f"  Top source: {response1['sources'][0].metadata.get('source_type', 'N/A')} - {response1['sources'][0].metadata.get('product_name', response1['sources'][0].metadata.get('question', 'N/A'))}")

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
        
# else:
#     print("RAG chain is not initialized, skipping tests for ask_nexora_guard.")

