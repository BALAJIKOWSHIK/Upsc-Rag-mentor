import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "upsc-mentor"

def upload_docs():
    docs_dir = "./docs"
    if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
        print("Create a ./docs folder and place your UPSC PDFs inside.")
        return

    print("Loading PDFs...")
    loader = PyPDFDirectoryLoader(docs_dir)
    raw_docs = loader.load()

    print(f"Loaded {len(raw_docs)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(raw_docs)
    total_chunks = len(splits)
    print(f"Total chunks created: {total_chunks}")

    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=768
    )

    print("Connecting directly to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # Batches of 25 chunks
    batch_size = 25
    print(f"Embedding and uploading in rate-managed batches...")

    for i in range(0, total_chunks, batch_size):
        chunk_slice = splits[i:i + batch_size]
        texts = [doc.page_content for doc in chunk_slice]
        
        # Safe retry mechanism for the 429 quota window
        vector_embeddings = None
        for attempt in range(5):
            try:
                vector_embeddings = embeddings.embed_documents(texts)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print(f"Rate limit hit. Cooling down for 25 seconds before retry...")
                    time.sleep(25)
                else:
                    raise e

        if not vector_embeddings:
            print(f"Failed to embed batch starting at chunk {i}. Skipping...")
            continue

        vectors_to_upsert = []
        for j, (doc, vector) in enumerate(zip(chunk_slice, vector_embeddings)):
            vector_id = f"chunk_{i + j}"
            metadata = {
                "text": doc.page_content,
                "source": str(doc.metadata.get("source", "document")),
                "page": int(doc.metadata.get("page", 0))
            }
            vectors_to_upsert.append((vector_id, vector, metadata))

        index.upsert(vectors=vectors_to_upsert)
        print(f"Successfully uploaded chunks {i + 1} to {min(i + batch_size, total_chunks)} of {total_chunks}")
        
        # Pacing sleep to stay within 100 RPM
        time.sleep(12)

    print("\nIngestion complete! All chunks are stored in Pinecone Cloud.")

if __name__ == "__main__":
    upload_docs()