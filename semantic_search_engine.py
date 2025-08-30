import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_text_splitters import RecursiveCharacterTextSplitter
from colorama import Fore, Back, Style, init


class SSE:
    def __init__(self, db_name):
        self.collection_name = 'LFP Battery'
        self.embedding_dim = 384
        self.qdrant = None
        self.client = None
        self.embedding_model = None
        self.db_name = db_name
        self.vector_db_path = db_name
        self.docs = []
        self.diag_mode = True
        self.start_setups()


    def xprint(self, text, colour):
        if self.diag_mode:
            if colour.lower() == 'red':
                print(Fore.RED + text)
            elif colour.lower() == 'green':
                print(Fore.GREEN + text)
            elif colour.lower() == 'bright':
                print(Style.BRIGHT + text)

    def start_setups(self):
        init(autoreset=True)
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.client = QdrantClient(path=self.vector_db_path)
        self.initialise_vector_db()

    def data_reader(self, folder="source_files"):
        source_path = os.path.join(os.getcwd(), folder)
        source_files = [f for f in os.listdir(source_path) if f.endswith(".txt")]
        self.xprint(f'Fetching data from folder {source_path}', 'green')
        for file in source_files:
            file_path = os.path.join(source_path, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.docs.append(Document(page_content=content, metadata={"station": file}))
                    self.xprint(f'{file} - Read', 'bright')
            except:
                self.xprint(f'{file} - Failed', 'red')
        self.injection_vector_db()

    def initialise_vector_db(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )
        self.qdrant = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding =self.embedding_model
        )

    def injection_vector_db(self):
        if not self.docs:
            self.xprint("No documents to ingest.", "red")
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        chunks = splitter.split_documents(self.docs)
        self.qdrant.add_documents(chunks)
        self.xprint(f"Ingested {len(chunks)} chunks.", "green")


    def reload_vector_db(self):
        self.xprint('Vector DB Refreshing request' , 'green')
        self.data_reader()

    def semantic_search(self, query, k=4):
        # results = self.qdrant.similarity_search(query, k=k)
        results = self.qdrant.similarity_search_with_score(query, k=k)

        for doc, score in results:
            percentage = round(score * 100, 2)
            print(percentage)
            print(doc)
        # print(results)
        # context = "\n\n".join(doc.page_content for doc in results)
        #
        # return context

if __name__ == "__main__":
    db = SSE("qdrant_data")
    db.data_reader()

    while True:
        q = input("🔍 Ask: ")
        if q.lower().strip() in ["exit", "quit"]:
            break
        db.semantic_search(q)

