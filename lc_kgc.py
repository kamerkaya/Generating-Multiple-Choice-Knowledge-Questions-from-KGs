"""
Knowledge Graph Construction (KGC) for Wikipedia dataset using GPT-4o and Neo4j
"""

# %%
from langchain_neo4j import Neo4jGraph
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from neo4j import GraphDatabase
import json
import pickle
import asyncio

# %%
# Set up language model
llm_model = "openai"  # "openai", "llama3.1", ...
if llm_model == "openai":
    api_key = open(".openai_api_key").read()
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
else:
    llm = ChatOllama(model=llm_model, temperature=0)

# %%
# Load dataset
with open("wikipedia_popular_pages_dataset.json") as f:
    dataset = json.load(f)

# %%
# Convert dataset to documents
docs = [Document(page_content=data["content"]) for data in dataset]

# %%
# Set up llm graph transformer
llm_graph_transformer = LLMGraphTransformer(
    llm=llm,
)

# %%
# Define async function to convert documents to graph
async def convert_documents_to_graph():
    # Convert documents to graph documents
    graph_documents = await llm_graph_transformer.aconvert_to_graph_documents(docs)
    # Inspect the first graph document's nodes and relationships
    print("Example graph document:")
    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")
    return graph_documents

# %%
# Call the async function and pickle the graph documents
graph_documents = asyncio.run(convert_documents_to_graph())

# Pickle graph_documents
pkl_path = "popular_wikipedia_dataset_gpt4o-graph-docs-v1.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(graph_documents, f)

# %%
# Set up neo4j graph and load graph documents to neo4j
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# Save graph to neo4j
load = False
if load:
    with open(pkl_path, "rb") as f:
        graph_documents = pickle.load(f)
    print(f"Loaded {len(graph_documents)} graph documents")
    print("Example graph document:")
    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")

print(f"Number of graph documents: {len(graph_documents)}")
graph.add_graph_documents(
    graph_documents=graph_documents,
    # include_source=True
)
print("Constructed knowledge graph has been saved to Neo4j")

# %%
# Set name field
driver = GraphDatabase.driver(
    uri="bolt://localhost:7687",
    auth=("neo4j", "password"),
    max_connection_lifetime=3600,
)
query = """MATCH (n) SET n.name = n.id"""
with driver.session() as session:
    result = session.run(query)
print("Setted field: name")

