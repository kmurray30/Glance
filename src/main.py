from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from pymilvus import MilvusException, connections
from llama_index.vector_stores.milvus import MilvusVectorStore
from milvus import default_server
from pymilvus import connections, utility

from dotenv import load_dotenv
load_dotenv()

import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

import atexit

def on_exit():
    # Stop and clean up Milvus server if it is running
    if (default_server.running):
        print("Stopping Milvus server")
        default_server.stop()
        default_server.cleanup()
    print("Exiting the script")

def loadVectors(manualName, manualDescription, manualShortName):
    # Load vectors
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/{manualShortName}"
        )
        index = load_index_from_storage(storage_context)

        index_loaded = True
    except:
        index_loaded = False

    if not index_loaded:
        # load data
        docs = SimpleDirectoryReader(
            input_files=[f"./data/10k/{manualName}.pdf"]
        ).load_data()

        # build index
        vector_store = MilvusVectorStore(host="localhost", port=default_server.listen_port, dim=1536, collection_name=manualShortName, overwrite=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

        # persist index
        index.storage_context.persist(persist_dir=f"./storage/{manualShortName}")

    engine = index.as_query_engine(similarity_top_k=3)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name=manualShortName,
                description=(
                    f"Provides information about {manualDescription} manual instructions. "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
    ]
    return query_engine_tools

# Begin execution here

atexit.register(on_exit)
# Only start milvus server if it is not already running
if (default_server.running):
    print("Milvus server is already running")
else:
    print("Starting Milvus server")
    default_server.start()

# Load vectors
query_engine_tools = loadVectors("AC-Unit", "AC Unit", "ac")

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-0613")

agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    # context=context
)

# response = agent.chat("For the AC unit, how do I change the filter?")
print("Ask me anything!")
while(True):
    inputStr = input()
    if (inputStr == "exit"):
        break
    response = agent.chat(inputStr)
    print(str(response))
    print("\nAsk me annything else! (or type 'exit' to quit)")