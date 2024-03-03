

try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/ac"
    )
    ac_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

if not index_loaded:
    # load data
    ac_docs = SimpleDirectoryReader(
        input_files=["./data/10k/AC-Unit.pdf"]
    ).load_data()

    # build index
    vector_store_ac = MilvusVectorStore(host="localhost", port=default_server.listen_port, dim=1536, collection_name="ac", overwrite=True)
    storage_context_ac = StorageContext.from_defaults(vector_store=vector_store_ac)
    ac_index = VectorStoreIndex.from_documents(ac_docs, storage_context=storage_context_ac)

    # persist index
    ac_index.storage_context.persist(persist_dir="./storage/ac")

ac_engine = ac_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=ac_engine,
        metadata=ToolMetadata(
            name="AC_manual",
            description=(
                "Provides information about AC unit manual instructions. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]