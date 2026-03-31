from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    uri="bolt://localhost:7687",
    auth=("neo4j", "password"),
    max_connection_lifetime=3600,
)

query = """
    MATCH (n)
    WITH n ORDER BY n.created ASC
    WITH n.name AS name, labels(n)[0] as label, collect(n) as nodes
    CALL apoc.refactor.mergeNodes(nodes, {properties: {
        name:'discard',
        `.*`: 'discard'
    }})
    YIELD node
    RETURN node;
"""
with driver.session() as session:
    result = session.run(query)

