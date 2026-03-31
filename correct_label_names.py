import pandas as pd
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    uri="bolt://localhost:7687",
    auth=("neo4j", "password"),
    max_connection_lifetime=3600,
)

data = pd.read_csv('label_name_corrections.csv')

for i in range(len(data)):
    old_name = data.iloc[i,0]
    new_name = data.iloc[i,1]

    query = """
        MATCH (n:`%s`)
        REMOVE n:`%s`
        SET n:`%s`
    """ % (old_name, old_name, new_name)
    with driver.session() as session:
        result = session.run(query)
