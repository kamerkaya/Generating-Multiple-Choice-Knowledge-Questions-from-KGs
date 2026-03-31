import csv
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    uri="bolt://localhost:7687",
    auth=("neo4j", "password"),
    max_connection_lifetime=3600,
)
query = """
    MATCH (n)
    WITH labels(n) AS nodeLabels
    UNWIND nodeLabels AS label
    RETURN label, COUNT(*) AS frequency
    ORDER BY frequency ASC;
"""
with driver.session() as session:
    result = session.run(query)
    out = result.data()
label_names = [x["label"] for x in out]
print(len(label_names))
print(label_names)
print("-" * 50)

changes = []
num_changes = 0
for label in label_names:
    new_label = label.replace("_", " ").replace("-", " ")
    if label != new_label:
        changes.append((label, new_label))
        num_changes += 1
print(f"Number of changes: {num_changes}")

with open("label_name_corrections.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(("0", "1"))
    writer.writerows(changes)
