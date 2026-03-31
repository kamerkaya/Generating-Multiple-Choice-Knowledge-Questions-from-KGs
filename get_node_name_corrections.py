import csv
from neo4j import GraphDatabase
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


class NameChange(BaseModel):
    is_name_change: bool
    corrected_name: str


driver = GraphDatabase.driver(
    uri="bolt://localhost:7687",
    auth=("neo4j", "password"),
    max_connection_lifetime=3600,
)
query = """MATCH (n) WHERE NOT "Document" IN labels(n) RETURN n.name"""
with driver.session() as session:
    result = session.run(query)
    out = result.data()
out = [x["n.name"] for x in out]
print(len(out))
print(out)
print("-" * 50)

api_key = open(".openai_api_key").read()
client = OpenAI(api_key=api_key)
corrected_items = []
for item_name in tqdm(out, desc="Processing names"):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": """Later, I will give you an input. I want you to tell me that if it is properly capitalized? It will be used as a title. I just want you to check for capitalization errors. Also, if the words are separated non-formally (e.g. with underscores), separate them with spaces.\n(If if you think the given input is proper, make is_name_change False, and make corrected_name same with the given input)\n(If not, how do you propose to correct it? In this case, make is_name_change True, and make corrected_name the corrected version of the input you propose.)\nHere is the input:\n\n"""
                + item_name,
            },
        ],
        response_format=NameChange,
    )
    res = completion.choices[0].message.parsed
    corrected_items.append((item_name, res.is_name_change, res.corrected_name))

changes = []
num_changes = 0
num_change_but_same = 0
for i in range(len(corrected_items)):
    if corrected_items[i][1]:
        if corrected_items[i][0] == corrected_items[i][2]:
            num_change_but_same += 1
        else:
            changes.append((corrected_items[i][0], corrected_items[i][2]))
            num_changes += 1
print(f"Number of changes: {num_changes}")
print(f"Number of changes but same: {num_change_but_same}")

with open("node_name_corrections.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(("0", "1"))
    writer.writerows(changes)

