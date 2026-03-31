from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password, max_connection_lifetime=None):
        """
        uri: URI of the Neo4j database
        user: Username of the Neo4j database
        password: Password of the Neo4j database
        """
        if max_connection_lifetime:
            self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=max_connection_lifetime)
        else:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """
        Close the connection to the Neo4j database
        """
        self.driver.close()

    def create_node(self, tx, name, label):
        """
        Create a node in the Neo4j database
        
        tx: Transaction object
        name: Name of the node
        label: Label of the node
        """
        label = label.replace(' ', '_').replace('-', '_').upper().replace('(', '').replace(')', '').replace(',', '').replace('"', '').replace("'", '').replace('×', 'x').replace('.', '').replace("/", "_").replace('.', '')
        # If label's first character is not a letter, prepend it with "X". This is because Neo4j does not allow labels to start with a non-letter character.
        if not label[0].isalpha():
            label = "X" + label
        query = (
            "MERGE (n:{label} {{name: $name}})".format(label=label)
        )
        tx.run(query, name=name)
    
    def create_relationship(self, tx, subj, predicate, obj):
        """
        Create a relationship between two nodes in the Neo4j database

        tx: Transaction object
        subj: Name of the subject node
        predicate: Predicate of the relationship
        obj: Name of the object node
        """
        predicate = predicate.replace(' ', '_').replace('-', '_').upper().replace('(', '').replace(')', '').replace(',', '').replace('"', '').replace("'", '').replace('×', 'x').replace('.', '').replace("/", "_").replace('.', '')
        query = (
            "MATCH (a {{name: $subj}}), (b {{name: $obj}}) "
            "MERGE (a)-[r:{predicate}]->(b)".format(predicate=predicate)
        )
        tx.run(query, subj=subj, obj=obj)

    def create_relationship_with_labels(self, tx, subj, subj_label, predicate, obj, obj_label):
        """
        Create a relationship between two nodes in the Neo4j database with labels
        
        tx: Transaction object
        subj: Name of the subject node
        subj_label: Label of the subject node
        predicate: Predicate of the relationship
        obj: Name of the object node
        obj_label: Label of the object node
        """
        predicate = predicate.replace(' ', '_').replace('-', '_').upper().replace('(', '').replace(')', '').replace(',', '').replace('"', '').replace("'", '').replace('×', 'x').replace('.', '').replace("/", "_").replace('.', '')
        subj_label = subj_label.replace(' ', '_').replace('-', '_').upper().replace('(', '').replace(')', '').replace(',', '').replace('"', '').replace("'", '').replace('×', 'x').replace('.', '').replace("/", "_").replace('.', '')
        # If label's first character is not a letter, prepend it with "X". This is because Neo4j does not allow labels to start with a non-letter character.
        if not subj_label[0].isalpha():
            subj_label = "X" + subj_label
        obj_label = obj_label.replace(' ', '_').replace('-', '_').upper().replace('(', '').replace(')', '').replace(',', '').replace('"', '').replace("'", '').replace('×', 'x').replace('.', '').replace("/", "_").replace('.', '')
        # If label's first character is not a letter, prepend it with "X". This is because Neo4j does not allow labels to start with a non-letter character.
        if not obj_label[0].isalpha():
            obj_label = "X" + obj_label
        query = (
            "MATCH (a:{subj_label} {{name: $subj}}), (b:{obj_label} {{name: $obj}}) "
            "MERGE (a)-[r:{predicate}]->(b)".format(subj_label=subj_label, obj_label=obj_label, predicate=predicate)
        )
        tx.run(query, subj=subj, obj=obj)

    def create_knowledge_graph(self, data):
        """
        Create a knowledge graph in the Neo4j database

        data: List of dictionaries containing the knowledge graph data
        """
        with self.driver.session() as session:
            i = 0
            for item in data:
                kg = item['kg']
                entities_section, triples_section = kg.split("Triples:")
                entities_section = entities_section.replace("Entities:", "").strip()
                triples_section = triples_section.strip()

                # Create nodes from entities
                entities = [line.strip() for line in entities_section.split("\n") if line.strip()]
                for entity in entities:
                    name, label = entity.rsplit(":", 1)
                    name = name.replace('(', '').replace(')', '').replace('-', '').strip()
                    label = label.replace('(', '').replace(')', '').replace('-', '').strip()
                    session.execute_write(self.create_node, name, label)

                # Create relationships from triples
                triples = [line.strip() for line in triples_section.split("\n") if line.strip()]
                for triple in triples:
                    triple = triple.replace('(', '').replace(')', '').replace('-', '').strip()
                    spb = [part.strip() for part in triple.split("<>")]
                    if len(spb) == 3:
                        subj, predicate, obj = spb
                        session.execute_write(self.create_relationship, subj, predicate, obj)
                    else:
                        i += 1
                        print(f"Triple {i} is not in the correct format: {triple}")
            print(f"\nTotal of {i} triples not in the correct format")
            print("Knowledge Graph creation completed")

    def query(self, query):
        """
        Execute a Cypher query in the Neo4j database and return the result

        query: Cypher query
        """
        with self.driver.session() as session:
            result = session.run(query)
            return result.data()
        
    def query_read_transaction(self, query):
        """
        Execute a read-only Cypher query in the Neo4j database and return the result

        query: Cypher query
        """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query).data())
            return result
        
    def calc_node_embeddings(self, embedding_dim:int = 256, random_seed:int = 42):
        """
        Calculate the node embeddings in the knowledge graph
        """
        # Drop the graph if it already exists
        q = """
            CALL gds.graph.drop('myGraph', false)
            YIELD graphName, nodeCount, relationshipCount;
        """
        self.query(q)
        
        # Create the graph projection
        q = """
            CALL gds.graph.project(
                'myGraph',
                '*',
                {
                    edge: {
                        type: '*',
                        orientation: 'UNDIRECTED'
                    }
                }
            );
        """
        self.query(q)
        
        # Calculate the node embeddings
        q = f"""
            CALL gds.fastRP.write(
                'myGraph',
                {{
                    embeddingDimension: {embedding_dim},
                    writeProperty: 'fastrpEmbedding',
                    randomSeed: {random_seed}
                }}
            )
            YIELD nodePropertiesWritten;
        """
        self.query(q)

    def get_node_embedding(self, name: str, label: str):
        """
        Get the node embedding of a node in the knowledge graph

        name: Name of the node
        label: Label of the node
        """
        q = """
            MATCH (n:`%s` {name: "%s"})
            RETURN n.fastrpEmbedding as embedding
        """ % (label, name.replace("'", "\\'").replace('"', '\\"'))
        result = self.query(q)
        return result[0]['embedding']
        
    def calc_node_centralities(self) -> dict:
        """
        Calculate the node centralities in the knowledge graph and return the related statistics
        """

        # Drop the graph if it already exists
        q = """
            CALL gds.graph.drop('myGraph', false)
            YIELD graphName, nodeCount, relationshipCount;
        """
        self.query(q)
        
        # Create the graph projection
        q = """
            CALL gds.graph.project(
                'myGraph',
                '*',
                {
                    edge: {
                        type: '*',
                        orientation: 'UNDIRECTED'
                    }
                }
            );
        """
        self.query(q)
        
        # Calculate the degree centralities of the nodes and write them to the graph as a property called 'degree'
        q = """
            CALL gds.degree.write('myGraph', {writeProperty: "degree"})
            YIELD nodePropertiesWritten, centralityDistribution
            RETURN nodePropertiesWritten, centralityDistribution
        """
        degree_centrality_results = self.query(q)
        
        # Calculate the PageRanks of the nodes and write them to the graph as a property called 'pageRank'
        q = """
            CALL gds.pageRank.write('myGraph', {writeProperty: "pageRank"})
            YIELD nodePropertiesWritten, centralityDistribution
            RETURN nodePropertiesWritten, centralityDistribution
        """
        pageRank_results = self.query(q)
        
        # Return the stats
        return {
            "degree": degree_centrality_results[0]['centralityDistribution'],
            "pageRank": pageRank_results[0]['centralityDistribution']
        }

    def get_degree_centrality(self, name: str, label: str):
        """
        Get the degree centrality of a node in the knowledge graph

        name: Name of the node
        label: Label of the node
        """
        q = """
            MATCH (n:`%s` {name: "%s"})
            RETURN n.degree as degree
        """ % (label, name.replace("'", "\\'").replace('"', '\\"'))
        result = self.query(q)
        return int(result[0]['degree'])
    
    def get_page_rank(self, name: str, label: str):
        """
        Get the PageRank of a node in the knowledge graph

        name: Name of the node
        label: Label of the node
        """
        q = """
            MATCH (n:`%s` {name: "%s"})
            RETURN n.pageRank as pageRank
        """ % (label, name.replace("'", "\\'").replace('"', '\\"'))
        result = self.query(q)
        return float(result[0]['pageRank'])
    
    def wcc(self):
        """
        Find the weakly connected components in the knowledge graph
        """
        q = """
            CALL gds.graph.drop('myGraph', false)
            YIELD graphName, nodeCount, relationshipCount;
        """
        self.query(q)
        q = """
            CALL gds.graph.project('myGraph', '*', '*')
            YIELD graphName, nodeCount, relationshipCount;
        """
        self.query(q)
        q = """
            CALL gds.wcc.write('myGraph', {
                writeProperty: 'componentId'
            })
            YIELD nodePropertiesWritten, componentCount;
        """
        self.query(q)
        q = """
            MATCH (n)
            WITH n.componentId AS componentId, COUNT(n) AS componentSize
            RETURN componentId, componentSize
            ORDER BY componentSize DESC;
        """
        result = self.query(q)
        return result

    def get_all_nodes(self) -> list:
        """
        Get all nodes in the knowledge graph
        """
        q = """
            MATCH (n)
            RETURN n.name as name, labels(n)[0] as label
            ORDER BY n.degree DESC
        """
        result = self.query(q)
        return [(node['name'], node['label']) for node in result]

    def get_all_nodes_with_type(self, label: str) -> list:
        """
        Get all nodes with a specific label in the knowledge graph

        label: Label of the nodes
        """
        q = """
            MATCH (n:`%s`)
            RETURN n.name as name
            ORDER BY n.degree DESC
        """ % label
        result = self.query(q)
        return [node['name'] for node in result]

    def get_the_category_of_doc_that_mentions_this_museum_item(self, museum_item: str) -> str: 
        """
        Get the category of the document that mentions given museum item    

        museum_item: Name of the item
        """
        q = """
            MATCH (n:`Museum item` {name: "%s"})<-[:MENTIONS]-(d:`Document`)
            RETURN d.text as doc
        """ % museum_item.replace("'", "\\'").replace('"', '\\"')
        result = self.query(q)
        txts = [doc['doc'] for doc in result]
        # Evert txt has this pattern on its second line:
        # "< Collection: collection_name >" where collection_name is the name of the collection
        categories = [txt.split('\n')[1].split(': ')[1][:-2] for txt in txts]
        # Return the most commmon category (in case of multiple categories from multiple documents)
        return max(set(categories), key=categories.count)

