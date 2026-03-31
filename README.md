# Generating Multiple-Choice Knowledge Questions with Interpretable Difficulty Estimations using Knowledge Graphs and Large Language Models

### Knowledge Graph Construction (KGC)
1. `wikipedia_popular_pages_dataset.json`: Textual list of docs scraped from Wikipedia's top-100 most popular pages (https://en.wikipedia.org/wiki/Wikipedia:Popular_pages#Top-100_list) using `get_popular_wikipedia_pages.py`
2. Construct a knowledge graph (KG) of `wikipedia_popular_pages_dataset.json` using `lc_kgc.py`
3. Correct the node names (capitalizations, underscores instead of spaces between words, etc.): `get_node_name_corrections.py` -> `node_name_corrections.csv` -> `correct_node_names.py`
4. Correct the label names: `get_label_name_corrections.py` -> `label_name_corrections.csv` -> `correct_label_names.py`
5. Merge nodes: `merge_nodes.py`

### Multiple-Choice Question (MCQ) Generation
Generate MCQs using `generate_mcqs.py` {'4 per item' for each item in most popular 40 items in the graph database (w.r.t. degree centrality)}

### Difficulty Estimation (DE)
1. `data.ipynb`: Read and prepare the dataset for difficulty estimation
2. `models.ipynb`: Train and evaluate models for difficulty estimation