from kg import KnowledgeGraph


def generate_distractors_by_level(
    kg: KnowledgeGraph,
    max_level: int,
    subj: str,
    subj_type: str,
    predicate: str,
    obj: str,
    obj_type: str,
    is_key_subj: bool = True,
) -> dict:
    """
    Traverse graph using BFS to find the entities (distractors) that have the same type with the start_entity but do not check the fact in the triple.

    Args:
        kg (KnowledgeGraph): The knowledge graph to query.
        max_level (int): The maximum level to traverse in the BFS.
        subj (str): The subject of the triple.
        subj_type (str): The type of the subject.
        predicate (str): The predicate of the triple.
        obj (str): The object of the triple.
        obj_type (str): The type of the object.
        is_key_subj (bool): A flag to indicate whether the key is the subject or object.

    Returns:
        dict: A dictionary of distractors by level as they are found in the BFS traversal. (1 is the easiest hardest level.)
    """
    if is_key_subj:
        # Get entities of the same type that do not have the same predicate-object pair
        if '"' in obj:
            q = """
                MATCH (entity:`%s`)
                WHERE NOT (entity)-[:`%s`]->(:`%s` {name: '%s'})
                RETURN entity.name as distractor, labels(entity)[0] as distractor_label
            """ % (
                subj_type,
                predicate,
                obj_type,
                obj,
            )
        else:
            q = """
                MATCH (entity:`%s`)
                WHERE NOT (entity)-[:`%s`]->(:`%s` {name: "%s"})
                RETURN entity.name as distractor, labels(entity)[0] as distractor_label
            """ % (
                subj_type,
                predicate,
                obj_type,
                obj,
            )
    else:
        # Get entities of the same type that do not have the same subject-predicate pair
        if '"' in subj:
            q = """
                MATCH (entity:`%s`)
                WHERE NOT (:`%s` {name: '%s'})-[:`%s`]->(entity)
                RETURN entity.name as distractor, labels(entity)[0] as distractor_label
            """ % (
                obj_type,
                subj_type,
                subj,
                predicate,
            )
        else:
            q = """
                MATCH (entity:`%s`)
                WHERE NOT (:`%s` {name: "%s"})-[:`%s`]->(entity)
                RETURN entity.name as distractor, labels(entity)[0] as distractor_label
            """ % (
                obj_type,
                subj_type,
                subj,
                predicate,
            )
    result = kg.query(q)
    all_possible_distractors = [
        (r["distractor_label"], r["distractor"]) for r in result
    ]
    if len(all_possible_distractors) < 3:
        print("Not enough distractors found. Could not generate a valid question.")
        return {"all_possible_distractors": all_possible_distractors}

    # BFS
    distractors = []
    visited = set()
    queue = []
    if is_key_subj:
        queue.append((subj_type, subj, 0))
    else:
        queue.append((obj_type, obj, 0))

    while queue:
        current = queue.pop(0)
        if (current[0], current[1]) in visited:
            continue
        visited.add((current[0], current[1]))

        if current[2] >= max_level:
            break

        if '"' in current[1]:
            q = """
                MATCH (n:`%s` {name: '%s'})-[]-(entity)
                RETURN entity.name as distractor, labels(entity)[0] as distractor_label
            """ % (
                current[0],
                current[1],
            )
        else:
            q = """
                MATCH (n:`%s` {name: "%s"})-[]-(entity)
                RETURN entity.name as distractor, labels(entity)[0] as distractor_label
            """ % (
                current[0],
                current[1],
            )
        result = kg.query(q)
        entities = [
            (r["distractor_label"], r["distractor"], current[2] + 1) for r in result
        ]
        for entity in entities:
            if (
                (entity[0], entity[1]) not in visited
                and (entity[0], entity[1], current[2]) not in queue
                and (entity[0], entity[1], entity[2]) not in queue
            ):
                queue.append(entity)
                if (entity[0], entity[1]) in all_possible_distractors:
                    distractors.append(entity)

    # Create a dictionary to store the distractors by level
    distractors_by_level = {}
    for d in distractors:
        if d[2] not in distractors_by_level:
            distractors_by_level[d[2]] = []
        distractors_by_level[d[2]].append((d[0], d[1]))

    return distractors_by_level


def generate_distractor_candidates_multihop(
    kg: KnowledgeGraph,
    i: str,
    i_type: str,
    predicate1: str,
    direction1: str,
    j: str,
    j_type: str,
    predicate2: str,
    direction2: str,
    k: str,
    k_type: str,
    is_key_i: bool,
) -> list:
    if is_key_i:
        if direction2 == "->":
            q = """
                MATCH (entity:`%s`)
                WHERE (entity)-[:`%s`]->(:`%s` {name: "%s"})
                RETURN entity.name as mid_entity
            """ % (
                j_type,
                predicate2,
                k_type,
                k,
            )
        elif direction2 == "<-":
            q = """
                MATCH (entity:`%s`)
                WHERE (entity)<-[:`%s`]-(:`%s` {name: "%s"})
                RETURN entity.name as mid_entity
            """ % (
                j_type,
                predicate2,
                k_type,
                k,
            )

        mid_entities = kg.query(q)
        mid_entities = [r["mid_entity"] for r in mid_entities]

        # Find distractors for i such that they are of the same type as i but not connected to mid_entities by predicate1
        if direction1 == "->":
            distractor_query = """
                MATCH (d:`%s`)
                WHERE NOT EXISTS {
                    MATCH (d)-[:`%s`]->(mid:`%s`)
                    WHERE mid.name IN [%s]
                } AND d.name <> "%s"
                RETURN d.name as distractor, labels(d)[0] as distractor_label
            """ % (
                i_type,
                predicate1,
                j_type,
                ", ".join(
                    ['"%s"' % m if '"' not in m else "'%s'" % m for m in mid_entities]
                ),
                i,
            )
        elif direction1 == "<-":
            distractor_query = """
                MATCH (d:`%s`)
                WHERE NOT EXISTS {
                    MATCH (d)<-[:`%s`]-(mid:`%s`)
                    WHERE mid.name IN [%s]
                } AND d.name <> "%s"
                RETURN d.name as distractor, labels(d)[0] as distractor_label
            """ % (
                i_type,
                predicate1,
                j_type,
                ", ".join(
                    ['"%s"' % m if '"' not in m else "'%s'" % m for m in mid_entities]
                ),
                i,
            )

    else:  # is_key_k
        if direction1 == "->":
            q = """
                MATCH (entity:`%s`)
                WHERE (:`%s` {name: "%s"})-[:`%s`]->(entity)
                RETURN entity.name as mid_entity
            """ % (
                j_type,
                i_type,
                i,
                predicate1,
            )
        elif direction1 == "<-":
            q = """
                MATCH (entity:`%s`)
                WHERE (:`%s` {name: "%s"})<-[:`%s`]-(entity)
                RETURN entity.name as mid_entity
            """ % (
                j_type,
                i_type,
                i,
                predicate1,
            )

        mid_entities = kg.query(q)
        mid_entities = [r["mid_entity"] for r in mid_entities]

        # Find distractors for k such that they are of the same type as k but not connected to mid_entities by predicate2
        if direction2 == "->":
            distractor_query = """
                MATCH (d:`%s`)
                WHERE NOT EXISTS {
                    MATCH (mid:`%s`)-[:`%s`]->(d)
                    WHERE mid.name IN [%s]
                } AND d.name <> "%s"
                RETURN d.name as distractor, labels(d)[0] as distractor_label
            """ % (
                k_type,
                j_type,
                predicate2,
                ", ".join(
                    ['"%s"' % m if '"' not in m else "'%s'" % m for m in mid_entities]
                ),
                k,
            )
        elif direction2 == "<-":
            distractor_query = """
                MATCH (d:`%s`)
                WHERE NOT EXISTS {
                    MATCH (mid:`%s`)<-[:`%s`]-(d)
                    WHERE mid.name IN [%s]
                } AND d.name <> "%s"
                RETURN d.name as distractor, labels(d)[0] as distractor_label
            """ % (
                k_type,
                j_type,
                predicate2,
                ", ".join(
                    ['"%s"' % m if '"' not in m else "'%s'" % m for m in mid_entities]
                ),
                k,
            )

    distractors = kg.query(distractor_query)
    distractors = [(r["distractor_label"], r["distractor"]) for r in distractors]
    if len(distractors) < 3:
        print("Not enough distractors found. Could not generate a valid question.")
    return distractors


def generate_distractors_by_level_multihop(
    kg: KnowledgeGraph,
    max_level: int,
    i: str,
    i_type: str,
    predicate1: str,
    direction1: str,
    j: str,
    j_type: str,
    predicate2: str,
    direction2: str,
    k: str,
    k_type: str,
    is_key_i: bool,
) -> dict:
    all_possible_distractors = generate_distractor_candidates_multihop(
        kg,
        i,
        i_type,
        predicate1,
        direction1,
        j,
        j_type,
        predicate2,
        direction2,
        k,
        k_type,
        is_key_i,
    )

    # BFS
    distractors = []
    visited = set()
    queue = []
    if is_key_i:
        queue.append((i_type, i, 0))
    else:
        queue.append((k_type, k, 0))

    while queue:
        current = queue.pop(0)
        if (current[0], current[1]) in visited:
            continue
        visited.add((current[0], current[1]))

        if current[2] >= max_level:
            break

        if '"' in current[1]:
            q = """
                MATCH (n:`%s` {name: '%s'})-[]-(entity)
                RETURN entity.name as distractor, labels(entity)[0] as distractor_label
            """ % (
                current[0],
                current[1],
            )
        else:
            q = """
                MATCH (n:`%s` {name: "%s"})-[]-(entity)
                RETURN entity.name as distractor, labels(entity)[0] as distractor_label
            """ % (
                current[0],
                current[1],
            )
        result = kg.query(q)
        entities = [
            (r["distractor_label"], r["distractor"], current[2] + 1) for r in result
        ]
        for entity in entities:
            if (
                (entity[0], entity[1]) not in visited
                and (entity[0], entity[1], current[2]) not in queue
                and (entity[0], entity[1], entity[2]) not in queue
            ):
                queue.append(entity)
                if (entity[0], entity[1]) in all_possible_distractors:
                    distractors.append(entity)

    # Create a dictionary to store the distractors by level
    distractors_by_level = {}
    for d in distractors:
        if d[2] not in distractors_by_level:
            distractors_by_level[d[2]] = []
        distractors_by_level[d[2]].append((d[0], d[1]))

    return distractors_by_level

