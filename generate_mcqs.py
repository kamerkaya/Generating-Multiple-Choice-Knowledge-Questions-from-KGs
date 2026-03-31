import json
import random
import numpy as np

from utils.kg import KnowledgeGraph
from utils.llm import LLM
from utils.distractors import (
    generate_distractors_by_level,
    generate_distractors_by_level_multihop,
)


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product / (magnitude_a * magnitude_b)


def format_mcq(
    distractors_ordered_by_difficulty_with_level,
    key,
    key_label,
    question,
    degree_centralities,
    page_ranks,
    centrality_stats,
    is_multihop,
    max_level,
    helper_triple_used,
    triple=None,
    quintuple=None,
    hidden_middle_entity=None,
    helper_triple=None,
):
    # Get the centrality statistics
    degree_centrality_stats = centrality_stats["degree"]
    pageRank_centrality_stats = centrality_stats["pageRank"]

    # Create the distractors that stores them like (distractor, level)
    distractors = []
    for level in distractors_ordered_by_difficulty_with_level:
        for distractor in distractors_ordered_by_difficulty_with_level[level]:
            distractors.append((distractor, level))

    # Randomly select 3 of the distractors
    distractors = random.sample(distractors, 3)

    # Calculate the cosine similarity between the key's and the distractors' node embeddings
    key_embedding = kg.get_node_embedding(key, key_label)
    distractor_embeddings = [
        kg.get_node_embedding(distractor[0][1], distractor[0][0])
        for distractor in distractors
    ]
    node_embedding_similarity_scores = [
        cosine_similarity(key_embedding, distractor_embedding)
        for distractor_embedding in distractor_embeddings
    ]

    # Generate the MCQ from the question and the distractors
    mcq = {"stem": question, "A": "", "B": "", "C": "", "D": ""}

    # Randomly sample one of the letters from A, B, C, D for the correct answer
    correct_answer = random.choice(["A", "B", "C", "D"])
    mcq[correct_answer] = key
    wrong_answers = [distractor[0][1] for distractor in distractors]
    i = 0
    while wrong_answers:
        if mcq[chr(ord("A") + i)] == "":
            mcq[chr(ord("A") + i)] = wrong_answers.pop(0)
        i += 1

    # Add the correct answer to the MCQ
    mcq["correct_answer"] = correct_answer

    # Add the hidden middle entity to the MCQ if it is a multihop question
    if hidden_middle_entity:
        mcq["hidden_middle_entity"] = hidden_middle_entity

    # Add is_multihop to the MCQ
    mcq["is_multihop"] = is_multihop

    # Add the helper triple used to the MCQ
    mcq["helper_triple_used"] = helper_triple_used

    # Add the degree centralities
    mcq["degree_centralities"] = degree_centralities

    # Add the page ranks
    mcq["page_ranks"] = [round(page_rank, 2) for page_rank in page_ranks]

    # Add the depths of the distractors to the MCQ
    distractor_depths = [distractor[1] for distractor in distractors]
    mcq["distractor_depths"] = distractor_depths

    # Add the node embedding similarity scores to the MCQ
    mcq["node_embedding_similarity_scores"] = [
        round(score, 2) for score in node_embedding_similarity_scores
    ]

    # Add the source triple/quintuple to the MCQ
    if triple:
        mcq["triple"] = triple
    elif quintuple:
        mcq["quintuple"] = quintuple
    else:
        raise ValueError("Either triple or quintuple should be provided.")

    # Add the helper triple to the MCQ
    if helper_triple:
        mcq["helper_triple"] = helper_triple

    print("\nMCQ:")
    print(json.dumps(mcq, indent=2, ensure_ascii=False))

    return mcq


def ask_llm(question: str, answer: str, llm: LLM=None):
    if llm is None:
        llm = LLM()
    prompt = f"Question: {question}\nAnswer: {answer}\n\nIs the given answer to the given question correct or not? Answer ONLY with CORRECT if it is correct. Answer ONLY with FALSE if it is not correct."
    print(prompt)
    return llm.call(prompt).strip()


def validate_mcq(mcq, llm: LLM=None):
    if llm is None:
        llm = LLM()
    stem = mcq["stem"]
    correct_answer = mcq["correct_answer"]
    faulty = False
    for option in ["A", "B", "C", "D"]:
        if option != correct_answer:
            answer = mcq[option]
            result = ask_llm(stem, answer, llm)
            print(f"LLM response for option {option}: {result}")
            if "CORRECT" in result:
                faulty = True
                print(f"Faulty MCQ detected for question:\n{stem}\nOption {option}: {mcq[option]}\nis also correct.")
                break
    if faulty:
        mcq["valid"] = False
        print("MCQ is faulty.")
    else:
        mcq["valid"] = True
        print("MCQ is valid.")
    return mcq


def mcq_singlehop(
    item_name: str,
    lbl: str,
    kg: KnowledgeGraph,
    centrality_stats: dict,
    question_generation_llm: str,
    max_depth_for_bfs: int = 5,
    retry=3,
) -> dict:
    mcq = {}

    q = (
        """
        MATCH (i)-[predicate]-(j)
        WHERE i.name = "%s" AND labels(i)[0] = "%s"
            AND i.name <> j.name
            AND NONE(label IN labels(i) WHERE label="Document")
            AND NONE(label IN labels(j) WHERE label="Document")
        RETURN i.name as i,
            labels(i) as i_labels,
            type(predicate) as predicate,
            CASE WHEN startNode(predicate) = i THEN '->' ELSE '<-' END as direction,
            j.name as j,
            labels(j) as j_labels
        ORDER BY rand()
        LIMIT 1
    """
        % (item_name, lbl)
    )
    result = kg.query(q)
    if not result:
        print(
            "This item is not a part of any triple in the KG. Hence, it cannot be used to generate MCQs."
        )
        return {}

    # Extract the triple
    triple = result[0]
    i = triple["i"]
    i_labels = triple["i_labels"]
    predicate = triple["predicate"]
    direction = triple["direction"]
    j = triple["j"]
    j_labels = triple["j_labels"]

    i_label = i_labels[0]
    j_label = j_labels[0]

    # Set answer
    answer = i
    answer_label = i_label

    # Generate distractors
    if direction == "->":
        distractors_by_level = generate_distractors_by_level(
            kg, max_depth_for_bfs, i, i_label, predicate, j, j_label, is_key_subj=True
        )
    elif direction == "<-":
        distractors_by_level = generate_distractors_by_level(
            kg, max_depth_for_bfs, j, j_label, predicate, i, i_label, is_key_subj=False
        )
    else:
        raise ValueError(f"Invalid direction value: {direction}")

    # Check if enough distractors were generated
    num_distractors = sum(
        [len(distractors_by_level[lvl]) for lvl in distractors_by_level]
    )
    if num_distractors < 3:
        print("Not enough distractors were generated.")
        return {}

    # Generate the question stem
    prompt = f"""Generate a question stem for an MCQ from the triple: ({i}:({i_label}), {predicate}:({direction}), {j}:({j_label})) where the answer is {answer}. Do not mention the name of the answer ({answer}) inside the question stem. The distractors of the MCQ will also be of type {answer_label}. The distractors of the MCQ can be any {answer_label}, prepare the question stem accordingly. Please try to prepare the question stem like the ones that are asked in the TV show 'Who Wants to Be a Millionaire'.

    Some examples:
    Bradley Cooper spent six years learning how to conduct an orchestra for his role in what acclaimed 2023 film?
    Once upon a time in Massachusetts, 11-year-old Mary Sawyer is said to have inspired a nursery rhyme when she was followed to school by a what?
    Which of these activities did Ernest Hemingway describe as “the only art in which the artist is in danger of death”?
    Vatican City is the only place in the world where ATM instructions are written in what language?
    Who became the first Black songwriter to win Song of the Year at the Country Music Awards?
    Europe’s most active volcano, Mount Etna is known for puffing smoke rings into the sky over Sicily, earning it comparison to what movie character?
    Which of the following is a character from the Charles Dickens novel “Great Expectations” and not a former contestant on “RuPaul’s Drag Race”?
    In character as Ron Burgundy, Will Ferrell declared, “Tom is boring. Dink donk…touchdown, who cares!” at a 2024 Netflix roast of what celebrity?

    Output only the question stem."""
    llm = LLM(model=question_generation_llm)
    question_stem = (
        llm.call(prompt)
        .replace("<start_of_turn>", "")
        .replace("</start_of_turn>", "")
        .replace("<end_of_turn>", "")
        .replace("</end_of_turn>", "")
        .replace("<start_of_turn", "")
        .replace("</start_of_turn", "")
        .replace("<end_of_turn", "")
        .replace("</end_of_turn", "")
        .strip()
    )

    degree_centralities = [
        kg.get_degree_centrality(i, i_label),
        kg.get_degree_centrality(j, j_label),
    ]
    page_ranks = [
        kg.get_page_rank(i, i_label),
        kg.get_page_rank(j, j_label),
    ]

    # Format and validate the MCQ
    mcq = format_mcq(
        distractors_by_level,
        answer,
        answer_label,
        question_stem,
        degree_centralities,
        page_ranks,
        centrality_stats,
        False,
        max_depth_for_bfs,
        False,
        triple=(i, i_label, predicate, direction, j, j_label),
    )
    mcq = validate_mcq(mcq, llm)
    if mcq["valid"]:
        return mcq
    elif retry > 0:
        return mcq_singlehop(
            item_name,
            lbl,
            kg,
            centrality_stats,
            question_generation_llm,
            max_depth_for_bfs,
            retry - 1,
        )
    else:
        return {}


def mcq_singlehop_helper(
    item_name: str,
    lbl: str,
    kg: KnowledgeGraph,
    centrality_stats: dict,
    question_generation_llm: str,
    max_depth_for_bfs: int = 5,
    retry=3,
) -> dict:
    mcq = {}

    q = (
        """
        MATCH (helper)-[helper_predicate]-(key), (key)-[predicate]-(ent)
        WHERE key.name = "%s" AND labels(key)[0] = "%s"
            AND helper.name <> ent.name
            AND key.name <> ent.name
            AND key.name <> helper.name
            AND NONE(label IN labels(helper) WHERE label="Document")
            AND NONE(label IN labels(key) WHERE label="Document")
            AND NONE(label IN labels(ent) WHERE label="Document")
        RETURN key.name as key,
            labels(key) as key_labels,
            type(predicate) as predicate,
            CASE WHEN startNode(predicate) = key THEN '->' ELSE '<-' END as direction,
            ent.name as ent,
            labels(ent) as ent_labels,
            helper.name as helper,
            labels(helper) as helper_labels,
            type(helper_predicate) as helper_predicate,
            CASE WHEN startNode(helper_predicate) = key THEN '->' ELSE '<-' END as helper_direction
        ORDER BY rand()
        LIMIT 1
    """
        % (item_name, lbl)
    )
    result = kg.query(q)
    if not result:
        print(
            "This item cannot be used to create a single-hop MCQ with a helper triple."
        )
        return {}

    # Extract
    source = result[0]
    helper = source["helper"]
    helper_labels = source["helper_labels"]
    helper_predicate = source["helper_predicate"]
    helper_direction = source["helper_direction"]
    key = source["key"]
    key_labels = source["key_labels"]
    predicate = source["predicate"]
    direction = source["direction"]
    ent = source["ent"]
    ent_labels = source["ent_labels"]

    helper_label = helper_labels[0]
    key_label = key_labels[0]
    ent_label = ent_labels[0]

    # Generate distractors
    if direction == "->":
        distractors_by_level = generate_distractors_by_level(
            kg,
            max_depth_for_bfs,
            key,
            key_label,
            predicate,
            ent,
            ent_label,
            is_key_subj=True,
        )
    elif direction == "<-":
        distractors_by_level = generate_distractors_by_level(
            kg,
            max_depth_for_bfs,
            ent,
            ent_label,
            predicate,
            key,
            key_label,
            is_key_subj=False,
        )
    else:
        raise ValueError(f"Invalid direction value: {direction}")

    # Check if enough distractors were generated
    num_distractors = sum(
        [len(distractors_by_level[lvl]) for lvl in distractors_by_level]
    )
    if num_distractors < 3:
        print("Not enough distractors were generated.")
        return {}

    # Generate the question stem
    prompt = f"""Generate a question stem for an MCQ from the triple: ({key}:({key_label}), {predicate}:({direction}), {ent}:({ent_label})) where the answer is {key}. The question stem should also include following information about the answer {key}: ({key}:({key_label}), {helper_predicate}:({helper_direction}), {helper}:({helper_label})). Do not mention the name of the answer inside the question stem. The distractors of the MCQ will also be of type {key_label}. The distractors of the MCQ can be any {key_label}, prepare the question stem accordingly. Please try to prepare the question stem like the ones that are asked in the TV show 'Who Wants to Be a Millionaire'.

    Some examples:
    Bradley Cooper spent six years learning how to conduct an orchestra for his role in what acclaimed 2023 film?
    Once upon a time in Massachusetts, 11-year-old Mary Sawyer is said to have inspired a nursery rhyme when she was followed to school by a what?
    Which of these activities did Ernest Hemingway describe as “the only art in which the artist is in danger of death”?
    Vatican City is the only place in the world where ATM instructions are written in what language?
    Who became the first Black songwriter to win Song of the Year at the Country Music Awards?
    Europe’s most active volcano, Mount Etna is known for puffing smoke rings into the sky over Sicily, earning it comparison to what movie character?
    Which of the following is a character from the Charles Dickens novel “Great Expectations” and not a former contestant on “RuPaul’s Drag Race”?
    In character as Ron Burgundy, Will Ferrell declared, “Tom is boring. Dink donk…touchdown, who cares!” at a 2024 Netflix roast of what celebrity?

    Output only the question stem."""
    llm = LLM(model=question_generation_llm)
    question_stem = (
        llm.call(prompt)
        .replace("<start_of_turn>", "")
        .replace("</start_of_turn>", "")
        .replace("<end_of_turn>", "")
        .replace("</end_of_turn>", "")
        .replace("<start_of_turn", "")
        .replace("</start_of_turn", "")
        .replace("<end_of_turn", "")
        .replace("</end_of_turn", "")
        .strip()
    )

    degree_centralities = [
        kg.get_degree_centrality(ent, ent_label),
        kg.get_degree_centrality(key, key_label),
        kg.get_degree_centrality(helper, helper_label),
    ]
    page_ranks = [
        kg.get_page_rank(ent, ent_label),
        kg.get_page_rank(key, key_label),
        kg.get_page_rank(helper, helper_label),
    ]

    # Format and validate the MCQ
    mcq = format_mcq(
        distractors_by_level,
        key,
        key_label,
        question_stem,
        degree_centralities,
        page_ranks,
        centrality_stats,
        False,
        max_depth_for_bfs,
        True,
        triple=(key, key_label, predicate, direction, ent, ent_label),
        helper_triple=(
            key,
            key_label,
            helper_predicate,
            helper_direction,
            helper,
            helper_label,
        ),
    )
    mcq = validate_mcq(mcq, llm)
    if mcq["valid"]:
        return mcq
    elif retry > 0:
        return mcq_singlehop_helper(
            item_name,
            lbl,
            kg,
            centrality_stats,
            question_generation_llm,
            max_depth_for_bfs,
            retry - 1,
        )
    else:
        return {}


def mcq_doublehop(
    item_name: str,
    lbl: str,
    kg: KnowledgeGraph,
    centrality_stats: dict,
    question_generation_llm: str,
    max_depth_for_bfs: int = 5,
    retry=3,
) -> dict:
    mcq = {}

    q = (
        """
        MATCH (i)-[predicate1]-(j), (j)-[predicate2]-(k)
        WHERE i.name = "%s" AND labels(i)[0] = "%s"
            AND i.name <> k.name
            AND i.name <> j.name
            AND j.name <> k.name
            AND NONE(label IN labels(i) WHERE label="Document")
            AND NONE(label IN labels(j) WHERE label="Document")
            AND NONE(label IN labels(k) WHERE label="Document")
        RETURN i.name as i, 
            labels(i) as i_labels, 
            type(predicate1) as predicate1, 
            CASE WHEN startNode(predicate1) = i THEN '->' ELSE '<-' END as direction1,
            j.name as j, 
            labels(j) as j_labels, 
            type(predicate2) as predicate2, 
            CASE WHEN startNode(predicate2) = j THEN '->' ELSE '<-' END as direction2,
            k.name as k, 
            labels(k) as k_labels
        ORDER BY rand()
        LIMIT 1
    """
        % (item_name, lbl)
    )
    result = kg.query(q)
    if not result:
        print("This item cannot be used to create a multi-hop MCQ.")
        return {}

    # Extract
    quintuple = result[0]
    i = quintuple["i"]
    i_labels = quintuple["i_labels"]
    predicate1 = quintuple["predicate1"]
    direction1 = quintuple["direction1"]
    j = quintuple["j"]
    j_labels = quintuple["j_labels"]
    predicate2 = quintuple["predicate2"]
    direction2 = quintuple["direction2"]
    k = quintuple["k"]
    k_labels = quintuple["k_labels"]

    i_label = i_labels[0]
    j_label = j_labels[0]
    k_label = k_labels[0]

    # Set answer
    answer = i
    answer_label = i_label

    # Generate distractors
    distractors_by_level = generate_distractors_by_level_multihop(
        kg,
        max_depth_for_bfs,
        i,
        i_label,
        predicate1,
        direction1,
        j,
        j_label,
        predicate2,
        direction2,
        k,
        k_label,
        is_key_i=True,
    )

    # Check if enough distractors were generated
    num_distractors = sum(
        [len(distractors_by_level[lvl]) for lvl in distractors_by_level]
    )
    if num_distractors < 3:
        print("Not enough distractors were generated.")
        return {}

    # Generate the question stem
    prompt = f"""Generate a multi-hop question stem for an MCQ from the following: \n(\n  {i}:({i_label})  {predicate1}:({direction1})  {j}:({j_label})\n  {j}:({j_label})  {predicate2}:({direction2})  {k}:({k_label})\n)\nwhere the answer is {answer}. Do not mention the name of the answer ({answer}) inside the question stem. Do not mention the name of the entity in the middle ({j}) inside the question stem. The distractors of the MCQ will also be of type {answer_label}. The distractors of the MCQ can be any {answer_label}, prepare the question stem accordingly. Please try to prepare the question stem like the ones that are asked in the TV show 'Who Wants to Be a Millionaire'.

    Some examples:
    Bradley Cooper spent six years learning how to conduct an orchestra for his role in what acclaimed 2023 film?
    Once upon a time in Massachusetts, 11-year-old Mary Sawyer is said to have inspired a nursery rhyme when she was followed to school by a what?
    Which of these activities did Ernest Hemingway describe as “the only art in which the artist is in danger of death”?
    Vatican City is the only place in the world where ATM instructions are written in what language?
    Who became the first Black songwriter to win Song of the Year at the Country Music Awards?
    Europe’s most active volcano, Mount Etna is known for puffing smoke rings into the sky over Sicily, earning it comparison to what movie character?
    Which of the following is a character from the Charles Dickens novel “Great Expectations” and not a former contestant on “RuPaul’s Drag Race”?
    In character as Ron Burgundy, Will Ferrell declared, “Tom is boring. Dink donk…touchdown, who cares!” at a 2024 Netflix roast of what celebrity?

    Output only the question stem."""
    llm = LLM(model=question_generation_llm)
    question_stem = (
        llm.call(prompt)
        .replace("<start_of_turn>", "")
        .replace("</start_of_turn>", "")
        .replace("<end_of_turn>", "")
        .replace("</end_of_turn>", "")
        .replace("<start_of_turn", "")
        .replace("</start_of_turn", "")
        .replace("<end_of_turn", "")
        .replace("</end_of_turn", "")
        .strip()
    )

    degree_centralities = [
        kg.get_degree_centrality(i, i_label),
        kg.get_degree_centrality(j, j_label),
        kg.get_degree_centrality(k, k_label),
    ]
    page_ranks = [
        kg.get_page_rank(i, i_label),
        kg.get_page_rank(j, j_label),
        kg.get_page_rank(k, k_label),
    ]

    # Format and validate the MCQ
    mcq = format_mcq(
        distractors_by_level,
        answer,
        answer_label,
        question_stem,
        degree_centralities,
        page_ranks,
        centrality_stats,
        True,
        max_depth_for_bfs,
        False,
        quintuple=(
            i,
            i_label,
            predicate1,
            direction1,
            j,
            j_label,
            predicate2,
            direction2,
            k,
            k_label,
        ),
        hidden_middle_entity=j,
    )
    mcq = validate_mcq(mcq, llm)
    if mcq["valid"]:
        return mcq
    elif retry > 0:
        return mcq_doublehop(
            item_name,
            lbl,
            kg,
            centrality_stats,
            question_generation_llm,
            max_depth_for_bfs,
            retry - 1,
        )
    else:
        return {}


def mcq_doublehop_helper(
    item_name: str,
    lbl: str,
    kg: KnowledgeGraph,
    centrality_stats: dict,
    question_generation_llm: str,
    max_depth_for_bfs: int = 5,
    retry=3,
) -> dict:
    mcq = {}

    q = (
        """
        MATCH (ent)-[predicate1]-(middle), (middle)-[predicate2]-(key), (key)-[helper_predicate]-(helper)
        WHERE key.name = "%s" AND labels(key)[0] = "%s"
            AND ent.name <> middle.name
            AND ent.name <> key.name
            AND ent.name <> helper.name
            AND middle.name <> key.name
            AND middle.name <> helper.name
            AND key.name <> helper.name
            AND NONE(label IN labels(ent) WHERE label="Document")
            AND NONE(label IN labels(middle) WHERE label="Document")
            AND NONE(label IN labels(key) WHERE label="Document")
            AND NONE(label IN labels(helper) WHERE label="Document")
        RETURN ent.name as ent,
            labels(ent) as ent_labels,
            type(predicate1) as predicate1,
            CASE WHEN startNode(predicate1) = ent THEN '->' ELSE '<-' END as direction1,
            middle.name as middle,
            labels(middle) as middle_labels,
            type(predicate2) as predicate2,
            CASE WHEN startNode(predicate2) = middle THEN '->' ELSE '<-' END as direction2,
            key.name as key,
            labels(key) as key_labels,
            type(helper_predicate) as helper_predicate,
            CASE WHEN startNode(helper_predicate) = key THEN '->' ELSE '<-' END as helper_direction,
            helper.name as helper,
            labels(helper) as helper_labels
        ORDER BY rand()
        LIMIT 1
    """
        % (item_name, lbl)
    )
    result = kg.query(q)
    if not result:
        print(
            "This item cannot be used to create a multi-hop MCQ with a helper triple."
        )
        return {}

    # Extract
    source = result[0]
    ent = source["ent"]
    ent_labels = source["ent_labels"]
    predicate1 = source["predicate1"]
    direction1 = source["direction1"]
    middle = source["middle"]
    middle_labels = source["middle_labels"]
    predicate2 = source["predicate2"]
    direction2 = source["direction2"]
    key = source["key"]
    key_labels = source["key_labels"]
    helper_predicate = source["helper_predicate"]
    helper_direction = source["helper_direction"]
    helper = source["helper"]
    helper_labels = source["helper_labels"]

    helper_label = helper_labels[0]
    key_label = key_labels[0]
    middle_label = middle_labels[0]
    ent_label = ent_labels[0]

    # Generate distractors
    distractors_by_level = generate_distractors_by_level_multihop(
        kg,
        max_depth_for_bfs,
        ent,
        ent_label,
        predicate1,
        direction1,
        middle,
        middle_label,
        predicate2,
        direction2,
        key,
        key_label,
        is_key_i=False,
    )

    # Check if enough distractors were generated
    num_distractors = sum(
        [len(distractors_by_level[lvl]) for lvl in distractors_by_level]
    )
    if num_distractors < 3:
        print("Not enough distractors were generated.")
        return {}

    # Generate the question stem
    prompt = f"""Generate a multi-hop question stem for an MCQ from the following: \n(\n  {ent}:({ent_label})  {predicate1}:({direction1})  {middle}:({middle_label})\n  {middle}:({middle_label})  {predicate2}:({direction2})  {key}:({key_label})\n)\nwhere the answer is {key}. The question stem should also include following information about the answer ({key}): \n(\n  {key}:({key_label})  {helper_predicate}:({helper_direction})  {helper}:({helper_label})\n)\nDo not mention the name of the answer ({key}) inside the question stem. Do not mention the name of the entity in the middle ({middle}) inside the question stem. The distractors of the MCQ will also be of type {key_label}. The distractors of the MCQ can be any {key_label}, prepare the question stem accordingly. Please try to prepare the question stem like the ones that are asked in the TV show 'Who Wants to Be a Millionaire'.

    Some examples:
    Bradley Cooper spent six years learning how to conduct an orchestra for his role in what acclaimed 2023 film?
    Once upon a time in Massachusetts, 11-year-old Mary Sawyer is said to have inspired a nursery rhyme when she was followed to school by a what?
    Which of these activities did Ernest Hemingway describe as “the only art in which the artist is in danger of death”?
    Vatican City is the only place in the world where ATM instructions are written in what language?
    Who became the first Black songwriter to win Song of the Year at the Country Music Awards?
    Europe’s most active volcano, Mount Etna is known for puffing smoke rings into the sky over Sicily, earning it comparison to what movie character?
    Which of the following is a character from the Charles Dickens novel “Great Expectations” and not a former contestant on “RuPaul’s Drag Race”?
    In character as Ron Burgundy, Will Ferrell declared, “Tom is boring. Dink donk…touchdown, who cares!” at a 2024 Netflix roast of what celebrity?

    Output only the question stem."""
    llm = LLM(model=question_generation_llm)
    question_stem = (
        llm.call(prompt)
        .replace("<start_of_turn>", "")
        .replace("</start_of_turn>", "")
        .replace("<end_of_turn>", "")
        .replace("</end_of_turn>", "")
        .replace("<start_of_turn", "")
        .replace("</start_of_turn", "")
        .replace("<end_of_turn", "")
        .replace("</end_of_turn", "")
        .strip()
    )

    degree_centralities = [
        kg.get_degree_centrality(ent, ent_label),
        kg.get_degree_centrality(middle, middle_label),
        kg.get_degree_centrality(key, key_label),
        kg.get_degree_centrality(helper, helper_label),
    ]
    page_ranks = [
        kg.get_page_rank(ent, ent_label),
        kg.get_page_rank(middle, middle_label),
        kg.get_page_rank(key, key_label),
        kg.get_page_rank(helper, helper_label),
    ]

    # Format and validate the MCQ
    mcq = format_mcq(
        distractors_by_level,
        key,
        key_label,
        question_stem,
        degree_centralities,
        page_ranks,
        centrality_stats,
        True,
        max_depth_for_bfs,
        True,
        quintuple=(
            ent,
            ent_label,
            predicate1,
            direction1,
            middle,
            middle_label,
            predicate2,
            direction2,
            key,
            key_label,
        ),
        hidden_middle_entity=middle,
        helper_triple=(
            key,
            key_label,
            helper_predicate,
            helper_direction,
            helper,
            helper_label,
        ),
    )
    mcq = validate_mcq(mcq, llm)
    if mcq["valid"]:
        return mcq
    elif retry > 0:
        return mcq_doublehop_helper(
            item_name,
            lbl,
            kg,
            centrality_stats,
            question_generation_llm,
            max_depth_for_bfs,
            retry - 1,
        )
    else:
        return {}


def generate_mcqs_for_item(
    item_name: str,
    lbl: str,
    kg: KnowledgeGraph,
    centrality_stats: dict,
    question_generation_llm: str,
    max_depth_for_bfs: int = 5,
) -> list:
    # Generate MCQs for a specific item in the KG
    mcqs = []

    # Generate a single-hop MCQ without helper triple
    mcq = mcq_singlehop(
        item_name, lbl, kg, centrality_stats, question_generation_llm, max_depth_for_bfs
    )
    if mcq:
        mcqs.append(mcq)
    else:
        print("Single-hop MCQ without helper triple could not be generated.")
        print("This item cannot be used to generate MCQs.")
        return mcqs  # Return empty list

    # Generate a single-hop MCQ with helper triple
    mcq = mcq_singlehop_helper(
        item_name, lbl, kg, centrality_stats, question_generation_llm, max_depth_for_bfs
    )
    if mcq:
        mcqs.append(mcq)
    else:
        print("Single-hop MCQ with helper triple could not be generated.")
        print(
            "(This item is not the start node of any quintuple in the KG. Hence, it can only be used to generate single-hop MCQs with no helper triple.)"
        )

    # Generate a multi-hop MCQ without helper triple
    mcq = mcq_doublehop(
        item_name, lbl, kg, centrality_stats, question_generation_llm, max_depth_for_bfs
    )
    if mcq:
        mcqs.append(mcq)
    else:
        print("Multi-hop MCQ without helper triple could not be generated.")
        print(
            "(This item is not the start node of any quintuple in the KG. Hence, it can only be used to generate single-hop MCQs with no helper triple.)"
        )

    # Generate a multi-hop MCQ with helper triple
    mcq = mcq_doublehop_helper(
        item_name, lbl, kg, centrality_stats, question_generation_llm, max_depth_for_bfs
    )
    if mcq:
        mcqs.append(mcq)
    else:
        print("Multi-hop MCQ with helper triple could not be generated.")

    return mcqs


if __name__ == "__main__":
    # Load the KG
    kg: KnowledgeGraph = KnowledgeGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        max_connection_lifetime=3600,
    )

    # Find the connected components in the KG
    wcc_stats = kg.wcc()
    print("\nConnected Components Statistics of the KG:")
    print("Number of Connected Components:", len(wcc_stats))
    print("Largest Connected Component Size:", wcc_stats[0]["componentSize"])

    print("\nConnected Components:")
    print(json.dumps(wcc_stats, indent=2, ensure_ascii=False))

    # Calculate the node centralities of the entities in the KG
    centrality_stats = kg.calc_node_centralities()
    print("\nCentrality Statistics of the KG:")
    print(centrality_stats)

    # Calculate node embeddings
    embedding_dim = 256  # The dimension of the node embeddings, paper says higher dimensions are better, but 256 is OK
    kg.calc_node_embeddings(embedding_dim=embedding_dim, random_seed=42)
    print(
        f"\nNode embeddings calculated with an embedding dimension of {embedding_dim}."
    )

    # Set the name of the LLM to use for question generation
    question_generation_llm = "gpt-4o"

    # Get the nodes from the KG
    num_nodes = 40 # Generate MCQs for the most popular {num_nodes} nodes
    nodes = kg.get_all_nodes()
    nodes = nodes[:num_nodes]
    print(f"\nGenerating MCQs for the following {len(nodes)} nodes:")
    for index, node in enumerate(nodes):
        print(index, node)

    for node in nodes:
        node_name, node_label = node

        print(f"\nGenerating MCQs for '{node_name}' ({node_label})...")

        # Generate MCQs for a specific item in the KG
        mcqs = generate_mcqs_for_item(
            node_name, node_label, kg, centrality_stats, question_generation_llm
        )
        print("\nMCQs for '%s':" % node_name)
        print(json.dumps(mcqs, indent=4, ensure_ascii=False))

        # Save the MCQs to a JSON file
        node_name_cleaned = node_name.replace("/", " ")
        node_label_cleaned = node_label.replace("/", " ")
        with open(f"mcqs/{node_name_cleaned}_{node_label_cleaned}.json", "w") as f:
            json.dump(mcqs, f, indent=4, ensure_ascii=False)

    # Close the KG
    kg.close()

