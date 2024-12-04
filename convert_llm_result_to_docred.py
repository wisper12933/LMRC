import json
import re
from sentence_transformers import SentenceTransformer, util
from thefuzz import fuzz
from tqdm import tqdm

# dataset
relations = 'your path to dataset/rel_info.json'
truth_path = 'your path to dataset/dev.json or test.json'

# text-based result from RC
result_path = ''

# structured result for evaluation
mapped_path = ''

# if sampled, select_path will get the id of sampled documents
select_path = ''

# id_path will get the corresponding document id of the input result
id_path = ''
log_path = './logs/your log path'
add_align = False
un_finetuned = False

model = SentenceTransformer('all-mpnet-base-v2')

truth = json.loads(open(truth_path).read())
ids = json.loads(open(id_path).read())
log_file = open(log_path, 'w')

if select_path:
    idx = json.loads(open(select_path).read())
    truth = [truth[i] for i in idx]

rel = json.loads(open(relations).read())
relation_set = [v for _, v in rel.items()]
twisted = {v: k for k, v in rel.items()}
relation_emb = model.encode(relation_set)

wrong_num = 0


def get_combinations(cosine_i):
    all_sentence_combinations = []
    for i in range(len(cosine_i)):
        combinations_temp = []
        for j in range(len(cosine_i[i])):
            combinations_temp.append([cosine_i[i][j], j])
        all_sentence_combinations.append(combinations_temp)

    sorted_combinations = []
    for combinations_temp in all_sentence_combinations:
        combinations_temp = sorted(combinations_temp, key=lambda x: x[0], reverse=True)
        sorted_combinations.append(combinations_temp)

    return sorted_combinations


def extract_inner_strings(input_str):
    """
    :param input_str: LLM extract result
    :return: List[str]——unformat_triples, get——Is the raw answer valid
    """
    get = True
    pattern = re.compile(r'\(([^)]+)\)')
    matches = pattern.findall(input_str)
    matches = [match.lstrip('(').strip('"') for match in matches]

    if not matches:
        get = False

    return matches, get


def parse_string(input_str, unfinetuned=False):
    """
    :param unfinetuned: unfinetuned model may generate some other formats
    :param input_str: unformat_triple
    :return: dict format_triple
    """
    first_comma_index = input_str.find('|')
    second_comma_index = input_str.find('|', first_comma_index + 1)

    if second_comma_index == -1:
        return second_comma_index

    head = input_str[:first_comma_index].strip()
    tail = input_str[second_comma_index + 1:].strip()
    relation = input_str[first_comma_index + 1:second_comma_index].strip()

    special_ids = relation.find(':')
    if special_ids != -1:
        relation = relation[special_ids + 1:].strip().strip('\'').strip('"')
    else:
        relation = relation.strip('\'').strip('"')

    if unfinetuned:
        match = re.search(r'[:\\/]\s*["\']?([^"\']+)["\']?', relation)
        if match:
            relation = match.group(1).strip()

    result_dict = {'head': head, 'relation': relation, 'tail': tail}
    return result_dict


def map_to_vertex(input_str, str_list):
    """
    :param str_list: list[str] named_entity_list
    :param input_str: entity_name
    :return: int entity_index
    """
    max_similarity = 0
    most_similar_vertex = 0

    for i, _str in enumerate(str_list):
        similarity = fuzz.ratio(input_str, _str)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_vertex = i
    return most_similar_vertex


def mapping(triple, str_list, title, align=False, unfinetuned=False):
    """
    :param unfinetuned: unfinetuned model may generate some other formats
    :param title: title of document
    :param align: align to relation map
    :param str_list: list[str] named_entity_list
    :param triple: triplet dict
    :return: dict mapped_triplet
    """
    global wrong_num
    if triple['relation'] == 'None':
        return None

    if unfinetuned:
        if triple['relation'] == '-':
            return None

    r = twisted.get(triple['relation'])
    if r is None:
        wrong_num += 1
        if not align:
            r = 'out of domain'
        else:
            r_emb = model.encode(triple['relation'])
            cosine = util.cos_sim(r_emb, relation_emb)
            sorted_comb = get_combinations(cosine)
            r = relation_set[sorted_comb[0][0][1]]
            r = twisted[r]

    mapped_triple = {'title': title,
                     'h_idx': map_to_vertex(triple['head'], str_list),
                     't_idx': map_to_vertex(triple['tail'], str_list),
                     'r': r,
                     'evidence': []}
    return mapped_triple


def main(align=False, unfinetuned=False):
    """
    :param unfinetuned: unfinetuned model may generate some other formats
    :param align: align to relation map
    """
    with open(result_path, 'r') as rf:
        results = json.load(rf)

    global wrong_num
    results = [d['text'] for d in results]
    mapped_re, extract_num = [], []
    count = 0
    for index, result in enumerate(tqdm(results)):
        vertexSet = truth[ids[index]]['vertexSet']
        title = truth[ids[index]]['title']
        named_en = []

        if result is None:
            continue

        for vertex in vertexSet:
            named_en.append(vertex[0]['name'])

        sequence_re, is_valid = extract_inner_strings(result)
        if not is_valid:
            count += 1

        format_re = []
        for triple in sequence_re:
            format_triple = parse_string(triple, unfinetuned)
            if format_triple == -1:
                continue
            format_re.append(format_triple)

        extract_num.append(len(format_re))

        mapped_re.extend([mapping(triple, named_en, title, align, unfinetuned) for triple in format_re])

    mapped_re = [i for i in mapped_re if i is not None]
    print(f'Number of triplets: {sum(extract_num)}')
    print(f'Average number of extractions: {sum(extract_num) / len(extract_num)}')
    print(f'Invalid LLM generations: {count}')
    print(f'Wrong relation number: {wrong_num}')
    print(len(mapped_re))

    log_file.write(f'Number of triplets: {sum(extract_num)}\n')
    log_file.write(f'Average number of extractions: {sum(extract_num) / len(extract_num)}\n')
    log_file.write(f'Invalid LLM generations: {count}\n')
    log_file.write(f'Wrong relation number: {wrong_num}\n')
    log_file.write(f'Num of Triplets extracted: {len(mapped_re)}\n')

    with open(mapped_path, 'w') as wf:
        json.dump(mapped_re, wf)


if __name__ == '__main__':
    main(add_align, un_finetuned)
