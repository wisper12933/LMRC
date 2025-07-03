import json
import random
from collections import defaultdict

from tqdm import tqdm

# 数据集相关文件路径
rel_info_path = 'your path to dataset/rel_info.json'
input_train_path = 'your path to dataset/train_annotated.json'
dev_set_path = 'your path to dataset/dev.json'
test_set_path = 'your path to dataset/test.json'

# 输出文件路技能
# idx_path: 切分后的query所对应的document编号, 编号存储于该路径便于evaluate
# output_path: 输出路径
idx_path = ''
output_path = ''

rel_info = json.loads(open(rel_info_path).read())
relation_set = []
for k, v in rel_info.items():
    relation_set.append(v)

# 模板
PROMPT_DICT = {
    "zero_shot_instruction": (
        "Below is an instruction that describes a task, paired with information that provides further context.Write a "
        "response that appropriately completes the request."
        "\n\n### Instruction:\nThis is a document-level relation extraction task. we will provide entity pairs that "
        "require relation extraction. Your task is to select a relation for each entity pair from the given relation "
        "set based on the information in the text.The format of input entity pairs is ’(head entity, -, tail entity)’."
        "Your output format is ’(head entity, relation, tail entity)’."
        "\n\n### Relation set:\n{relation_set}"
    ),
    "input": (
        "### Text:\n{text}\n\n### {e_num} Entity pairs:\n{e_pairs}"
    ),
    "few_shot_instruction": (
        "Below is an instruction that describes a task, paired with information and {n_shot} examples that provide "
        "further context. Write a response that appropriately completes the request."
        "\n\n### Instruction:\nThis is a document-level relation extraction task. We will provide text and entity "
        "pairs that require relation extraction. Your task is to determine whether there are relations between the "
        "entity pairs based on the information in the text. If there are relations, select relation for the entity "
        "pair from the relation set; if there is no relation, return None. "
        "\n The format of the input entity pair is ’(head entity| -| tail entity)’."
        "\n Your output format is ’(head entity| relation/None| tail entity)’. "
        "\n\n### Relation set:\n{relation_set}"
    ),
    "examples": (
        "### Example {num}:\n## Text:\n{text}\n## {e_num} Entity pairs:\n{e_pairs}\n## Output:\n{truth}"
    ),
    "few_shot_input": (
        "### Input:\n## Text:\n{text}\n## {e_num} Entity pairs:\n{e_pairs}\n## Output:"
    ),
    "full_re_instruction_ft": (
        "This is a document-level relation extraction task. We will provide text and entity pairs that require "
        "relation extraction. Your task is to determine whether there are relations between the entity pairs based on "
        "the information in the text. If there are relations, select relation for the entity pair from the relation "
        "set; if there is no relation, return None. "
        "\n The format of the input entity pair is ’(head entity| -| tail entity)’."
        "\n Your output format is ’(head entity| relation/None| tail entity)’. "
        "\n\n### Relation set:\n{relation_set}"
    ),
    "full_re_instruction_ft_revise": (
        "This is a document-level relation extraction task. We will provide text and entity pairs that require "
        "relation extraction. Your task is to determine whether there are relations between the entity pairs based on "
        "the information in the text. If there are relations, select relation for the entity pair from the relation "
        "set, and there may be multiple relations between an entity pair; if there is no relation, return None. "
        "\n The format of the input entity pair is ’(head entity| -| tail entity)’."
        "\n Your output format is ’(head entity| relation/None| tail entity)’. "
        "\n\n### Relation set:\n{relation_set}"
    ),
    "step1_instruction_ft": (
        "We will provide the text and several entity pairs in the text. You need to determine whether these entity "
        "pairs express relations based on the text. "
        "The format of input entity pairs is ’(head entity | tail entity)’. "
        "Return \"Yes\" if you believe the entity pair expresses relations, otherwise return \"No\". The number of "
        "\"Yes\" and \"No\" should be the same as the number of input entity pairs. "
    ),
    "step2_instruction_ft": (
        "This is a document-level relation extraction task. we will provide entity pairs that require relation "
        "extraction. Your task is to select relations for each entity pair from the given relation set based on the "
        "information in the text. There may be multiple relations between an entity pair. "
        "\nThe format of input entity pairs is ’(head entity| -| tail entity)’. "
        "\nYour output format is ’(head entity| relation| tail entity)’. "
        "\n\n### Relation set:\n{relation_set}"
    )
}


def convert_format(doc_dict, relation_map):
    labels = doc_dict['labels']
    vertexSet = doc_dict['vertexSet']
    sents = doc_dict['sents']

    vertex, rel_types = [], []

    for entity in vertexSet:
        vertex.append(entity[0]['name'])

    e_pairs, truth = [], []
    for label in labels:
        n1 = vertexSet[label['h']][0]['name']
        n2 = vertexSet[label['t']][0]['name']
        r = relation_map[label['r']]

        if f'({n1}| -| {n2})' not in e_pairs:
            e_pairs.append(f'({n1}| -| {n2})')
        truth.append(f'({n1}| {r}| {n2})')
        if r not in rel_types:
            rel_types.append(r)

    concat_sents = [' '.join(words) for words in sents]
    concat_doc = ' '.join(concat_sents)

    return concat_doc, e_pairs, truth, rel_types


###
# generate_n_shot_prompt: 生成用于测试GPT的query
# generate_zero_shot_prompt: 生成用于测试GPT的query
# generate_for_step1_finetune: 生成用于微调 RCP 阶段的数据
# generate_for_step2_finetune: 生成用于微调 RC 阶段的数据
# generate_for_re_finetune: 生成用于微调整个 Document-level RE 过程的数据
###
def generate_n_shot_prompt(rel_path, dev_path, anno_path,
                           out_path,
                           select_idx_path='',
                           n_shot=3,
                           split=36):
    """
    构造n_shot提示以输入LLM
    :param split:
    :param select_idx_path: 保存所选择案例编号的文件路径
    :param n_shot: n-shot
    :param rel_path:关系标签存储文件路径
    :param dev_path:开发集路径
    :param anno_path:人工标注训练集路径
    :param out_path:输出路径
    """
    data = json.loads(open(anno_path).read())
    test = json.loads(open(dev_path).read())
    id2rel = json.loads(open(rel_path).read())

    outs, idxs, inputs = [], [], []

    for example in tqdm(data):
        sents = example['sents']
        doc_sents = ' '.join([' '.join(words) for words in sents])

        vertex = example['vertexSet']
        labels = example['labels']
        pair2rel = defaultdict(list)
        for label in labels:
            pair2rel[(label['h'], label['t'])].append(label['r'])

        en_num = len(vertex)
        e_pairs, outputs, flag = [], [], 0
        for h_idx in range(en_num):
            for t_idx in range(en_num):
                if h_idx != t_idx:
                    h_n = vertex[h_idx][0]['name']
                    t_n = vertex[t_idx][0]['name']
                    e_pairs.append(f'({h_n}| -| {t_n})')
                    if (h_idx, t_idx) in pair2rel:
                        flag = 1
                        for r in pair2rel[(h_idx, t_idx)]:
                            rel = id2rel[r]
                            outputs.append(f'({h_n}| {rel}| {t_n})')
                    else:
                        outputs.append(f'({h_n}| None| {t_n})')
                    # split to control token length
                    if len(e_pairs) == split:
                        if flag:
                            inputs.append(
                                {'text': doc_sents, 'e_num': split, 'e_pairs': '\n'.join(e_pairs),
                                 'truth': '\n'.join(outputs)}
                            )
                        e_pairs, outputs, flag = [], [], 0
        # add the remaining triplets to the data
        if flag:
            inputs.append(
                {'text': doc_sents, 'e_num': len(e_pairs), 'e_pairs': '\n'.join(e_pairs), 'truth': '\n'.join(outputs)}
            )

    examples_num = len(inputs)
    print("Total examples:", examples_num)
    if select_idx_path:
        select_idx = json.loads(open(select_idx_path).read())
        test = [test[i] for i in select_idx]
    for i, doc in enumerate(tqdm(test)):
        sents = doc['sents']
        doc_sents = ' '.join([' '.join(words) for words in sents])

        vertex = doc['vertexSet']
        en_num = len(vertex)
        e_pairs, outputs, flag = [], [], 0
        for h_idx in range(en_num):
            for t_idx in range(en_num):
                if h_idx != t_idx:
                    h_n = vertex[h_idx][0]['name']
                    t_n = vertex[t_idx][0]['name']
                    e_pairs.append(f'({h_n}| -| {t_n})')
                    if len(e_pairs) == split:
                        if select_idx_path:
                            idxs.append(select_idx[i])
                        else:
                            idxs.append(i)
                        instruction = PROMPT_DICT['few_shot_instruction'].format_map(
                            {'n_shot': n_shot, 'relation_set': relation_set})
                        examples = []
                        random_num = random.sample(range(0, examples_num), n_shot)
                        for j, n in enumerate(random_num):
                            e = PROMPT_DICT['examples'].format_map(
                                {'num': j + 1,
                                 'text': inputs[n]['text'],
                                 'e_num': inputs[n]['e_num'],
                                 'e_pairs': inputs[n]['e_pairs'],
                                 'truth': inputs[n]['truth']}
                            )
                            examples.append(e)
                        f_input = PROMPT_DICT['few_shot_input'].format_map(
                            {'text': doc_sents, 'e_num': split, 'e_pairs': '\n'.join(e_pairs)}
                        )

                        outs.append({
                            'instruction': instruction,
                            'examples': examples,
                            'input': f_input,
                        })
                        e_pairs, outputs = [], []
        if select_idx_path:
            idxs.append(select_idx[i])
        else:
            idxs.append(i)
        instruction = PROMPT_DICT['few_shot_instruction'].format_map(
            {'n_shot': n_shot, 'relation_set': relation_set})
        examples = []
        random_num = random.sample(range(0, examples_num), n_shot)
        for j, n in enumerate(random_num):
            e = PROMPT_DICT['examples'].format_map(
                {'num': j + 1,
                 'text': inputs[n]['text'],
                 'e_num': inputs[n]['e_num'],
                 'e_pairs': inputs[n]['e_pairs'],
                 'truth': inputs[n]['truth']}
            )
            examples.append(e)
        f_input = PROMPT_DICT['few_shot_input'].format_map(
            {'text': doc_sents, 'e_num': len(e_pairs), 'e_pairs': '\n'.join(e_pairs)}
        )

        outs.append({
            'instruction': instruction,
            'examples': examples,
            'input': f_input,
        })

    print("Total outs:", len(outs))

    with open(out_path, 'w') as fo:
        jsoned = json.dumps(outs)
        fo.write(jsoned)

    with open('../docred/idx/gpt_sample_100_expand_index.json', 'w') as fid:
        jsoned = json.dumps(idxs)
        fid.write(jsoned)


def generate_zero_shot_prompt(rel_path, path_in, path_out):
    data = json.loads(open(path_in).read())
    rel = json.loads(open(rel_path).read())
    # Store golden label and original input document
    data_dict_list = []

    for example in data:
        concat_sent, e_pairs, _, _ = convert_format(example, rel)

        data_dict_list.append({'text': concat_sent, 'e_num': len(e_pairs), 'e_pairs': '\n'.join(e_pairs)})

    prompt_input = PROMPT_DICT['input']
    sources = [
        prompt_input.format_map(_dict) for _dict in data_dict_list
    ]
    instruction = PROMPT_DICT['zero_shot_instruction'].format_map({'relation_set': relation_set})

    with open(path_out, 'w') as fo:
        for s in sources:
            jsoned = json.dumps({'instruction': instruction, 'input': s})
            fo.write(jsoned)
            fo.write('\n')


def generate_for_step1_finetune(anno_path, split, path_out, index_path='', test=False):
    data = json.loads(open(anno_path).read())

    datas, idxs = [], []

    for idx, example in enumerate(tqdm(data)):
        sents = example['sents']
        doc_sents = ' '.join([' '.join(words) for words in sents])

        vertex = example['vertexSet']
        labels = example['labels']
        pair2rel = defaultdict(list)
        for label in labels:
            pair2rel[(label['h'], label['t'])].append(label['r'])

        en_num = len(vertex)
        e_pairs, outputs, count = [], [], 0
        for h_idx in range(en_num):
            for t_idx in range(en_num):
                if h_idx != t_idx:
                    h_n = vertex[h_idx][0]['name']
                    t_n = vertex[t_idx][0]['name']
                    count += 1
                    e_pairs.append(f'{count}.({h_n} | {t_n})')
                    if (h_idx, t_idx) in pair2rel:
                        outputs.append(f'{count}.Yes')
                    else:
                        outputs.append(f'{count}.No')
                    # split to control token length
                    if len(e_pairs) == split:
                        idxs.append(idx)
                        instruction = PROMPT_DICT['step1_instruction_ft']
                        inputs = PROMPT_DICT['input'].format_map(
                            {'text': doc_sents, 'e_num': split, 'e_pairs': '\n'.join(e_pairs)}
                        )
                        if test:
                            datas.append({
                                'instruction': instruction,
                                'input': inputs,
                            })
                        else:
                            datas.append({
                                'instruction': instruction,
                                'input': inputs,
                                'output': '\n'.join(outputs)
                            })
                        e_pairs, outputs, count = [], [], 0
        # add the remaining triplets to the data
        idxs.append(idx)
        instruction = PROMPT_DICT['step1_instruction_ft']
        inputs = PROMPT_DICT['input'].format_map(
            {'text': doc_sents, 'e_num': len(e_pairs), 'e_pairs': '\n'.join(e_pairs)}
        )
        if test:
            datas.append({
                'instruction': instruction,
                'input': inputs,
            })
        else:
            datas.append({
                'instruction': instruction,
                'input': inputs,
                'output': '\n'.join(outputs)
            })

    print(len(datas))
    print(len(idxs))

    with open(path_out, 'w') as fo:
        jsoned = json.dumps(datas)
        fo.write(jsoned)

    with open(index_path, 'w') as fid:
        jsoned = json.dumps(idxs)
        fid.write(jsoned)


def generate_for_step2_finetune(anno_path, rel_path, path_out, index_path='', test=False, split=0):
    data = json.loads(open(anno_path).read())
    id2rel = json.loads(open(rel_path).read())

    if split:
        datas, idxs = [], []

        for idx, example in enumerate(tqdm(data)):
            sents = example['sents']
            doc_sents = ' '.join([' '.join(words) for words in sents])

            vertex = example['vertexSet']
            labels = example['labels']
            pair2rel = defaultdict(list)
            pairs = []
            for label in labels:
                if (label['h'], label['t']) not in pairs:
                    pairs.append((label['h'], label['t']))
                pair2rel[(label['h'], label['t'])].append(label['r'])
            e_pairs, outputs = [], []
            for pair in pairs:
                h_idx = pair[0]
                t_idx = pair[1]
                h_n = vertex[h_idx][0]['name']
                t_n = vertex[t_idx][0]['name']
                if f'({h_n}| -| {t_n})' not in e_pairs:
                    e_pairs.append(f'({h_n}| -| {t_n})')
                    for r in pair2rel[pair]:
                        rel = id2rel[r]
                        outputs.append(f'({h_n}| {rel}| {t_n})')
                # split to control token length
                if len(e_pairs) == split:
                    idxs.append(idx)
                    instruction = PROMPT_DICT['step2_instruction_ft'].format_map({'relation_set': relation_set})
                    inputs = PROMPT_DICT['input'].format_map(
                        {'text': doc_sents, 'e_num': split, 'e_pairs': '\n'.join(e_pairs)}
                    )
                    if test:
                        datas.append({
                            'instruction': instruction,
                            'input': inputs,
                        })
                    else:
                        datas.append({
                            'instruction': instruction,
                            'input': inputs,
                            'output': '\n'.join(outputs)
                        })
                    e_pairs, outputs = [], []
            # add the remaining triplets to the data
            idxs.append(idx)
            instruction = PROMPT_DICT['step2_instruction_ft'].format_map({'relation_set': relation_set})
            inputs = PROMPT_DICT['input'].format_map(
                {'text': doc_sents, 'e_num': len(e_pairs), 'e_pairs': '\n'.join(e_pairs)}
            )
            if test:
                datas.append({
                    'instruction': instruction,
                    'input': inputs,
                })
            else:
                datas.append({
                    'instruction': instruction,
                    'input': inputs,
                    'output': '\n'.join(outputs)
                })

        print(len(datas))
        print(len(idxs))

        with open(path_out, 'w') as fo:
            jsoned = json.dumps(datas)
            fo.write(jsoned)

        with open(index_path, 'w') as fid:
            jsoned = json.dumps(idxs)
            fid.write(jsoned)
    else:
        list_data_dict, outputs = [], []
        for example in tqdm(data):
            concat_sent, e_pairs, truth, _ = convert_format(example, id2rel)
            if e_pairs:
                re_pairs = [e_pairs[0]]
                for i in range(1, len(e_pairs)):
                    if e_pairs[i] != e_pairs[i - 1]:
                        re_pairs.append(e_pairs[i])
            else:
                re_pairs = []

            list_data_dict.append({'text': concat_sent, 'e_num': len(re_pairs), 'e_pairs': '\n'.join(re_pairs)})
            outputs.append('\n'.join(truth))

        prompt_input = PROMPT_DICT['input']
        inputs = [
            prompt_input.format_map(example) for example in list_data_dict
        ]
        if test:
            datas = [
                {'instruction': PROMPT_DICT['step2_instruction_ft'].format_map({'relation_set': relation_set}),
                 'input': inputs[idx]} for idx in range(len(inputs))
            ]
        else:
            datas = [
                {'instruction': PROMPT_DICT['step2_instruction_ft'].format_map({'relation_set': relation_set}),
                 'input': inputs[idx],
                 'output': outputs[idx]} for idx in range(len(inputs))
            ]
        with open(path_out, 'w') as fo:
            jsoned = json.dumps(datas)
            fo.write(jsoned)


def generate_for_re_finetune(anno_path, rel_path, index_path, split, path_out, test=False):
    data = json.loads(open(anno_path).read())
    id2rel = json.loads(open(rel_path).read())

    """sample"""
    data = data[:200]

    datas, idxs = [], []

    for idx, example in enumerate(tqdm(data)):
        sents = example['sents']
        doc_sents = ' '.join([' '.join(words) for words in sents])

        vertex = example['vertexSet']
        pair2rel = defaultdict(list)
        if not test:
            labels = example['labels']
            for label in labels:
                pair2rel[(label['h'], label['t'])].append(label['r'])

        en_num = len(vertex)
        e_pairs, outputs = [], []
        for h_idx in range(en_num):
            for t_idx in range(en_num):
                if h_idx != t_idx:
                    h_n = vertex[h_idx][0]['name']
                    t_n = vertex[t_idx][0]['name']
                    e_pairs.append(f'({h_n}| -| {t_n})')
                    if (h_idx, t_idx) in pair2rel:
                        for r in pair2rel[(h_idx, t_idx)]:
                            rel = id2rel[r]
                            outputs.append(f'({h_n}| {rel}| {t_n})')
                    else:
                        outputs.append(f'({h_n}| None| {t_n})')
                    # split to control token length
                    if len(e_pairs) == split:
                        idxs.append(idx)
                        instruction = PROMPT_DICT['full_re_instruction_ft'].format_map(
                            {'relation_set': relation_set})
                        inputs = PROMPT_DICT['input'].format_map(
                            {'text': doc_sents, 'e_num': split, 'e_pairs': '\n'.join(e_pairs)}
                        )
                        if test:
                            datas.append({
                                'instruction': instruction,
                                'input': inputs,
                            })
                        else:
                            datas.append({
                                'instruction': instruction,
                                'input': inputs,
                                'output': '\n'.join(outputs)
                            })
                        e_pairs, outputs = [], []
        # add the remaining triplets to the data
        idxs.append(idx)
        instruction = PROMPT_DICT['full_re_instruction_ft'].format_map({'relation_set': relation_set})
        inputs = PROMPT_DICT['input'].format_map(
            {'text': doc_sents, 'e_num': len(e_pairs), 'e_pairs': '\n'.join(e_pairs)}
        )
        if test:
            datas.append({
                'instruction': instruction,
                'input': inputs,
            })
        else:
            datas.append({
                'instruction': instruction,
                'input': inputs,
                'output': '\n'.join(outputs)
            })

    print(len(datas))
    print(len(idxs))

    with open(path_out, 'w') as fo:
        jsoned = json.dumps(datas)
        fo.write(jsoned)

    with open(index_path, 'w') as fid:
        jsoned = json.dumps(idxs)
        fid.write(jsoned)


if __name__ == "__main__":
    # generate for GPT
    # idx_path = '../docred/idx/gpt_sample_100_index.json'
    # generate_n_shot_prompt(rel_info_path, dev_set_path, input_train_path, '../docred_LLM/raw/gpt4_sample_100.json', idx_path)
    # generate for llama2
    generate_for_re_finetune(dev_set_path, rel_info_path, idx_path, 20, output_path, True)
