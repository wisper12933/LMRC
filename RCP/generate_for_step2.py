import json


# RCP 分类结果路径
step1_result_path = './results/your path to step1 RCP result'
results = json.loads(open(step1_result_path).read())

# outpath: 格式转换输出路径
# index_path: 存储每个query对应的document编号，方便evaluation
outpath = './results/your path'
index_path = './results/your path'

# 数据集文件路径
dev_path = '../datasets/docred/dev.json'
test_path = '../datasets/docred/test.json'
rel_path = '../datasets/docred/rel_info.json'

dev = json.loads(open(dev_path).read())
test = json.loads(open(test_path).read())
rel = json.loads(open(rel_path).read())
relation_set = [v for _, v in rel.items()]

PROMPT_DICT = {
    "input": (
        "### Text:\n{text}\n\n### {e_num} Entity pairs:\n{e_pairs}"
    ),
    "step2_instruction": (
        "This is a document-level relation extraction task. we will provide entity pairs that require relation "
        "extraction. Your task is to select relations for each entity pair from the given relation set based on the "
        "information in the text. There may be multiple relations between an entity pair "
        "\nThe format of input entity pairs is ‘(head entity| -| tail entity)’. Your output format is ‘(head entity| "
        "relation| tail entity)’. "
        "\n\n### Relation set:\n{relation_set}"
    )
}

def convert_format(dataset, result, output_path, split=0, index_path=''):
    '''
    dataset: docred/re-docred dev/test
    result: step1 RCP 的结果
    output_path: 格式化输出位置
    split: 每个query保留多少个实体对, 0代表不切分 
    index_path: 切分后的query对应的document编号
    '''
    title2vectex = {}
    title2pairs = {}
    title2text = {}
    titles = []
    
    for x in dataset:
        title = x['title']
        titles.append(title)
        vertexSet = x['vertexSet']
        sents = x['sents']
        title2vectex[title] = vertexSet
        
        concat_sents = [' '.join(words) for words in sents]
        title2text[title] = ' '.join(concat_sents)
    
    for title in titles:
        title2pairs[title] = []
    
    for item in result:
        title = item['title']
        h_idx = item['h_idx']
        t_idx = item['t_idx']
        vertexSet = title2vectex[title]
        h = vertexSet[h_idx][0]['name']
        t = vertexSet[t_idx][0]['name']
        title2pairs[title].append(f'({h}| -| {t})')
    
    data_list, index = [], []
    for idx, title in enumerate(title2pairs.keys()):
        text = title2text[title]
        e_pairs = title2pairs[title]
        if not e_pairs:
            continue
        if split:
            start_id = 0
            while start_id < len(e_pairs):
                end_id = start_id + split
                tmp_pairs = e_pairs[start_id:end_id]
                data_list.append({'text':text, 'e_num': len(tmp_pairs), 'e_pairs': '\n'.join(tmp_pairs)})
                start_id = end_id
                index.append(idx)
        else:
            index.append(idx)
            data_list.append({'text': text, 'e_num': len(e_pairs), 'e_pairs': '\n'.join(e_pairs)})
        
    prompt_input = PROMPT_DICT['input']
    inputs = [
        prompt_input.format_map(example) for example in data_list
    ]
    datas = [
        {'instruction': PROMPT_DICT['step2_instruction'].format_map({'relation_set': relation_set}),
         'input': inputs[idx]} for idx in range(len(inputs))
    ]
    
    with open(output_path, 'w') as fo:
        jsoned = json.dumps(datas)
        fo.write(jsoned)

    with open(index_path, 'w') as findex:
        jsoned = json.dumps(index)
        findex.write(jsoned)


def get_uncorrect(tmp, dataset, output_path):
    # Change to binary
    '''
        Adapted from the official evaluation code
    '''

    truth = dataset

    std = {}
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            h_idx = label['h']
            t_idx = label['t']
            if (title, h_idx, t_idx) not in std:
                std[(title, h_idx, t_idx)] = {'have_relation': True}

    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx']) != (y['title'], y['h_idx'], y['t_idx']):
            submission_answer.append(tmp[i])

    print(len(submission_answer))
    titleset2 = set([])
    uncorrect = []
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, h_idx, t_idx) not in std:
            h = vertexSet[h_idx][0]['name']
            t = vertexSet[t_idx][0]['name']
            uncorrect.append({'title': title, 'h_entity': h, 't_entity': t})

    print(len(uncorrect))
    with open(output_path, 'w') as fo:
        jsoned = json.dumps(uncorrect)
        fo.write(jsoned)
    

if __name__ == "__main__":
    convert_format(dev, results, outpath, split=18, index_path=index_path)
