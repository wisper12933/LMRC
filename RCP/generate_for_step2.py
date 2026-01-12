import json

# path to dataset files
dev_path = ''
test_path = ''
rel_path = ''


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
        "\nThe format of input entity pairs is '(head entity| -| tail entity)'. Your output format is '(head entity| "
        "relation| tail entity)'. "
        "\n\n### Relation set:\n{relation_set}"
    )
}

def convert_format(dataset, result, output_path, split=0):
    title2vectex = {}
    title2pairs = {}
    title2text = {}
    
    for x in dataset:
        title = x['title']
        vertexSet = x['vertexSet']
        sents = x['sents']
        title2vectex[title] = vertexSet
        
        concat_sents = [' '.join(words) for words in sents]
        title2text[title] = ' '.join(concat_sents)
    
    for item in result:
        title = item['title']
        h_idx = item['h_idx']
        t_idx = item['t_idx']
        if title not in title2pairs:
            title2pairs[title] = []
        vertexSet = title2vectex[title]
        h = vertexSet[h_idx][0]['name']
        t = vertexSet[t_idx][0]['name']
        title2pairs[title].append(f'({h}| -| {t})')
    
    data_list = []
    for title, e_pairs in title2pairs.items():
        text = title2text[title]
        # confirm split step length
        step = split if split > 0 else len(e_pairs)
        
        for i in range(0, len(e_pairs), step):
            batch_pairs = e_pairs[i : i + step]
            data_list.append({
                'title': title,
                'text': text, 
                'e_num': len(batch_pairs), 
                'e_pairs': '\n'.join(batch_pairs)
            })
        
    prompt_input = PROMPT_DICT['input']
    inputs = [
        prompt_input.format_map(example) for example in data_list
    ]
    datas = [
        {'title': data_list[idx]['title'],
         'instruction': PROMPT_DICT['step2_instruction'].format_map({'relation_set': relation_set}),
         'input': inputs[idx]} for idx in range(len(inputs))
    ]
    
    with open(output_path, 'w') as fo:
        jsoned = json.dumps(datas, indent=4)
        fo.write(jsoned)

if __name__ == "__main__":
    # prediction from step1
    step1_result_path = ''
    dev = json.loads(open(dev_path).read())
    test = json.loads(open(test_path).read())
    result = json.loads(open(step1_result_path).read())
    # output path for step2 input
    outpath = ''
    
    convert_format(dev, result, outpath, split=18)