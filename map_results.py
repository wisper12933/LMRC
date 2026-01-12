import json
import re
import argparse
from tqdm import tqdm
from thefuzz import process, fuzz

try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


def load_data(truth_path, rel_info_path):
    """load ground truth data and relation info"""
    # load ground truth data
    raw_data = json.load(open(truth_path, 'r', encoding='utf-8'))
    title2data = {item['title']: item for item in raw_data}
    
    rel_info = json.load(open(rel_info_path, 'r', encoding='utf-8'))
    name2rel = {v: k for k, v in rel_info.items()}
    rel_names = list(rel_info.values())
    
    return title2data, name2rel, rel_names


class RelationMapper:
    def __init__(self, name2rel, rel_names, use_align=False):
        self.name2rel = name2rel
        self.rel_names = rel_names
        self.use_align = use_align
        self.wrong_rel_count = 0
        self.wrong_formats = 0
        self.model = None
        self.rel_embeddings = None

        if self.use_align:
            if not SBERT_AVAILABLE:
                raise ImportError("Please install sentence-transformers to use align mode.")
            print("Loading SBERT model for relation alignment...")
            self.model = SentenceTransformer('all-mpnet-base-v2')
            self.rel_embeddings = self.model.encode(self.rel_names)

    def find_closest_relation(self, pred_rel):
        """If the predicted relation is not in the known set, find the closest one using SBERT."""
        pred_emb = self.model.encode(pred_rel)
        scores = util.cos_sim(pred_emb, self.rel_embeddings)[0]
        
        best_idx = int(scores.argmax())
        best_rel_name = self.rel_names[best_idx]
        return self.name2rel[best_rel_name]

    def map_entity(self, entity_name, vertex_set):
        """
        map the predicted entity name to the index in vertex_set using fuzzy matching.
        vertex_set : [[{name: 'Obama', ...}, {name: 'Barack Obama'}], ...]
        """
        # extract candidate names
        candidate_names = [v[0]['name'] for v in vertex_set]
        
        # use the fuzz library to find the best match
        # extractOne return (match_string, score, index)
        best_match = process.extractOne(entity_name, candidate_names, scorer=fuzz.ratio)
        
        if best_match:
            matched_name = best_match[0]
            return candidate_names.index(matched_name)
        return -1

    def parse_and_map(self, prediction_item, ground_truth_item):
        """parse the prediction text and map to indices"""
        
        text_output = prediction_item.get('output', '') 
        if isinstance(prediction_item, str):
            text_output = prediction_item
            
        # map ( A | B | C ) 
        pattern = re.compile(r'\(([^|]+)\|([^|]+)\|([^|]+)\)')
        matches = pattern.findall(text_output)
        
        mapped_triples = []
        vertex_set = ground_truth_item['vertexSet']
        title = ground_truth_item['title']

        for h_text, r_text, t_text in matches:
            h_text, r_text, t_text = h_text.strip(), r_text.strip(), t_text.strip()
            
            # filter out invalid triples
            if not h_text or not r_text or not t_text:
                continue
            if r_text == 'None' or r_text == '-':
                continue

            # Relation Mapping
            r_id = self.name2rel.get(r_text)
            
            if r_id is None:
                self.wrong_rel_count += 1
                if self.use_align:
                    r_id = self.find_closest_relation(r_text)
                else:
                    # if not using alignment, skip this triple
                    continue
            
            # Entity Mapping
            h_idx = self.map_entity(h_text, vertex_set)
            t_idx = self.map_entity(t_text, vertex_set)
            
            if h_idx == -1 or t_idx == -1:
                continue
            
            # Exclude Self-loops
            if h_idx == t_idx:
                continue

            mapped_triples.append({
                'title': title,
                'h_idx': h_idx,
                't_idx': t_idx,
                'r': r_id,
                'evidence': []
            })
            
        origin_len = len(text_output.strip().split('\n'))
        mapped_len = len(mapped_triples)
        if mapped_len < origin_len:
            self.wrong_formats += (origin_len - mapped_len)
            
        return mapped_triples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rel_info', type=str, default='', help="Path to relation info (rel_info) JSON file")
    parser.add_argument('--truth_path', type=str, default='', help="Path to ground truth data (dev / test) JSON file")
    parser.add_argument('--result_path', type=str, default='', help="Path to LLM output results JSON file")
    parser.add_argument('--output_path', type=str, default='', help="Path to save the mapped results JSON file")
    parser.add_argument('--add_align', action='store_true', help="Enable semantic alignment for unknown relations")
    args = parser.parse_args()

    print("Loading data...")
    title2data, name2rel, rel_names = load_data(args.truth_path, args.rel_info)
    
    with open(args.result_path, 'r', encoding='utf-8') as f:
        llm_results = json.load(f)

    mapper = RelationMapper(name2rel, rel_names, use_align=args.add_align)
    
    final_results = []
    total_triples = 0
    missing_docs = 0

    print("Mapping results...")
    for item in tqdm(llm_results):

        title = item.get('title')
        
        if not title or title not in title2data:
            missing_docs += 1
            continue
            
        gt_item = title2data[title]
        triples = mapper.parse_and_map(item, gt_item)
        
        final_results.extend(triples)
        total_triples += len(triples)

    print("="*30)
    print(f"Total processed outputs: {len(llm_results)}")
    print(f"Docs missing in ground truth: {missing_docs}")
    print(f"Total extracted triples: {total_triples}")
    print(f"Relations not in strict set: {mapper.wrong_rel_count}")
    print(f"Wrongly formatted triples: {mapper.wrong_formats - mapper.wrong_rel_count}")
    print(f"Saving to {args.output_path}...")
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)

if __name__ == '__main__':
    main()