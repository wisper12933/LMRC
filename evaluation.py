#!/usr/bin/env python
import sys
import os
import os.path
import json

input_dir = 're-docred' # docred / re-docred
log_dir = 'logs/your path'

idx_dir = 'sampled id dir'
submit_dir = 'mapped result dir'
truth_dir = os.path.join(input_dir, 'ref')

submission_answer_file = os.path.join(submit_dir, "your result.json")
truth_file = os.path.join(truth_dir, "test/dev.json")
# if sampled
selected = False


def findSmallestDifference(A, B, m, n):
    # Sort both arrays
    # using sort function
    A.sort()
    B.sort()

    a = 0
    b = 0

    # Initialize result as max value
    result = sys.maxsize

    # Scan Both Arrays upto
    # sizeof of the Arrays
    while a < m and b < n:

        if abs(A[a] - B[b]) < result:
            result = abs(A[a] - B[b])

        # Move Smaller Value
        if A[a] < B[b]:
            a += 1

        else:
            b += 1
    # return final sma result
    return result


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


if __name__ == '__main__':
    '''
       Adapted from the official evaluation code
    '''
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        fact_in_train_annotated = gen_train_facts("docred/ref/train_annotated.json", truth_dir)

        output_filename = os.path.join(log_dir, 're_3.1_8b_lmrc_test.txt')
        output_file = open(output_filename, 'w')

        truth = json.load(open(truth_file))

        # change truth_list based on idx
        if selected:
            select_idx_file = os.path.join(idx_dir, 'gpt_sample_100_index.json')
            select_idx = json.load(open(select_idx_file))
            truth = [truth[i] for i in select_idx]

        std = {}
        std_intra = {}
        std_inter = {}
        tot_evidences = 0
        titleset = set([])

        title2vectexSet = {}

        # statistics from truth
        for x in truth:
            title = x['title']
            titleset.add(title)

            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet

            for label in x['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                h_sent_set = [x['sent_id'] for x in vertexSet[h_idx]]
                t_sent_set = [x['sent_id'] for x in vertexSet[t_idx]]

                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])
                if findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set)) == 0:
                    std_intra[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if 1 <= findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set)):
                    std_inter[(title, r, h_idx, t_idx)] = set(label['evidence'])

        tot_relations = len(std)
        tot_relations_intra = len(std_intra)
        tot_relations_inter = len(std_inter)

        tmp = json.load(open(submission_answer_file))
        tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        submission_answer = [tmp[0]]
        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i - 1]
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                submission_answer.append(tmp[i])

        submission_answer_intra = []
        submission_answer_inter = []
        for i in range(len(submission_answer)):
            title = submission_answer[i]['title']
            if title not in title2vectexSet:
                print(title)
                continue
            vertexSet = title2vectexSet[submission_answer[i]['title']]

            h_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['h_idx']]]
            t_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['t_idx']]]
            if findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set)) == 0:
                submission_answer_intra.append(submission_answer[i])
            if 1 <= findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set)):
                submission_answer_inter.append(submission_answer[i])

        correct_re = 0
        correct_re_intra = 0
        correct_re_inter = 0
        correct_evidence = 0
        pred_evi = 0

        correct_in_train_annotated = 0
        titleset2 = set([])

        # count correct
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if 'evidence' in x:
                evi = set(x['evidence'])
            else:
                evi = set([])
            pred_evi += len(evi)

            if (title, r, h_idx, t_idx) in std:
                correct_re += 1
                stdevi = std[(title, r, h_idx, t_idx)]
                correct_evidence += len(stdevi & evi)
                in_train_annotated = in_train_distant = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True

                if in_train_annotated:
                    correct_in_train_annotated += 1

        for x in submission_answer_intra:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            if (title, r, h_idx, t_idx) in std_intra:
                correct_re_intra += 1

        for x in submission_answer_inter:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            if (title, r, h_idx, t_idx) in std_inter:
                correct_re_inter += 1

        # calculate F1
        # F1 & Ign F1
        re_p = 1.0 * correct_re / len(submission_answer)
        re_r = 1.0 * correct_re / tot_relations
        print(f'precision:{re_p}')
        print(f'recall:{re_r}')
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
        evi_r = 1.0 * correct_evidence / tot_evidences
        if evi_p + evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
                len(submission_answer) - correct_in_train_annotated)

        if re_p_ignore_train_annotated + re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (
                    re_p_ignore_train_annotated + re_r)

        # Intra F1 & Inter F1
        if len(submission_answer_intra) > 0:
            re_p_intra = 1.0 * correct_re_intra / len(submission_answer_intra)
        else:
            re_p_intra = 0

        re_r_intra = 1.0 * correct_re_intra / tot_relations_intra
        if re_p_intra + re_r_intra == 0:
            re_f1_intra = 0
        else:
            re_f1_intra = 2.0 * re_p_intra * re_r_intra / (re_p_intra + re_r_intra)

        if len(submission_answer_inter) > 0:
            re_p_inter = 1.0 * correct_re_inter / len(submission_answer_inter)
        else:
            re_p_inter = 0
        re_r_inter = 1.0 * correct_re_inter / tot_relations_inter
        if re_p_inter + re_r_inter == 0:
            re_f1_inter = 0
        else:
            re_f1_inter = 2.0 * re_p_inter * re_r_inter / (re_p_inter + re_r_inter)

        # print & save
        print('RE_F1:', re_f1)
        print('Evi_F1:', evi_f1)
        print('RE_ignore_annotated_F1:', re_f1_ignore_train_annotated)
        print('\nIntra precision:', re_p_intra)
        print('Intra recall:', re_r_intra)
        print('Intra F1:', re_f1_intra)
        print('\nInter precision:', re_p_inter)
        print('Inter recall:', re_r_inter)
        print('Inter F1:', re_f1_inter)

        output_file.write("RE_Precision: %f\n" % re_p)
        output_file.write("RE_Recall: %f\n" % re_r)
        output_file.write("RE_F1: %f\n" % re_f1)
        output_file.write("Evi_F1: %f\n" % evi_f1)

        output_file.write("RE_ignore_annotated_F1: %f\n" % re_f1_ignore_train_annotated)
        output_file.write("\nIntra precision: %f\n" % re_p_intra)
        output_file.write("Intra recall: %f\n" % re_r_intra)
        output_file.write("Intra F1: %f\n" % re_f1_intra)
        output_file.write("\nInter precision: %f\n" % re_p_inter)
        output_file.write("Inter recall: %f\n" % re_r_inter)
        output_file.write("Inter F1: %f\n" % re_f1_inter)

        output_file.close()
