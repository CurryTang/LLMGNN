from openail.utils import efficient_openai_text_api, set_endpoints, openai_text_api, openai_text_api_with_top_p, load_partial_openai_result, save_partial_openai_result, retrieve_dict, compute_ece, plot_calibration_curve, openai_text_api_with_backoff, num_tokens_from_string
from helper.data import get_dataset, inject_random_noise_y_level
from helper.args import get_command_line_args
from helper.active import train_lr, inference_lr
from helper.utils import load_yaml
from openail.config import configs
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import ast
from openail.utils import load_mapping
import ipdb
import os.path as osp
import editdistance
from collections import Counter
import random
import re
import string
import numpy as np
from models.nn import LinearRegression
from helper.utils import noise_transition_matrix
import seaborn as sns
import ipdb 
from helper.train_utils import calibration_plot
import pandas as pd
import sys

pal = sns.color_palette("crest")
sns.set_palette(pal)

# Set the figure size
plt.figure(figsize=(12, 12))  # 12 inches by 8 inches

# Set global text sizes
plt.rcParams['font.size'] = 20         # Default font size
plt.rcParams['axes.titlesize'] = 20    # Axes title size
plt.rcParams['axes.labelsize'] = 20    # X and Y axes label size
plt.rcParams['xtick.labelsize'] = 20   # X-tick label size
plt.rcParams['ytick.labelsize'] = 20   # Y-tick label size
plt.rcParams['legend.fontsize'] = 20   # Legend font size
plt.rcParams['font.family'] = 'sans-serif' # Figure title size

colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")


def most_common_number(numbers):
    counter = Counter(numbers)
    most_common = counter.most_common(1)  # Get the most common number and its count
    return most_common[0][0]  # Return the most common number


def get_closest_label(input_string, label_names):
    min_distance = float('inf')
    closest_label = None

    for label in label_names:
        distance = editdistance.eval(input_string, label)
        if distance < min_distance:
            min_distance = distance
            closest_label = label
    
    return closest_label

def keep_first_n_words(paragraph, n):
    words = paragraph.split()
    first_512_words = ' '.join(words[:n])
    return first_512_words


class Experiment:
    def __init__(self, data, api_key, data_path) -> None:
        self.raw_texts = data.raw_texts
        self.label_names = data.label_names
        self.category_names = data.category_names
        self.api_key = api_key
        self.data_path = data_path
        self.num_of_node = len(self.raw_texts)

    def load_cache(self, dataset, prompt_key):
        cache = load_partial_openai_result(self.data_path, dataset, prompt_key)
        return cache

    def save_cache(self, dataset, prompt_key, res):
        save_partial_openai_result(self.data_path, dataset, res, prompt_key, load_pre_existing=None, num_of_elements=self.num_of_node)

    def sync_api(self, prompts, temperature = None, top_p = None, n = 1, dataset = 'cora', key = 'zero_shot', rewrite = False):
        assert (temperature != None or top_p != None), "one of them must be set" 
        responses = []
        cache = self.load_cache(dataset, key)
        if cache != None:
            cache = cache.get(key, None)
        for i, prompt in tqdm(enumerate(prompts)):
            if prompt == "":
                responses.append("")
                continue
            if cache != None and cache[i] != "" and not rewrite:
                responses.append(cache[i])
            else:
                if temperature != None:
                    res = openai_text_api_with_backoff(prompt, self.api_key, temperature=temperature, n = n)
                else:
                    res = openai_text_api_with_top_p(prompt, self.api_key, top_p=top_p, n = n)
                if not self.check_grammar(res['choices'][0]['message']['content']) and key != 'consistency':
                    res = self.fix_grammar(res['choices'][0]['message']['content'])
                    time.sleep(3)
                responses.append(res)
                time.sleep(3)
        return responses
     

     
    
    def test_no_correction(self, prompts, label_names, temperature = None, top_p = None, n = 1, dataset = 'cora', key = 'zero_shot', rewrite = False, fix = False, seed = 0, sp = 2, ss = 2):
        responses = []
        errors = 0
        valid_number = len([x for x in prompts if x != ""])
        input_filename = "no_correct_prompt_async_input_{}_{}_temperature_{}_n_{}_input_seed_{}_{}.json".format(dataset, key, temperature, n, seed, valid_number)
        output_filename = "no_correct_prompt_async_input_{}_{}_temperature_{}_n_{}_output_seed_{}_{}.json".format(dataset, key, temperature, n, seed, valid_number)
        openai_result = efficient_openai_text_api(prompts, input_filename, output_filename, sp, ss, api_key=self.api_key, temperature=temperature, n = n, rewrite = rewrite)
        for i, res in tqdm(enumerate(openai_result)):
            if i not in select_ids:
                responses.append("")
                continue
            try:
                if key != 'consistency' and 'all' not in key:
                    check_res, error_type, error_message = self.check_grammar(res[0][0])
                    if check_res:
                        responses.append(res[0])
                        continue
                else:
                    check_res, error_type, error_message = self.check_grammar(res[0][0])
                    if check_res:
                        responses.append(res[0])
                        continue
            except Exception:
                check_res = False
                error_type = 'grammar error'
                error_message = None
                responses.append('')
                errors += 1
        print("Error number: {}".format(errors))
        return responses, None

    def async_api(self, prompts, label_names, temperature = None, top_p = None, n = 1, dataset = 'cora', key = 'zero_shot', rewrite = False, fix = False, seed = 0, sp = 2, ss = 2):
        responses = []
        select_ids = [i for i in range(len(prompts)) if prompts[i] != ""]
        fixed_grammar = 0
        g_error_idx = []
        cache = self.load_cache(dataset, key)
        ## cache for openai result
        if cache != None:
            cache = cache.get(key, None)
        ## cache for extracted annotation result, may present inconsistency with openai result
        result_cache = self.load_saved(dataset, key)
        result_cache_pred = result_cache['pred']
        # import ipdb; ipdb.set_trace()
        valid_number = len([x for x in prompts if x != ""])
        input_filename = "prompt_async_input_{}_{}_temperature_{}_n_{}_input_seed_{}_{}.json".format(dataset, key, temperature, n, seed, valid_number)
        output_filename = "prompt_async_input_{}_{}_temperature_{}_n_{}_output_seed_{}_{}.json".format(dataset, key, temperature, n, seed, valid_number)
        # import ipdb; ipdb.set_trace()
        cache_hit = 0
        result_cache_hit = 0
        for i in range(len(prompts)):
            if cache != None and cache[i] != "" and isinstance(cache[i], list) and prompts[i] != "" and not rewrite:
                check_res, error_type, error_message = self.check_grammar(cache[i][0])
                if check_res:
                    prompts[i] = ""
                    cache_hit += 1
            else:
                try:
                    if result_cache_pred[i] != -1 and prompts[i] != "":
                        prompts[i] = ""
                        cache_hit += 1
                        result_cache_hit += 1
                except Exception:
                    pass
        # import ipdb; ipdb.set_trace()
        print("cache hit number: {}".format(cache_hit))
        if result_cache_hit == len(select_ids):
            print("All cache hit, no need to run the annotation")
            sys.exit()
        openai_result = efficient_openai_text_api(prompts, input_filename, output_filename, sp, ss, api_key=self.api_key, temperature=temperature, n = n, rewrite = rewrite)
        # import ipdb; ipdb.set_trace()
        # if len(select_ids) - cache_hit >= 20:
        #     time.sleep(60)
        for i, res in tqdm(enumerate(openai_result)):
            if i not in select_ids:
                responses.append("")
                continue
            # if key != 'consistency':
            if (cache != None and cache[i] != "" and res[0] == "") or (result_cache_pred[i] != -1 and res[0] == ""):
                responses.append(cache[i])
                continue
            # else:
            #     if cache != None and cache[i] != "" and res[0] == "":
            #         responses.append(cache[i])
            #         continue
            else:
                try:
                    if key != 'consistency' and 'all' not in key:
                        check_res, error_type, error_message = self.check_grammar(res[0][0])
                        if check_res:
                            responses.append(res[0])
                            continue
                    else:
                        check_res, error_type, error_message = self.check_grammar(res[0][0])
                        if check_res:
                            responses.append(res[0])
                            continue
                except Exception:
                    check_res = False
                    error_type = 'grammar error'
                    error_message = None
                if not check_res:
                    # import ipdb; ipdb.set_trace()
                    if key != 'consistency' and 'all' not in key:
                        new_results = self.correct_grammar(prompts[i], res[0], error_type, temperature, n, error_message)
                        fixed_grammar += 1
                    else:
                        new_results = self.correct_grammar(prompts[i], res[0][0], error_type, temperature, n, error_message)
                        fixed_grammar += 1
                responses.append(new_results)
                g_error_idx.append(i)
                time.sleep(3)
        print("fixed grammar number: {}".format(fixed_grammar))
        return responses, g_error_idx

    
    def original_prompt(self, question):
        return question
        
    def vanilla_verbalized_confidence(self, question):
        return question
    
    def cot_confidence(self, question):
        return " Question: {}. Please analyze the question step by step, give \
your step-by-step analysis, answer, and a confidence level ranging from 0 to 100 in the format of a dictionary. \
For example, [{{\"analysis\": <your step-by-step analysis>, \"answer\": <answer to the question>, \"confidence\": <confidence of the answer>}}] \n \
Analysis, answer, and confidence: ".format(question)
    
    def multi_step_confidence(self, question):
        return " Question: {}. Read the question, give your answer by analyzing step by step, and \
assign a confidence level to each step and the final answer. The output \
format is as follows: \
Step 1: {{\"Reasoning\": <Your reasoning here>, \"Confidence\": <Your confidence \
here>}} \
Step 2: ... \
Step 3: ... \
... \
Step N: ... \
Final Answer and Overall Confidence (0-100): [{{\"Answer\": <Your answer \
here>, \"Confidence\": <Your confidence here>}}] \n \
".format(question)
    
    def topk_confidence(self, question, k = 3, name = 'arxiv'):
        if name == 'arxiv':
            return "Paper: {}. Which arxiv CS-subcategory does this paper belong to? Output your answer in the form of arXiv CS sub-categories like \"cs.XX\" together with a confidence ranging from 0 to 100, in the form of a list of python dicts like [{{\"answer\":<answer_here>, \"confidence\": <confidence_here>}}, ...]\n".format(question)
        else:
            return "Question: {}. Provide your {} best guesses and a confidence number that each is correct \
(0 to 100) for the following question from most probable to least. The sum of all confidence should be 100. For example,  \
[{{\"answer\": <your_first_answer>, \"confidence\": <confidence_for_first_answer>}}, ...]        \
".format(question, k)


    def retrieve_multiple_answers(self, answer, data, num):
        output = []
        invalid = 0
        for result in answer:
            # import ipdb; ipdb.set_trace()
            if result == "":
                res = [("", 0)]
                output.append(res)
                continue
            res = []
            for r in result:
            # import ipdb; ipdb.set_trace()
                line = r.lower()
                this_line = []
                try:
                    ## if no error, retrieve all dicts in a list
                    this_dict = retrieve_dict(line)
                    for dic in this_dict:
                        # dic = this_dict[0]
                        answer = dic['answer']
                        confidence = dic['confidence']
                        this_line.append((answer, confidence))
                        # res.append((answer, confidence))
                    res.append(this_line)
                except:
                    ## if error, split the result based on }, 
                    parts = line.split("},")
                    for p in parts:
                        try: 
                            ans = get_closest_label(p, self.label_names)
                            confidence = max(int(''.join(filter(str.isdigit, p))), 100)
                        except Exception:
                            confidence = 0
                        this_line.append((ans, confidence))
                        invalid += 1
                    # res.append(("", 0))
                    # continue
                    res.append(this_line)
            output.append(res)
        print("invalid number: {}".format(invalid))
        return output
    
    def retrieve_answer(self, answer, data):
        output = []
        invalid = 0
        for result in answer:
            # import ipdb; ipdb.set_trace()
            if result == "":
                res = [("", 0)]
                output.append(res)
                continue
            line = result[0].lower()
            try:
                this_dict = retrieve_dict(line)
                res = []
                for dic in this_dict:
                    answer = dic['answer']
                    confidence = dic['confidence']
                    if isinstance(confidence, str) and '%' in confidence:
                        confidence = int(confidence.replace('%', ''))
                    res.append((answer, confidence))
                output.append(res)
            except:
                answer = get_closest_label(line, self.label_names)
                confidence = min(int(''.join(filter(str.isdigit, line))), 100)
                # res = [("", 0)]
                res = [(answer, confidence)]
                output.append(res)
                invalid += 1
                continue
        print("invalid number: {}".format(invalid))
        return output


    def save_result(self, num_of_nodes, select_ids, pred, conf, data_obj, dataset_name = 'cora', seed = 0, strategy = 'random', method = 'zero_shot'):
        y_pred = torch.tensor([-1 for _ in range(num_of_nodes)])
        y_conf = torch.tensor([0. for _ in range(num_of_nodes)])
        y_pred[select_ids] = pred
        y_conf[select_ids] = conf
        res = {
            'pred': y_pred,
            'conf': y_conf,
            'test_mask': data_obj.test_masks[seed]
        }
        torch.save(res, osp.join(self.data_path, 'active', '{}^result^{}^{}^{}.pt'.format(dataset_name, strategy, method, seed)))
        # torch.save(data_obj, osp.join(self.data_path, '{}_data_{}_{}.pt'.format(dataset_name, num_seeds)))

    def load_saved(self, dataset_name, prompt_key):
        cache_path = osp.join(self.data_path, 'active', '{}^cache^{}.pt'.format(dataset_name, prompt_key))
        if osp.exists(cache_path):
            cache = torch.load(cache_path)
            return cache
    
    def save_saved(self, dataset_name, prompt_key, new_y_pred, new_y_conf, select_ids, total_num):
        cache_path = osp.join(self.data_path, 'active', '{}^cache^{}.pt'.format(dataset_name, prompt_key))
        # torch.save(self.cache, cache_path)
        num_nodes = total_num
        if osp.exists(cache_path):
            cache = torch.load(cache_path)
            y_pred, y_conf = cache['pred'], cache['conf']
            if len(y_pred) != num_nodes:
                y_pred = torch.tensor([-1 for _ in range(num_nodes)])
                y_conf = torch.tensor([0. for _ in range(num_nodes)])
        else:
            y_pred = torch.tensor([-1 for _ in range(num_nodes)])
            y_conf = torch.tensor([0. for _ in range(num_nodes)])
        y_pred[select_ids] = new_y_pred
        y_conf[select_ids] = new_y_conf  
        # for i, p in enumerate(new_y_pred):
        #     if p != -1:
        #         y_pred[i] = p
        #         y_conf[i] = new_y_conf[i]
        res = {
            'pred': y_pred,
            'conf': y_conf,
        }
        torch.save(res, cache_path)

    def fix_grammar(self, old):
        if '[' not in old or ']' not in old: return ""
        start = old.find('[')
        end = old.find(']', start) + 1  # +1 to include the closing bracket
        old = old[start:end]
        prompt = "Extract a valid python object from the following text, just output the processed object, do not output anything else. \
Old one: {} \n New one here:".format(old)
        new_res = openai_text_api(prompt, self.api_key, temperature=0, n = 1)
        return new_res
    
    def correct_grammar(self, previous_prompt, old_output, correct_type = 'grammar error', temperature = 0, n = 1, error_message = None):
        """
            The wrong type can be grammar_error, format_error, etc.
        """
        prompt = "previous prompt: {} \n".format(previous_prompt)
        prompt += "Your previous output doesn't follow the format, please correct it\n"
        prompt += "old output: {} \n".format(old_output)
        if correct_type == 'grammar error':
            prompt += "Your output should be a valid python object as a list of dictionaries"
        else:
            prompt += "Your previous answer {} is not a valid class.\n".format(error_message)
            prompt += "Your should only output categories from the following list: \n"
            prompt += "[" + ", ".join(self.label_names) + "]" + "\n"
        prompt += "New output here: "
        # import ipdb; ipdb.set_trace()
        new_res = openai_text_api_with_backoff(prompt, self.api_key, temperature=1, n = n)
        if len(new_res['choices']) == 1:
            return [new_res['choices'][0]['message']['content']]
        else:
            return [x['message']['content'] for x in new_res['choices']]


    def check_grammar(self, old, format = '[]'):
        clean_t = old
        list_str = ""
        if format == '[]':
            start = clean_t.find('[')
            end = clean_t.find(']', start) + 1  # +1 to include the closing bracket
            list_str = clean_t[start:end]
        else:
            start = clean_t.find('{')
            end = clean_t.find('}', start) + 1  # +1 to include the closing bracket
            list_str = clean_t[start:end]
        list_str = list_str.lower()
        try:
            result = ast.literal_eval(list_str)
        except Exception:
            return False, 'grammar error', None
        try:
            first_answer = result[0]
            if not isinstance(first_answer, dict):
                return False, 'grammar error', None
            else:
                answer = first_answer['answer']
                confidence = first_answer['confidence']
                # import ipdb; ipdb.set_trace()
                # idx = self.label_names.index(answer)
                if answer in self.label_names:
                    return True, 'success', None
                else:
                    return False, 'format error', answer
        except Exception:
            return False, 'format error', None
        
        


    def eval_result(self, res, label_names, select_ids, gt, dataset_name = 'cora', method='zero_shot', strategy = 'random', data_obj = None, seed = 0, g_error_idx = None):
        ## accuracy
        ## ECE 
        ## plot
        ## save the result
        pred = []
        gt_y = gt[select_ids]
        conf = []
        cannot_fix = 0
        # g_error_mask = torch.tensor([0 for _ in range(len(gt))])
        # g_error_mask[g_error_idx] = 1
        debug = []
        for i, r in enumerate(res):
            if i not in select_ids: continue
            # import ipdb; ipdb.set_trace()
            # debug.append(i)
            voting = []
            if 'consistency' in method or 'all' in method:
                k = len(r)
                this_pred = []
                this_conf = []
                selected = False
                for selection in r:
                    # import ipdb; ipdb.set_trace()
                    if selection[0][0] not in label_names:
                        continue    
                    p = label_names.index(selection[0][0])
                    c = selection[0][1]
                    this_pred.append(p)
                    this_conf.append(c)
                    selected = True
                if not selected:
                    cannot_fix += 1
                    p = get_closest_label(selection[0][0], label_names)
                    c = selection[0][1] / 2
                    p = label_names.index(p)
                    pred.append(p)
                    conf.append(c)
                    continue
                p = most_common_number(this_pred)
                first_appear = this_pred.index(p)
                base_c = 0
                orig_c = this_conf[first_appear]
                # import ipdb; ipdb.set_trace()
                for pp, cc in zip(this_pred, this_conf):
                    if pp == p:
                        this_c = (orig_c + cc) / 2
                        base_c += this_c
                    else:
                        base_c += (100 - cc)
                base_c /= k
                pred.append(p)
                conf.append(base_c)
                # import ipdb; ipdb.set_trace()
            else:
                ans = r[0]
                p = ans[0]
                c = int(ans[1])
                if p not in label_names:
                    p = get_closest_label(p, label_names)
                    c /= 2
                    cannot_fix += 1
                p = label_names.index(p)
                pred.append(int(p))
                conf.append(int(c))
        pred = torch.tensor(pred)
        conf = torch.tensor(conf) / 100.0
        ## 1. all acc
        all_acc = (pred == gt_y).float().mean()
        ## 2. filtered acc (filter -1 term)
        filter_acc = (pred[conf > 0] == gt_y[conf > 0]).float().mean()
        ## 3. ece 
        filter_conf = conf[conf > 0]
        filter_pred = pred[conf > 0]
        filter_label = gt_y[conf > 0]
        # error_mask = (conf > 0) & (g_error_mask == 1)
        # not_error_mask = (conf > 0) & (g_error_mask == 0)
        # error_mask_acc = (pred[error_mask] == gt_y[error_mask]).float().mean()
        # non_error_mask_acc = (pred[not_error_mask] == gt_y[not_error_mask]).float().mean()
        self.save_result(len(gt), select_ids, pred, conf, data_obj, dataset_name, seed, strategy, method)
        total_num = len(gt)
        self.save_saved(dataset_name, method, pred, conf, select_ids, total_num)
        ece = compute_ece(filter_conf, filter_pred, filter_label, n_bins = 10)
        print("cannot fix number: {}".format(cannot_fix))
        # print("non error mask acc: {}, error mask acc: {}".format(non_error_mask_acc, error_mask_acc))
        print("all acc: {:.2f}, filter acc: {:.2f}, ece: {:.2f}, number of labels: {}".format(all_acc, filter_acc, ece, len(filter_label)))
        # plot_calibration_curve(filter_conf, filter_pred, filter_label, dataset_name, method, n_bins = 10)
        # calibration_plot(filter_conf, filter_pred, filter_label, dataset_name, method, n_bins = 10)
        # output_name = 'calibration_{}_method_{}'.format(dataset_name, method)
        # calibration_plot(filter_conf, filter_pred, filter_label, output_name)
        return all_acc, filter_acc, ece



def arxiv_prompt(texts, k = 1):
    prompts = []
    for text in texts:
        prompt = "Paper: \n"
        prompt += (text + "\n")
        prompt += "Question: Which arXiv CS sub-category does this paper belong to? Give {} likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely, in the form “cs.XX”, and provide your confidence score. The sum of all confidence score should be 100%. "
        prompt = prompt.format(k)
        prompts.append(prompt)
    return prompts



def zero_shot_prompt(texts, label_names, need_tasks = True, object_cat = "Paper", question = "Which arxiv cs subcategories does this paper belong to?", \
                    answer_format = "Give 3 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 100, in the form of a list python dicts like [{\"answer:\":<answer_here>, \"confidence\": <confidence_here>}]"):
    prompts = []
    for text in texts:
        prompt = "{}: \n".format(object_cat)
        prompt += (text + "\n")
        prompt += "Task: \n"
        if not 'arxiv' in question:
            prompt += "There are following categories: \n"
            prompt += "[" + ", ".join(label_names) + "]" + "\n"
        prompt += question + "\n"
        if need_tasks:
            prompt +=  answer_format
        prompts.append(prompt)
    return prompts


def few_shot_prompts(texts, label_names, example, example_label, object_cat = "Paper", question = "Which arxiv cs subcategories does this paper belong to?", answer_format = "Give 3 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 100, in the form of a list python dicts like [{\"answer:\":<answer_here>, \"confidence\": <confidence_here>}]"):
    prompts = []
    for text in texts:
        prompt = "I will first give you an example and you should complete task following the example.\n"
        prompt += zero_shot_prompt([example], label_names, need_tasks = True, object_cat = object_cat, question = question, answer_format = answer_format)[0]
        prompt += "\nOutput:\n[{{\"answer\":\"{}\", \"confidence\":{}}}]\n".format(example_label, 100)
        prompt += zero_shot_prompt([text], label_names, need_tasks = True, object_cat = object_cat, question = question, answer_format = answer_format)[0]
        prompt += "\nOutput:\n"
        prompts.append(prompt)
    return prompts

def few_shot_topk_prompts(texts, label_names, example, example_dict, exp, object_cat = "Paper", question = "Which arxiv cs subcategories does this paper belong to?", answer_format = "Give 3 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 100, in the form of a list python dicts like [{\"answer:\":<answer_here>, \"confidence\": <confidence_here>}]", name = 'arxiv'):
    prompts = []
    for text in texts:
        prompt = "I will first give you an example and you should complete task following the example.\n"
        in_context = example + "\n"
        in_context += "Task: \n"
        if not 'arxiv' in question:
            in_context += "There are following categories: \n"
            in_context += "[" + ", ".join(label_names) + "]" + "\n"
            in_context += question + "\n"
        prompt += exp.topk_confidence(in_context, 3, name)
        # prompt += question + "\n"
        prompt += "\nOutput:\n"
        prompt += example_dict + "\n"
        in_context = text + "\n"
        in_context += "Task: \n"
        if not 'arxiv' in question:
            in_context += "There are following categories: \n"
            in_context += "[" + ", ".join(label_names) + "]" + "\n"
        in_context += question + "\n"
        prompt += exp.topk_confidence(in_context, 3, name)
        # prompt += question + "\n"
        prompt += "\nOutput:\n"
        prompts.append(prompt)
    return prompts 


def calculate_cost_from_a_list_of_texts(texts, select_ids = None):
    cost = 0
    for i, text in enumerate(texts):
        if select_ids != None and i not in select_ids: continue
        if len(text) > 0:
            cost += num_tokens_from_string(text, model = 'gpt-3.5-turbo-0301')
    return cost



def pack_prompt(prompts, select_idx, number):
    new_prompts = ["" for _ in range(number)]
    for i, prompt in enumerate(prompts):
        if i in select_idx:
            new_prompts[i] = prompt
    return new_prompts


def consistency(topk_prompt, dataname, exp, temperature = 1.2, data = None, strategy = 'random', seed = 0, key_name = 'consistency'):
    ## 
    n = 3
    topk_response, error_idx = exp.async_api(topk_prompt, data.label_names, temperature, None, n = n, dataset = dataname, key = key_name, rewrite = False, fix = True, seed = seed, sp = 60, ss = 1.5)
    exp.save_cache(dataname, key_name, topk_response)
    res = exp.retrieve_multiple_answers(topk_response, data, n)
    eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method=key_name, strategy = strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
    # response_str = [x[0] + x[1] + x[2] for x in topk_response if x != '']
    response_str = []
    for x in topk_response:
        try:
            if x != '':
                response_str.append(x[0] + x[1] + x[2])
        except Exception:
            response_str.append("")
    return eval_out, response_str

def consistency_no_topk(prompt, dataname, exp, temperature = 1.2, data = None, strategy = 'random', seed = 0, key_name = 'consistency_no_topk'):
    ## 
    n = 3
    response, error_idx = exp.async_api(prompt, data.label_names, temperature, None, n = n, dataset = dataname, key = key_name, rewrite = False, fix = True, seed = seed, sp = 60, ss = 1.5)
    exp.save_cache(dataname, key_name, response)
    res = exp.retrieve_multiple_answers(response, data, n)
    eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method=key_name, strategy = strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
    # response_str = [x[0] + x[1] + x[2] for x in response if x != '']
    response_str = []
    for x in response:
        try:
            if x != '':
                response_str.append(x[0] + x[1] + x[2])
        except Exception:
            response_str.append("")
    return eval_out, response_str



def clean_header(text):
    text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)

    return text

def clean_text(text):  
    re_url = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
    re_email = re.compile('(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')      
    text = text.lower()
    text = text.strip()
    text = re.sub(re_url, '', text)
    text = re.sub(re_email, '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'(\d+)', ' ', text)
    text = re.sub(r'(\s+)', ' ', text)
    
    return text



def few_shot_all_query(orig_question, data, configs, dataname, select_ids, rewrite = False):
    object_cat = configs[dataname]['zero-shot']['object-cat']
    question = configs[dataname]['zero-shot']['question']
    answer_format = configs[dataname]['zero-shot']['answer-format']
    few_shot_topk = configs[dataname]['few-shot-2']['examples']
    fst_example = few_shot_topk[0][0]
    fst_result = few_shot_topk[0][1]
    select_ids, _ = select_ids.sort()
    few_shot_all_prompts = few_shot_topk_prompts(orig_question, data.label_names, fst_example, fst_result, exp, object_cat, question, answer_format)
    few_shot_all_prompts = pack_prompt(few_shot_all_prompts, select_ids, data.x.shape[0])
    few_shot_all_response, error_idx = exp.async_api(few_shot_all_prompts, data.label_names, 0.7, None, n = 3, dataset = dataname, key = 'few_shot_all', rewrite = rewrite, fix = True, seed = seed, sp = 60, ss = 3)
    exp.save_cache(dataname, 'few_shot_all', few_shot_all_response)
    res = exp.retrieve_multiple_answers(few_shot_all_response, data, 3)
    eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='few_shot_all', strategy = args.strategy, data_obj = data, seed = seed, g_error_idx = error_idx)


def zero_shot_query(orig_question, data, configs, dataname, select_ids, rewrite = False):
    object_cat = configs[dataname]['zero-shot']['object-cat']
    question = configs[dataname]['zero-shot']['question']
    answer_format = configs[dataname]['zero-shot']['answer-format']
    few_shot_topk = configs[dataname]['few-shot-2']['examples']
    fst_example = few_shot_topk[0][0]
    fst_result = few_shot_topk[0][1]
    select_ids, _ = select_ids.sort()
    questions = zero_shot_prompt(orig_question, data.label_names, True, object_cat, question, answer_format)
    vanilla_prompt = [exp.vanilla_verbalized_confidence(q) for q in questions]
    vanilla_prompt = pack_prompt(vanilla_prompt, select_ids, data.x.shape[0])
    vanilla_response, error_idx = exp.async_api(vanilla_prompt, data.label_names, temperature = 0, top_p = None, n = 1, dataset = dataname, key = 'zero_shot', rewrite = rewrite, fix = True, seed = seed, sp = 60, ss = 0)
    exp.save_cache(dataname, 'zero_shot', vanilla_response)
    res = exp.retrieve_answer(vanilla_response, data)
    eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='zero_shot', strategy = args.strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
    


def index_to_mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

# print("123")



def count_labels(label_tensor):
    unique_labels, counts = torch.unique(label_tensor, return_counts=True)
    
    for label, count in zip(unique_labels, counts):
        print(f"Label {label.item()}: {count.item()} times")


def count_and_plot(tensor, tensor2, xtick_labels=None, output_name = "", title="Cora original sampled label distribution"):
    plt.figure(figsize=(12, 6))
    unique_labels, counts = torch.unique(tensor, return_counts=True)
    unique_labels2, counts2 = torch.unique(tensor2, return_counts=True)

    hue = ['Ground truth' for _ in range(len(unique_labels2))] + ['Annotations' for _ in range(len(unique_labels))]

    x = unique_labels.tolist() + unique_labels2.tolist()

    x = [str(a) for a in x]

    counts = counts.tolist() + counts2.tolist()

    df = pd.DataFrame({'Labels': x, '#Appearance': counts, 'Type': hue})

    cmap_pal = sns.color_palette("deep")

    sns.set_palette(cmap_pal)

    ax = sns.barplot(data=df, x="Labels", y="#Appearance", hue="Type")

    plt.legend(loc='upper right', fontsize=20)

    plt.ylim(0, 375)

    # plt.legend(title=None, loc='upper left', labels=['Ground truth', 'Annotations'])



    # # Convert to Python list for plotting
    # labels = unique_labels.tolist()
    # counts = counts.tolist()

    # labels2 = unique_labels2.tolist()
    # counts2 = counts2.tolist()

    # max_data = np.maximum(counts, counts2)
    
    # # Create the bar plot
    # plt.bar(labels, counts, color='lime', label='Annotations',)
    # plt.bar(labels2, counts2, bottom = counts, color='mediumblue', label='Ground truth')

    # plt.xlabel('Labels')
    # plt.ylabel('Number of Appearances')
    plt.xticks(fontsize=35, fontweight='bold')

    # Adjust y-axis label properties
    plt.yticks(fontsize=35, fontweight='bold')

    plt.xlabel('Labels', fontsize=35, fontweight='bold')

    plt.ylabel('#Appearances', fontsize=35, fontweight='bold')

    plt.tick_params(axis='x', which='both', length=0)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plt.tick_params(axis='x', which='both', length=0)

    legend = ax.legend()
    legend.set_title('')
    plt.setp(legend.get_title(), visible=False)


    plt.tight_layout()
    plt.savefig(output_name)
    plt.clf()





if __name__ == '__main__':
    # ipdb.set_trace()
    print("LLM GNN")
    args = get_command_line_args()    
    params_dict = load_yaml(args.yaml_path)
    # endpoint = params_dict.get('OPENAI_BASE', None)
    # if params_dict.get('OPENAI_KEY'):
    key = params_dict['OPENAI_KEY']
    data_path = params_dict['DATA_PATH']
    # set_endpoints(key, endpoint)
    seeds = [i for i in range(args.main_seed_num)]
    # print(args.dataset)
    if args.dataset == '':
        need_datasets = ['citeseer', 'cora', 'wikics']
    else:
        need_datasets = [args.dataset]
    # need_datasets = ['cora']
    # budget = [140, 120, 60, 800, 200, 200]
    full_mapping = load_mapping()
    # total_budget = 150
    # exps = ['consistency', 'consistency_no_topk', 'topk', 'few_shot_all', 'zero_shot', 'few_shot']
    if args.filter_strategy == 'none':
        # exps = ['draw']
        exps = ['draw']
        # exps = ['zero_shot', 'few_shot']
    else:
        exps = [args.filter_strategy]
    # exps = ['few_shot_all']
    
    for dataname in need_datasets:
        # for seed in seeds:
        reliability_list = []
        # data = get_dataset(seeds, dataname, args.split, args.data_format, params_dict['DATA_PATH'], None, 0, args.no_val, args.budget, 
        #                     args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, 1 - args.pl_noise, reliability_list, args.total_budget, 'none', False, args.alpha, args.beta, args.gamma, args.ratio)
        params_dict = load_yaml(args.yaml_path)
        data_path = params_dict['DATA_PATH']
        data = get_dataset(seeds, dataname, args.split, args.data_format, data_path, None, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, False, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio)
        # ipdb.set_trace()
        data.label_names = [full_mapping[dataname][x] for x in data.label_names]
        data.label_names = [x.lower() for x in data.label_names]
        exp = Experiment(data, key, data_path)
        object_cat = configs[dataname]['zero-shot']['object-cat']
        question = configs[dataname]['zero-shot']['question']
        answer_format = configs[dataname]['zero-shot']['answer-format']
        examples = configs[dataname]['few-shot']['examples']
        example_text = examples[0][0]
        example_labels = examples[0][1]
        few_shot_topk = configs[dataname]['few-shot-2']['examples']
        fst_example = few_shot_topk[0][0]
        fst_result = few_shot_topk[0][1]
        idxs = torch.arange(data.x.shape[0])
        performance = {}
        performance[dataname] = {}
        performance[dataname]['zero_shot'] = []
        performance[dataname]['few_shot'] = []
        performance[dataname]['cot'] = []
        performance[dataname]['topk'] = []
        performance[dataname]['consistency'] = []
        performance[dataname]['consistency_no_topk'] = []
        performance[dataname]['few_shot_all'] = []

        cost = {}
        cost[dataname] = {}
        cost[dataname]['zero_shot'] = []
        cost[dataname]['few_shot'] = []
        cost[dataname]['cot'] = []
        cost[dataname]['topk'] = []
        cost[dataname]['consistency'] = []
        cost[dataname]['consistency_no_topk'] = []
        cost[dataname]['few_shot_all'] = []

        

        for seed in seeds:
            # import ipdb; ipdb.set_trace()
            select_samples = data.train_masks[seed]
            select_ids = idxs[select_samples]
            # import ipdb; ipdb.set_trace()
            orig_question = data.raw_texts
            if dataname == '20newsgroup':
                orig_question = [clean_header(x) for x in tqdm(orig_question)]
                orig_question = [clean_text(x) for x in tqdm(orig_question)]
            if dataname == 'wikics' or dataname == 'products' or dataname == '20newsgroup':
                orig_question = [keep_first_n_words(x, 256) for x in tqdm(orig_question)]
            questions = zero_shot_prompt(orig_question, data.label_names, True, object_cat, question, answer_format)
            no_task_question = zero_shot_prompt(orig_question, data.label_names, need_tasks=False, object_cat = object_cat, question=question, answer_format=answer_format)
            
            if 'draw' in exps:
                cbar = False
            # do plot here
                saved_results = exp.load_saved(dataname, 'few_shot_all')
                pred = saved_results['pred']
                conf = saved_results['conf']


                # continue
                # noise_transition_matrix()

                total_idxs = torch.arange(len(pred))
                random_idx = total_idxs[(pred != -1)][:1000]

                # # wrong_idx = total_idxs[(pred != -1)]
                # random_mask = index_to_mask(random_idx, len(pred))
                # wrong_idx = total_idxs[(pred != -1) & random_mask]

                # few_shot_all_query(orig_question, data, configs, dataname, random_idx)

                pred = pred[random_idx]
                gt = data.y[random_idx]


                axis_labels = [f"{i}" for i in range(len(data.label_names))]

                noise_transition_matrix(pred, gt, "{}_noise_transition_matrix_few_shot.pdf".format(dataname), x_axis_labels = axis_labels, y_axis_labels = axis_labels, cbar = cbar)

                # print("Prediction: \n")
                # count_labels(pred)
                xticks = [f"c{i}" for i in range(len(data.label_names))]
                count_and_plot(pred, gt, xtick_labels = xticks, output_name = "{}_few_shot_pred_label_distribution.pdf".format(dataname), title = "{} sampled llm label distribution".format(dataname))
                # count_and_plot(gt, xtick_labels = xticks, output_name = "{}_few_shot_gt_label_distribution.pdf".format(dataname), title = "{} sampled original label distribution".format(dataname))
                # acc = (pred == gt).float().mean()

                # zero_shot_query(orig_question, data, configs, dataname, random_idx)

                zero_shot_results = exp.load_saved(dataname, "consistency")
                pred = zero_shot_results['pred']
                conf = zero_shot_results['conf']

                total_idxs = torch.arange(len(pred))
                random_idx = total_idxs[(pred != -1)][:1000]

                pred = pred[random_idx]
                gt = data.y[random_idx]

                acc = (pred == gt).float().mean()

                noise_transition_matrix(pred, gt, "{}_noise_transition_matrix_zero_shot.pdf".format(dataname), x_axis_labels = axis_labels, y_axis_labels = axis_labels, cbar = cbar)

                # print("Prediction: \n")
                # count_labels(pred)
                # print("Ground truth: \n")
                # count_labels(gt)
                count_and_plot(pred, gt, xtick_labels = xticks, output_name = "{}_zero_shot_pred_label_distribution.pdf".format(dataname), title = "{} sampled llm label distribution".format(dataname))
                # count_and_plot(gt, xtick_labels = xticks, output_name = "{}_zero_shot_gt_label_distribution.pdf".format(dataname), title = "{} sampled original label distribution".format(dataname))

                new_y = inject_random_noise_y_level(gt, 1 - acc)

                noise_transition_matrix(new_y, gt, "{}_noise_transition_matrix_synthetic.pdf".format(dataname), x_axis_labels = axis_labels, y_axis_labels = axis_labels, cbar = cbar)
                
                # continue

            # conf = saved_results['conf']

            # continue
            
            # ## test linear regression


            # total_idxs = torch.arange(len(pred))

            # non_empty_column = (pred != -1)
            # new_idxs = torch.randperm(total_idxs[non_empty_column].shape[0])
            # select_idxs = total_idxs[non_empty_column][new_idxs][:50]
            # random_confidence = conf[select_idxs]
            # random_x = data.x[select_idxs]
            # x_embed = data.x
            # lr_model = LinearRegression(x_embed.shape[1])
            # train_lr(lr_model, random_x, random_confidence, 100)
            # lr_pred = inference_lr(lr_model, x_embed).reshape(-1)

            # new_mask = torch.ones(len(data.y), dtype = torch.bool)
            # new_mask[select_idxs] = 0
            # lr_pred[new_mask] = 0
            # max_conf_id = torch.argsort(lr_pred, descending=True)[:50]

            # few_shot_all_query(orig_question, data, configs, dataname, max_conf_id)

            # import ipdb; ipdb.set_trace()
            
            ## vanilla
            ## cot 
            # cot_prompt = [exp.cot_confidence(q) for q in no_task_question]
            ## multi step
            # multi_step_prompt = [exp.multi_step_confidence(q) for q in no_task_question]
            ## topk
            # topk_step_prompt = [exp.topk_confidence(q) for q in no_task_question]
            # cot_prompt = pack_prompt(cot_prompt, select_ids, data.x.shape[0])

            # multi_step_prompt = pack_prompt(multi_step_prompt, select_ids, data.x.shape[0])

            # topk_step_prompt = pack_prompt(topk_step_prompt, select_ids, data.x.shape[0])

            ## zero-shot vanilla performance
            
                # time.sleep(30)
                # continue

            if 'zero_shot' in exps:
                print("zero shot all")
                vanilla_prompt = [exp.vanilla_verbalized_confidence(q) for q in questions]
                prompt_cost = calculate_cost_from_a_list_of_texts(vanilla_prompt, select_ids)
                vanilla_prompt = pack_prompt(vanilla_prompt, select_ids, data.x.shape[0])
                vanilla_response, error_idx = exp.async_api(vanilla_prompt, data.label_names, temperature = 0, top_p = None, n = 1, dataset = dataname, key = 'zero_shot', rewrite = False, fix = True, seed = seed, sp = 60, ss = 0)
                exp.save_cache(dataname, 'zero_shot', vanilla_response)
                res = exp.retrieve_answer(vanilla_response, data)
                eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='zero_shot', strategy = args.strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
                performance[dataname]['zero_shot'].append(eval_out)
                response_str = [x[0] for x in vanilla_response if x != '']
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['zero_shot'].append(all_costs)
            
            if 'no_fix' in exps:
                print("No fix")
                vanilla_prompt = [exp.vanilla_verbalized_confidence(q) for q in questions]
                prompt_cost = calculate_cost_from_a_list_of_texts(vanilla_prompt, select_ids)
                vanilla_prompt = pack_prompt(vanilla_prompt, select_ids, data.x.shape[0])
                vanilla_response, error_idx = exp.test_no_correction(vanilla_prompt, data.label_names, temperature = 0, top_p = None, n = 1, dataset = dataname, key = 'zero_shot', rewrite = False, fix = True, seed = seed, sp = 60, ss = 0)
                # exp.save_cache(dataname, 'zero_shot', vanilla_response)
                res = exp.retrieve_answer(vanilla_response, data)
                eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='no_fix', strategy = args.strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
                performance[dataname]['zero_shot'].append(eval_out)
                response_str = [x[0] for x in vanilla_response if x != '']
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['zero_shot'].append(all_costs)

            
            if 'few_shot_all' in exps:
                print("few shot all")
                few_shot_all_prompts = few_shot_topk_prompts(orig_question, data.label_names, fst_example, fst_result, exp, object_cat, question, answer_format, name = dataname)
                prompt_cost = calculate_cost_from_a_list_of_texts(few_shot_all_prompts, select_ids)
                few_shot_all_prompts = pack_prompt(few_shot_all_prompts, select_ids, data.x.shape[0])
                few_shot_all_response, error_idx = exp.async_api(few_shot_all_prompts, data.label_names, 0.7, None, n = 3, dataset = dataname, key = 'few_shot_all', rewrite = False, fix = True, seed = seed, sp = 60, ss = 3)
                exp.save_cache(dataname, 'few_shot_all', few_shot_all_response)
                res = exp.retrieve_multiple_answers(few_shot_all_response, data, 3)
                eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='few_shot_all', strategy = args.strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
                performance[dataname]['few_shot_all'].append(eval_out)
                response_str = [x[0] + x[1] + x[2] for x in few_shot_all_response if x != '']
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['few_shot_all'].append(all_costs)

            ## few-shot vanilla performance
            if 'few_shot' in exps:
                print("few shot")
                few_shot_prompt = few_shot_prompts(orig_question, data.label_names, example_text, example_labels, object_cat, question, answer_format)
                prompt_cost = calculate_cost_from_a_list_of_texts(few_shot_prompt, select_ids)
                few_shot_prompt = pack_prompt(few_shot_prompt, select_ids, data.x.shape[0])
                few_shot_response, error_idx = exp.async_api(few_shot_prompt, data.label_names, temperature = 0, top_p = None, n = 1, dataset = dataname, key = 'few_shot', rewrite = False, fix = True, seed = seed, sp = 60, ss = 2)
                exp.save_cache(dataname, 'few_shot', few_shot_response)
                res = exp.retrieve_answer(few_shot_response, data)
                eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='few_shot', strategy = args.strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
                performance[dataname]['few_shot'].append(eval_out)
                response_str = [x[0] for x in few_shot_response if x != '']
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['few_shot'].append(all_costs)

            ## cot performance
            # cot_prompt = [exp.cot_confidence(q) for q in no_task_question]
            # cot_prompt = pack_prompt(cot_prompt, select_ids, data.x.shape[0])
            # cot_response, error_idx = exp.async_api(cot_prompt, data.label_names, 0, None, n = 1, dataset = dataname, key = 'cot', rewrite = True, fix = True, seed = seed, sp = 60, ss = 0)
            # exp.save_cache(dataname, 'cot', cot_response)
            # res = exp.retrieve_answer(cot_response, data)
            # eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='cot', strategy = args.strategy, data_obj = data, seed = seed)
            # performance[dataname]['cot'].append(eval_out)

            ## topk performance
            if 'topk' in exps:
                print("topk")
                topk_step_prompt = [exp.topk_confidence(q, 3, dataname) for q in no_task_question]
                prompt_cost = calculate_cost_from_a_list_of_texts(topk_step_prompt, select_ids)
                topk_step_prompt = pack_prompt(topk_step_prompt, select_ids, data.x.shape[0])
                topk_response, error_idx = exp.async_api(topk_step_prompt, data.label_names, 0, None, n = 1, dataset = dataname, key = 'topk', rewrite = False, fix = True, seed = seed, sp = 60, ss = 0)
                exp.save_cache(dataname, 'topk', topk_response)
                res = exp.retrieve_answer(topk_response, data)
                eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='topk', strategy = args.strategy, data_obj = data, seed = seed)
                performance[dataname]['topk'].append(eval_out)
                response_str = [x[0] for x in topk_response if x != '']
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['topk'].append(all_costs)

            if 'consistency' in exps:
                print("consistency")
                topk_step_prompt = [exp.topk_confidence(q, 3, dataname) for q in no_task_question]
                topk_step_prompt = pack_prompt(topk_step_prompt, select_ids, data.x.shape[0])
                prompt_cost = calculate_cost_from_a_list_of_texts(topk_step_prompt, select_ids)
                acc, response_str = consistency(topk_step_prompt, dataname, exp, 0.7, data, strategy = args.strategy, seed = seed)
                # exp.save_cache(dataname, 'consistency', consistency_response)
                # res = exp.retrieve_multiple_answers(consistency_response, data, 3)
                # eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='consistency', strategy = args.strategy, data_obj = data, seed = seed)
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                performance[dataname]['consistency'].append(acc)
                cost[dataname]['consistency'].append(all_costs)
                # time.sleep(30)

            if 'consistency_no_topk' in exps:
                print("consistency no topk")
                vanilla_prompt = [exp.vanilla_verbalized_confidence(q) for q in questions]
                vanilla_prompt = pack_prompt(vanilla_prompt, select_ids, data.x.shape[0])
                prompt_cost = calculate_cost_from_a_list_of_texts(vanilla_prompt, select_ids)
                acc, response_str = consistency_no_topk(vanilla_prompt, dataname, exp, 0, data, strategy = args.strategy, seed = seed, key_name = 'consistency_0_vanilla')
                # exp.save_cache(dataname, 'consistency_0_vanilla', consistency_no_topk_response)
                # res = exp.retrieve_multiple_answers(consistency_no_topk_response, data, 3)
                # eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='consistency_no_topk', strategy = args.strategy, data_obj = data, seed = seed)
                performance[dataname]['consistency_no_topk'].append(acc)
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['consistency_no_topk'].append(all_costs)
                # time.sleep(30)
            
            

            # ipdb.set_trace()
        if 'zero_shot' in exps:
            mean_test_acc = np.mean(performance[dataname]['zero_shot']) * 100
            std_test_acc = np.std(performance[dataname]['zero_shot']) * 100
            avg_cost = np.mean(cost[dataname]['zero_shot'])
            print(f"Zero-shot Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Zero-shot Average Cost: {avg_cost:.2f}")

        if 'few_shot' in exps:
            mean_test_acc = np.mean(performance[dataname]['few_shot']) * 100
            std_test_acc = np.std(performance[dataname]['few_shot']) * 100
            avg_cost = np.mean(cost[dataname]['few_shot'])
            print(f"Few-shot Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Few-shot Average Cost: {avg_cost:.2f}")

        # mean_test_acc = np.mean(performance[dataname]['cot'])
        # std_test_acc = np.std(performance[dataname]['cot'])
        # print(f"COT Test Accuracy: {mean_test_acc} ± {std_test_acc}")

        if 'topk' in exps:
            mean_test_acc = np.mean(performance[dataname]['topk']) * 100
            std_test_acc = np.std(performance[dataname]['topk']) * 100
            avg_cost = np.mean(cost[dataname]['topk'])
            print(f"Top-k Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Top-k Average Cost: {avg_cost:.2f}")

        if 'consistency' in exps:
            mean_test_acc = np.mean(performance[dataname]['consistency']) * 100
            std_test_acc = np.std(performance[dataname]['consistency']) * 100
            avg_cost = np.mean(cost[dataname]['consistency'])
            print(f"Consistency Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Consistency Average Cost: {avg_cost:.2f}")

        if 'consistency_no_topk' in exps:
            mean_test_acc = np.mean(performance[dataname]['consistency_no_topk']) * 100
            std_test_acc = np.std(performance[dataname]['consistency_no_topk']) * 100
            avg_cost = np.mean(cost[dataname]['consistency_no_topk'])
            print(f"Consistency No Top-k Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Consistency No Top-k Average Cost: {avg_cost:.2f}")
        
        if 'few_shot_all' in exps:
            mean_test_acc = np.mean(performance[dataname]['few_shot_all']) * 100
            std_test_acc = np.std(performance[dataname]['few_shot_all']) * 100
            avg_cost = np.mean(cost[dataname]['few_shot_all'])
            print(f"Few-shot All Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Few-shot All Average Cost: {avg_cost:.2f}")

        # import ipdb; ipdb.set_trace()








    



