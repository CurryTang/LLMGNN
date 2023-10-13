from transformers import pipeline
import torch
import random
from openail.utils import load_mapping_2
from tqdm import tqdm
import ipdb
from helper.train_utils import seed_everything
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", cachedir="/localscratch/czk", device=0)


seed_everything(42)
for dataset in ['arxiv', 'products']:
    obj = torch.load(f'../../ogb/preprocessed_data/new/{dataset}_fixed_sbert.pt', map_location='cpu')
    raw_texts = obj.raw_texts
    test_num = 500
    total_idxs = list(range(len(raw_texts)))
    selected_ids = random.sample(total_idxs, test_num)
    test_texts = [raw_texts[idx] for idx in selected_ids]
    test_labels = [obj.y[idx] for idx in selected_ids]
    res = load_mapping_2()
    mapping = res
    label_names = [mapping[dataset][x] for x in obj.label_names]
    category_names = [mapping[dataset][x] for x in obj.category_names]
    gt = [category_names[idx] for idx in selected_ids]

    # pred = []
    # gt = []
    a = 0
    for i in tqdm(range(test_num)):
        t = test_texts[i]
        c = gt[i]
        output = classifier(t, candidate_labels=label_names)
        # ipdb.set_trace()
        l = output['labels'][0]
        if l == c:
            a += 1
    print(f'{dataset} acc: {a/test_num}')




