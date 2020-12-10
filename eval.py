import argparse
import torch
import os
import json
import pickle
from config import config
from data_load import data_load
from models.transformer import TransformerCap
from utils.metric import coco_eval
from utils.search import beam_search
from utils.vocab import Vocabulary

# ******************************
#       evaluation prepare
# ******************************

# make log direction
log_path = config.log_dir.format(config.id)

# load vocabulary
with open(config.vocab, 'rb') as f:
    vocab = pickle.load(f)

# load data
loader = data_load(config, 'test')

# load model
model = TransformerCap(config).to('cuda')
ckpt = torch.load(os.path.join(log_path, 'best_model.pt'))
model.load_state_dict(ckpt)
# ******************************
#           validation
# ******************************

model.eval()
res = []
with torch.no_grad():
    for step, data in enumerate(loader):
        img_id = data['img_id']
        feat = data['feat'].to('cuda')
        bs = len(feat)
        memory = model.encode(feat)
        sent = beam_search(model.decode, config.num_beams, memory)
        # inp = torch.ones(bs, 1).to('cuda').long()
        # for step in range(20):
        #     logit = model.decode(memory, inp)
        #     score, token_id = torch.max(logit[:, -1, :], dim=-1)
        #     inp = torch.cat([inp, token_id.unsqueeze(1)], dim=1)
        #
        for i, img_id in enumerate(img_id):
            s = vocab.idList_to_sent(sent[i].to('cpu'))
            res.append({'image_id': int(img_id), 'caption': s})
            print(f"{int(img_id)}: {s}")

    # save generated sentence
    res_path = os.path.join(log_path, 'test_result.json')
    with open(res_path, 'w') as f:
        json.dump(res, f)

    # coco evaluation
    scores = coco_eval(config.val_gts, res_path)

    sup_score = scores[config.sup_score]



