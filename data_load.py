import torch
import json
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Budata(Dataset):

    def __init__(self, config, mode):
        super(Budata, self).__init__()
        self.mode = mode
        self.config = config
        self.text = json.load(open(self.config.cap.format(mode), 'r'))
        self.bu_att = self.config.feat
        self.ids = list(self.text.keys())

    def __getitem__(self, item):
        if self.mode == 'train':
            sent_id = self.ids[item]
            img_id = self.text[sent_id]['img_id']
            text = torch.Tensor(self.text[sent_id]['idList']).long()
            text_mask = text.gt(0)

            feat_path = os.path.join(self.bu_att, str(img_id) + '.npz')
            feat = torch.Tensor(np.load(feat_path)['feat'])

            return feat, text, text_mask, img_id

        else:
            img_id = self.ids[item]
            feat_path = os.path.join(self.bu_att, str(img_id) + '.npz')
            feat = torch.Tensor(np.load(feat_path)['feat'])

            return feat, img_id

    def __len__(self):
        return len(self.ids)


def train_collate_fn(data):
    batch_data = {}
    feat, cap, cap_mask, img_id = zip(*data)

    feat = torch.stack(feat, 0)
    cap = torch.stack(cap, 0)
    cap_mask = torch.stack(cap_mask, 0)

    batch_data['img_id'] = img_id
    batch_data['feat'] = feat
    batch_data['cap'] = cap
    batch_data['cap_mask'] = cap_mask

    return batch_data


def val_collae_fn(data):
    batch_data = {}

    feat, img_id = zip(*data)
    feat = torch.stack(feat, 0)

    batch_data['img_id'] = img_id
    batch_data['feat'] = feat
    return batch_data


# unfixed collate_fn
# def collate_fn(data):
#     batch_data = {}
#     feats, box_nums, texts, text_masks, img_ids = zip(*data)
#     bs = len(box_nums)
#     feat_dim = feats[0].size(-1)
#     text_gt_num = texts[0].size(0)
#     max_num = max(box_nums)
#
#     batch_feats = torch.zeros(bs, max_num, feat_dim)
#     for i, feat in enumerate(feats):
#         batch_feats[i, :box_nums[i], :] = feat
#     # 扩展为每张图片真实refs数量的倍数（coco一张图片有5个refs)
#     batch_feats = batch_feats.repeat(1, text_gt_num, 1).reshape(bs * text_gt_num, max_num, feat_dim)
#     feat_masks = batch_feats.gt(0).sum(-1).gt(0)
#
#     batch_data['img_ids'] = img_ids
#     batch_data['feats'] = batch_feats
#     batch_data['feat_masks'] = feat_masks
#
#     # 把ref展开成batch的形式
#     texts = torch.stack(texts, 0).reshape(bs * text_gt_num, -1)
#     text_masks = torch.stack(text_masks, 0).reshape(bs * text_gt_num, -1)
#     batch_data['texts'] = texts
#     batch_data['text_masks'] = text_masks
#
#     return batch_data

def data_load(config, mode):
    dataset = Budata(config, mode)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.train_bs if mode == 'train' else config.test_bs,
                             shuffle=True if mode == 'train' else False,
                             collate_fn=train_collate_fn if mode == 'train' else val_collae_fn,
                             num_workers=config.num_workers,
                             )

    return data_loader

# import sys
# sys.path.append('.')
# from configs.com_config import config
# from utils.vocab import Vocabulary
# if __name__ == '__main__':
#
#     data_loader = data_load(config,'train')
#     with open(config.vocab, 'rb') as f:
#         vocab = pickle.load(f)
#     for data in data_loader:
#         img_id = data['img_id']
#         feat = data['feat']
#         text = data['text']
#         text_mask = data['text_mask']
#         print(text)
