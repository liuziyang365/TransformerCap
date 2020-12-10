import argparse
import json
import pickle
import sys

sys.path.append('.')
from utils.vocab import Vocabulary


def main(args):
    karp_data = json.load(open(args.karp_path, 'r'))['images']
    vocab = Vocabulary()

    # count word frequency
    counter = {}
    for item in karp_data:
        for token_items in item['sentences']:
            tokenList = token_items['tokens']
            for token in tokenList:
                counter[token] = counter.get(token, 0) + 1

    cand_word = [token for token, f in counter.items() if f > args.threshold]
    for w in cand_word:
        vocab.add_word(w)
    vocab_path = args.out_path + 'vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f'vocab size: {vocab.get_size()}, saved to {vocab_path}')

    # re-split dataset and convert tokenList to idList
    train = {}
    val = {}
    test = {}

    for item in karp_data:
        split = item['split']
        img_id = item['cocoid']
        img_path = item['filepath'] + '/' + item['filename']

        for i, token_items in enumerate(item['sentences']):
            sent_id = token_items['sentid']
            tokenList = token_items['tokens']
            idList, length = vocab.tokenList_to_idList(tokenList, args.fixed_len)

            train_item = {'img_id': img_id,
                          'img_path': img_path,
                          'idList': idList,
                          'len': length,
                          }

            other_item = {
                'img_id': img_id,
                'img_path': img_path,
            }

            if split == 'test':
                test[img_id] = other_item
            elif split == 'val':
                val[img_id] = other_item
            else:
                train[sent_id] = train_item

    # save files
    train_path = args.out_path + 'train.json'
    val_path = args.out_path + 'val.json'
    test_path = args.out_path + 'test.json'

    with open(train_path, 'w') as f:
        json.dump(train, f)
    with open(val_path, 'w') as f:
        json.dump(val, f)
    with open(test_path, 'w') as f:
        json.dump(test, f)

    print(f'train sentences num: {len(train)}, saved to {train_path}')
    print(f'val images num: {len(val)}, saved to {val_path}')
    print(f'test images num: {len(test)}, saved to {test_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--karp_path', default='./data/dataset_coco.json')
    parser.add_argument('-threshold', default=5, help='the lowest work frequency of word added to vocabulary')
    parser.add_argument('--fixed_len', default=16, help='fixed length of a caption')
    parser.add_argument('--out_path', default='./data/', help='save train.json, val.json, test.json and vocab.pkl')
    args = parser.parse_args()
    main(args)
