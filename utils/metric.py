import sys
sys.path.append('coco-caption')

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def coco_eval(ref_json, sen_json):
    '''compute coco metrics'''

    coco = COCO(ref_json)
    cocorefs = coco.loadRes(sen_json)
    cocoEval = COCOEvalCap(coco, cocorefs)
    # val_img_num ～= 40000, we only use 5000 for validation，other 5000 for test，others for training
    cocoEval.params['image_id'] = cocorefs.getImgIds()
    cocoEval.evaluate()
    return cocoEval.eval