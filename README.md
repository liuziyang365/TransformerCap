### Readme

---

#### Prepare data(only COCO)

* Text: ([Download](https://cs.stanford.edu/people/karpathy/deepimagesent/)） 

* image
    * raw images([Download](https://cocodataset.org/#download))
    * Buttom-up Attention([Download](https://github.com/peteanderson80/bottom-up-attention))

---

#### Process data

* For text

  ``` python scripts/prepro_text.py```

  you will obtain four files, train.json, val.json, test.json, vacal.pkl

  train.json is a dictionary with sentence id as key

  val.json and test.json both are dictionaries with image id as key

  vocal.pkl is an example of vocabulary class 

* For Buttom-up Attention

   ``` python scripts/prepro_bu.py --num_box fixed```(Only use fixed 36 features)

---

#### Training

``` python train.py  --id baseline ```

---

#### Evaluation

``` python eval.py --id baseline ```

---

#### Acknowledgenments

Thanks to a captioning base code repository [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) and a lot of codes refer to it.  

