# HSG_ExSUM_NER
This repository presents and compares HeterSUMGraph and variants doing extractive summarization, named entity recognition or both.  
HeterSUMGraph and variants use GATv2Conv (from torch_geometric).  

HeterSUMGraph using GATv2Conv is the best variant of HeterSUMGraph, better than original HeterSUMGraph on NYT50, for more information
see: [https://github.com/Baragouine/HeterSUMGraph](https://github.com/Baragouine/HeterSUMGraph)

The dataset is a part of general geography, architecture town planning and geology French wikipedia articles.

## Clone project
```bash
git clone https://github.com/Baragouine/HSG_ExSUM_NER.git
```

## Enter into the directory
```bash
cd HSG_ExSUM_NER
```

## Create environnement
```bash
conda create --name HSG_ExSUM_NER python=3.9
```

## Activate environnement
```bash
conda activate HSG_ExSUM_NER
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Install nltk data
To install nltk data:
  - Open a python console.
  - Type ``` import nltk; nltk.download()```.
  - Download all data.
  - Close the python console.

## Scrap, preprocessing and split articles
  - Run `00-00-scrap_wiki.ipynb` to scrap data.
  - Run `00-01-raw_dataset_to_preprocessed.ipynb` to compute summarization and ner labels.
  - Run `00-02-drop_article_without_body.ipynb` to drop articles without body.
  - Run `00-03-split_preprocessed_dataset_to_25_high_25_low_0.5.ipynb` to split the previous dataset to three subsets depending of summary/article ratio (Wikipedia-0.5, Wikipedia-high-25, Wikipedia-low-25).
  - Run `00-04-split_wiki_datasets_to_train_val_test.ipynb` to split previous datasets to train, val and test set.
  - Run ```python scripts/compute_tfidf_dataset.py -input data/wiki_geo_ratio_sc_0.5.json -output data/wiki_geo_ratio_sc_0.5_dataset_tfidf.json``` (compute tfidfs for whole dataset).
  - Run ```python scripts/compute_tfidf_sent_dataset.py -input data/wiki_geo_ratio_sc_0.5.json -output data/wiki_geo_ratio_sc_0.5_sent_tfidf.json``` (compute tfidfs for each document).
  - Run ```python scripts/compute_tfidf_dataset.py -input data/wiki_geo_low_25.json -output data/wiki_geo_low_25_dataset_tfidf.json``` (compute tfidfs for whole dataset).
  - Run ```python scripts/compute_tfidf_sent_dataset.py -input data/wiki_geo_low_25.json -output data/wiki_geo_low_25_sent_tfidf.json``` (compute tfidfs for each document).
  - Run ```python scripts/compute_tfidf_dataset.py -input data/wiki_geo_high_25.json -output data/wiki_geo_high_25_dataset_tfidf.json``` (compute tfidfs for whole dataset).
  - Run ```python scripts/compute_tfidf_sent_dataset.py -input data/wiki_geo_high_25.json -output data/wiki_geo_high_25_sent_tfidf.json``` (compute tfidfs for each document).

tfidfs computing is only necessary for HeterSUMGraph based models.

## Embeddings
For training you must use french fasttext embeddings, they must have the following path: `data/cc.fr.300.vec`

## Training
  - `01-train_HeterSUMGraph.ipynb`: model for summarization only.
  - `02-train_HeterSUMGraphNER.ipynb`: model for both summarization and named entity recognition.
  - 


