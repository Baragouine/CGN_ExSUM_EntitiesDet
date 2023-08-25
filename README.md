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
  - Run  `00-04-split_wiki_datasets_to_train_val_test.ipynb` to split previous datasets to train, val and test set.
  - 

