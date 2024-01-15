# HSG_ExSUM_NER (extractive summarization and named entity recognition)
This repository presents and compares HeterSUMGraph and variants doing extractive summarization, named entity recognition or both.  
  
This repository also present the influence of the summary/document ratio on performance.  
  
HeterSUMGraph and variants use GATv2Conv (from torch_geometric).  

HeterSUMGraph using GATv2Conv is the best variant of HeterSUMGraph, better than original HeterSUMGraph on NYT50, for more information
see: [https://github.com/Baragouine/HeterSUMGraph](https://github.com/Baragouine/HeterSUMGraph)

The dataset is a part of general geography, architecture town planning and geology French wikipedia articles.

Warning: this code uses a French Tokenizer.

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
preprocessing mean cleaning, labeling, etc. not mean preprocessing before training.
  - Run `00-00-scrap_wiki.ipynb` to scrap data.
  - Run `00-01-raw_dataset_to_preprocessed.ipynb` to compute summarization and ner labels.
  - Run `00-02-drop_article_without_body.ipynb` to drop articles without body.
  - Run `00-03-split_preprocessed_dataset_to_25_high_25_low_0.5.ipynb` to split the previous dataset to three subsets depending of summary/article ratio (Wikipedia-0.5, Wikipedia-high-25, Wikipedia-low-25).
  - Run `00-04-split_wiki_datasets_to_train_val_test.ipynb` to split previous datasets to train, val and test set.
  - Run ```python scripts/compute_tfidf_dataset.py -input data/wiki_geo_ratio_sc_0.5.json -output data/wiki_geo_ratio_sc_0.5_dataset_tfidf.json -docs_col_name flat_contents``` (compute tfidfs for whole dataset).
  - Run ```python scripts/compute_tfidf_sent_dataset.py -input data/wiki_geo_ratio_sc_0.5.json -output data/wiki_geo_ratio_sc_0.5_sent_tfidf.json -docs_col_name flat_contents``` (compute tfidfs for each document).
  - Run ```python scripts/compute_tfidf_dataset.py -input data/wiki_geo_low_25.json -output data/wiki_geo_low_25_dataset_tfidf.json -docs_col_name flat_contents``` (compute tfidfs for whole dataset).
  - Run ```python scripts/compute_tfidf_sent_dataset.py -input data/wiki_geo_low_25.json -output data/wiki_geo_low_25_sent_tfidf.json -docs_col_name flat_contents``` (compute tfidfs for each document).
  - Run ```python scripts/compute_tfidf_dataset.py -input data/wiki_geo_high_25.json -output data/wiki_geo_high_25_dataset_tfidf.json -docs_col_name flat_contents``` (compute tfidfs for whole dataset).
  - Run ```python scripts/compute_tfidf_sent_dataset.py -input data/wiki_geo_high_25.json -output data/wiki_geo_high_25_sent_tfidf.json -docs_col_name flat_contents``` (compute tfidfs for each document).

tfidfs computing is only necessary for HeterSUMGraph based models.

## Embeddings
For training you must use french fasttext embeddings, they must have the following path: `data/cc.fr.300.vec`

## Training
Run one of the \*train\* notebooks to train and evaluate the associated model:
The names of notebooks containing HeterSUMGraph mean that they can be used to train HeterSUMGraph. If the name contains GAT, it means that the notebook trains the original version of HeterSUMGraph. If the name contains GATv2, it means that the GAT layer has been replaced by GATv2. If it contains NER without the "Only", it means that the notebook performs summary and named entity recognition. If it contains OnlyNER, it means that the model only performs named entity recognition; if the name contains POL, it means that edge features are taken into account for the NER; finally, if instead of HeterSUMGraph we have HSGRNN, it means that the model is a combination of HeterSUMGraph and SummaRuNNer. 

## Result

### All XPs scores
see: https://www.overleaf.com/read/gbfxvfvykxsc#77a14f  

### HeterSUMGraph (GATv2Conv, limited-length ROUGE Recall)
| dataset | ROUGE-1 | ROUGE-2 | ROUGE-L |  
|:-:      |:-:      |:-:      |:-:      |  
| Wikipedia-0.5 |29.1 &plusmn; 0.0|8.6 &plusmn; 0.0|18.9 &plusmn; 0.0|  
| Wikipedia-high-25 |23.8 &plusmn; 0.0|6.8 &plusmn; 0.0|14.9 &plusmn; 0.0|  
| Wikipedia-low-25 |33.1 &plusmn; 0.0|13.3 &plusmn; 0.0|22.9 &plusmn; 0.0|  

### Other models on Wikipedia-0.5 (wiki_geo_ratio_sc_0.5) (limited-length ROUGE Recall)
| model | ROUGE-1 | ROUGE-2 | ROUGE-L | BCELoss |  
|:-:      |:-:      |:-:      |:-:      |:-:          |  
| HeterSUMGraph\_GAT         | 31.11 $\pm$ 0.85    | 9.79 $\pm$ 0.73     | 19.59 $\pm$ 0.58    | N/A               |
| HeterSUMGraphNER\_GAT      | 31.70 $\pm$ 0.12    | 10.22 $\pm$ 0.15    | 20.02 $\pm$ 0.12    | 0.926+/-0.000     |
| HeterSUMGraphOnlyNER\_GAT  | N/A                 | N/A                 | N/A                 | 0.929+/-0.001     |
| HeterSUMGraphNERPOL\_GAT   | N/A                 | N/A                 | N/A                 | N/A               |
| HeterSUMGraph\_GATv2       | 31.56 $\pm$ 0.29    | 10.12 $\pm$ 0.30    | 19.91 $\pm$ 0.28    | N/A               |
| HeterSUMGraphNER\_GATv2    | 31.66 $\pm$ 0.13    | 10.22 $\pm$ 0.09    | 20.01 $\pm$ 0.10    | 0.925+/-0.001     |
| HeterSUMGraphOnlyNER\_GATv2| N/A                 | N/A                 | N/A                 | 0.930+/-0.001     |
| HSGRNN\_GATv2              | 30.86 $\pm$ 0.00    | 9.29 $\pm$ 0.00     | 19.59 $\pm$ 0.00    | N/A               |
| HSGRNNNER\_GATv2           | 31.52 $\pm$ 0.10    | 10.06 $\pm$ 0.09    | 19.97 $\pm$ 0.05    | 0.926+/-0.000     |
| HSGRNNOnlyNER\_GATv2       | N/A                 | N/A                 | N/A                 | 0.930+/-0.001     |
 

&ast; Wikipedia-0.5: general geography, architecture town planning and geology French wikipedia articles with len(summary)/len(content) <= 0.5.  
&ast; Wikipedia-high-25: first 25% of general geography, architecture town planning and geology French wikipedia articles sorted by len(summary)/len(content) descending.  
&ast; Wikipedia-low-25: first 25% of general geography, architecture town planning and geology French wikipedia articles sorted by len(summary)/len(content) ascending.  
