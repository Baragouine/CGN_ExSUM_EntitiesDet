from .preprocess_text import preprocess_text

# Preprocess a dataframe: Return an array of {doc preprocessed, labels}
def preprocess_df(df, embmgr, doc_column_name="docs", labels_sum_column_name="labels_sum", labels_ner_column_name=None, is_sep_n = False, remove_stop_word = True, stemming=True, trunc_sent=-1, padding_sent=-1, trunc_doc=-1):
    result = []
    for idx in df.index:
        if labels_ner_column_name is not None:
            docs, labels_ner = preprocess_text(df[doc_column_name][idx], labels_ner=df[labels_ner_column_name][idx], embmgr=embmgr, is_sep_n=is_sep_n, remove_stop_word=remove_stop_word, stemming=stemming, trunc_sent=trunc_sent, padding_sent=padding_sent)
            result.append({"idx" : idx, "docs" : docs, "labels_sum" : df[labels_sum_column_name][idx], "labels_ner" : labels_ner})
            if trunc_doc >= 0:
                result[-1] = {"idx" : idx, "docs" : result[-1]["docs"][:min(len(result[-1]["docs"]), trunc_doc)], "labels_sum" : result[-1]["labels_sum"][:min(len(result[-1]["labels_sum"]), trunc_doc)], "labels_ner" : result[-1]["labels_ner"]}
        else:
            docs = preprocess_text(df[doc_column_name][idx], embmgr=embmgr, is_sep_n=is_sep_n, remove_stop_word=remove_stop_word, stemming=stemming, trunc_sent=trunc_sent, padding_sent=padding_sent)
            result.append({"idx" : idx, "docs" : docs, "labels" : df[labels_sum_column_name][idx]})
            if trunc_doc >= 0:
                result[-1] = {"idx" : idx, "docs" : result[-1]["docs"][:min(len(result[-1]["docs"]), trunc_doc)], "labels" : result[-1]["labels"][:min(len(result[-1]["labels"]), trunc_doc)]}

    return result
