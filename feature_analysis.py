from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.stats import pearsonr
import pandas as pd


def tokenize_split(text):
    """Returns a list of the tokens spliting by space only given a preprocessed text (use preprocessing pipeline first)"""
    return text.split(" ")


def calculate_rcoeff(X, y, features):
    r_values, p_values = [], []
    r_dict = {"features": features, "r": [], "p-value": []}
    for i, feature in enumerate(features):
        if i % 1000 == 0:
            print("{}/{} features processed".format(i, len(features)))
        try:
            x = X[:, i]
            xlist = [x_.item(0) for x_ in x]
            ylist = [y_.item(0) for y_ in y]
            r = pearsonr(xlist, ylist)
            r_dict["r"].append(r[0])
            r_dict["p-value"].append(r[1])
        except Exception as e:
            print(e)
            r_dict["r"].append("NA")
            r_dict["p-value"].append("NA")
    return pd.DataFrame(r_dict).sort_values(by='r', ascending=False)


def get_features_normMatrix(messages, ngrams, tokenizer, max_features=50000, min_df=2, stop_words=None):
    cvec = CountVectorizer(max_features=max_features, tokenizer=tokenizer, min_df=min_df,
                           stop_words=stop_words, ngram_range=ngrams)
    cvec.fit(messages)
    matrix = cvec.transform(messages)
    nonzero = np.where(matrix.sum(axis=1) != 0)[0]
    matrix = matrix[nonzero, :]
    norm_matrix = matrix / matrix.sum(axis=1)
    features = cvec.get_feature_names()
    return features, norm_matrix, nonzero

'''
Example:
Assume train is a pandas DataFrame with the columns text (str) and label (int)
'''
train = pd.DataFrame({"text": ["I am happy", "I am sad", "Feeling great", "Feeling awful", "This is the best"], "label": [1, 0, 1, 0, 1]})
print(train,"\n")
X_train = train.text.values
y_train = train.label.values

target = [-1 if x == 0 else x for x in y_train]
y = np.asarray([target]).transpose()

features, norm_matrix, nonzero = get_features_normMatrix(X_train, (1, 1), tokenize_split, min_df=1)
uni_rcoeff_df = calculate_rcoeff(norm_matrix, y[nonzero, :], features)

pos = uni_rcoeff_df.iloc[:3]
print("Feature correlations with positive class")
print(pos)
neg = uni_rcoeff_df.iloc[-2:].reset_index(drop=True)
neg["r"] = [x*-1 for x in neg["r"]]
neg = neg.sort_values(by='r', ascending=False)
print("Feature correlations with negative class")
print(neg)

'''
Output:

               text  label
0        I am happy      1
1          I am sad      0
2     Feeling great      1
3     Feeling awful      0
4  This is the best      1 

Feature correlations with positive class
  features         r   p-value
5    happy  0.408248  0.495025
2     best  0.408248  0.495025
4    great  0.408248  0.495025

Feature correlations with negative class
  features         r   p-value
0    awful  0.612372  0.272228
1      sad  0.612372  0.272228
'''
