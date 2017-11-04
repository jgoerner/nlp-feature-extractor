from sklearn.pipeline import FeatureUnion

def fu_to_df(feature_union, X):
    """Convert result of a feature union to pandas df"""
    # number of features
    n_feat = len(feature_union.transformer_list)
    # names of transformer
    name_trans = [n for n, _ in fu.transformer_list]
    return pd.DataFrame(fu.transform(X).reshape(n_feat, -1).T, columns=name_trans)
