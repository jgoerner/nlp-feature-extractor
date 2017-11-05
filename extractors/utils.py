def fu_to_df(feature_union, X):
    """Helper to convert the result of a feature union to dataframe"""
    # number of features
    n_feat = len(feature_union.transformer_list)
    # names of transformer
    name_trans = [n for n, _ in feature_union.transformer_list]
    # initially all "object"
    df_raw = pd.DataFrame(feature_union.transform(X).reshape(n_feat, -1).T, columns=name_trans)
    # try to convert all numeric columns to numeric dtypes
    df_cleaned_dtypes = df_raw.copy()
    for col in df_cleaned_dtypes.columns:
        df_cleaned_dtypes[col] = pd.to_numeric(df_cleaned_dtypes[col], errors="ignore")
    return df_cleaned_dtypes

def one_hot_df(df, ignore=[], verbose=True):
    """Helper to convert str columns in a one hot way"""
    columns = df.drop(ignore, axis=1).select_dtypes(["object"]).columns.values
    if verbose:
        print("Following columns one hot encoded: {}".format(columns))
    return pd.get_dummies(df, columns=columns)
