import numpy as np


def add_explanatory_power(df, derived):
    """
    Computes the explanatory power for all elements in the dataframe.
    :param df: pandas dataframe.
    :param derived: boolean, if derived measures are used.
    :return: pandas dataframe with added column for the explanatory power.
    """
    if derived:
        F_a = df['predict_a'].sum()
        F_b = df['predict_b'].sum()

        n = (df['real_a'] - df['predict_a']) * F_b - (df['real_b'] - df['predict_b']) * F_a
        d = F_b * (F_b + df['real_b'] - df['predict_b'])
        df['ep'] = n / d

        # Normalize to sum up to 1
        df['ep'] = df['ep'] / df['ep'].sum()
    else:
        F = df['predict'].sum()
        A = df['real'].sum()

        df['ep'] = (df['real'] - df['predict']) / (A - F)
    return df


def add_surpise(df, derived, merged_divide=1):
    """
    Computes the surprise for all elements in the dataframe.
    :param df: pandas dataframe.
    :param derived: boolean, if derived measures are used.
    :param merged_divide: int, if the total sum should be divided
      (this is true if the dataframe elements have been merged over the dimensions
      as done in the adtributor code).
    :return: dataframe with added column for the surprise.
    """
    def compute_surprise(col_real, col_predict):
        with np.errstate(divide='ignore'):
            F = df[col_predict].sum() / merged_divide
            A = df[col_real].sum() / merged_divide

            p = df[col_predict] / F
            q = df[col_real] / A
            p_term = np.nan_to_num(p * np.log(2 * p / (p + q)))
            q_term = np.nan_to_num(q * np.log(2 * q / (p + q)))
            surprise = 0.5 * (p_term + q_term)
        return surprise

    if derived:
        df['surprise'] = compute_surprise('real_a', 'predict_a') + compute_surprise('real_b', 'predict_b')
    else:
        df['surprise'] = compute_surprise('real', 'predict')
    return df


def add_deviation_score(df):
    """
    Computes the deviation score for all elements in the dataframe.
    :param df: pandas dataframe.
    :return: dataframe with added column for the deviation score.
    """
    df['deviation'] = 2 * (df['predict'] - df['real']).divide(df['predict'] + df['real']).fillna(0.0)
    return df
