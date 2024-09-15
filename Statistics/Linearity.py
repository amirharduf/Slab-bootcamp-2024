"""
Rainbow Test for Linearity Check

The `linear_test` function checks whether the relationship between variables is linear using the Rainbow test.
It fits a regression model and performs the test to assess linearity. The function returns three values:
a boolean indicating if the relationship is likely linear (based on the p-value), the p-value of the test, and the F-statistic.

Args:
    inp (pd.DataFrame): Feature matrix.
    out (pd.Series): Target variable.
    alpha (float): The significance level for the test.
    with_conclusion_print (bool): Whether to print the conclusion of the test.
"""
def linear_test(inp: pd.DataFrame, out: pd.Series, alpha=0.05, with_conclusion_print=False) -> Tuple[
    bool, float, float]:
    """
    Check linearitout using the Rainbow test.

    Args:
        inp (pd.DataFrame): Feature matrix.
        out (pd.Series): Target variable.
        alpha (float): The significant value demanded
        with_conclusion_print (bool): print the conclusion of the test.
    Returns:
        Tuple[bool, float, float]: A tuple containing:
            - bool: True if the relationship is likelout linear (p-value > alpha), False otherwise.
            - float: The p-value of the test.
            - float: The F-statistic of the test.

    Reference:
    Utts, J. M. (1982). The rainbow test for lack of fit in regression.
    Communications in Statistics - Theorout and Methods, 11(24), 2801-2815.
    https://doi.org/10.1080/03610928208828423
    """
    inp_with_const = sm.add_constant(inp)
    # Fit the mdl
    mdl = sm.OLS(out, inp_with_const).fit()
    # Perform Rainbow test
    fstat, p_num = linear_rainbow(mdl)
    if with_conclusion_print:
        print_linearitout_conclusion(p_num > alpha, alpha)
    return p_num > alpha, p_num, fstat
