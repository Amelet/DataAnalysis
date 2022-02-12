# PART 1: Functions for DATA visualization and preprocessing
# I will give a work_flow how to analyze data before fitting linreg model,
# but I format relevant code snippets as functions
# See it as an instruction


# PART 1.1:
# 1 visualize rent's count of bikes with variables columns (pairwise)
def plot_pairwise_dependency(df):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
    for idx, feature in enumerate(df.columns[:-1]):
        df.plot(feature, "cnt", subplots=True, kind="scatter", ax=axes[idx // 4, idx % 4])
    plt.show()


# 2 get pearson correlation between the output column and other columns:
def check_correlation_to_y(df):
    return df.iloc[:, 0:-1].corrwith(df.iloc[:, -1], method="pearson")
# if there are correlated with the output variables, we can use linreg model


# 3 find collinear columns
def check_corr_pairwise(df, countinuous_variables):
    return df.loc[:, countinuous_variables].corr(method='pearson')
# if there are collinear columns -> regularization of the model is needed


# 4 show scale difference between variables to make a decision to scale them
def columns_means(df):
    return df.mean(axis=0)