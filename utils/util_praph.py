def show_tsne(in_output, out_output, save_path, id_labels):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn import manifold
    import seaborn as sns
    from pandas import DataFrame
    concat = lambda x: np.concatenate(x, axis=0)
    data = concat((in_output, out_output)).copy()
    label = concat((id_labels, 10 * np.ones(len(out_output)))).copy()
    X, y = data, label
    X_embedded = manifold.TSNE(n_components=2).fit_transform(X)
    data = np.column_stack((X_embedded, y))
    df = DataFrame(data, columns=['_DIM_1_', '_DIM_2_', 'Label'])
    df = df.astype({'Label': 'int'})
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")

    unique_labels = np.unique(df['Label'])  # 找出所有唯一的类别标签
    palette = {label: '#FFD4A9' if label == 10 else '#80D1C8' for label in unique_labels}  # 设置调色板，加号为黑色，其他类别为蓝色
    sns.lmplot(x='_DIM_1_',
               y='_DIM_2_',
               data=df,
               fit_reg=False,
               legend=True,
               size=9,
               hue='Label',
               markers=['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
               palette=palette
               )
    # sns.scatterplot(x="_DIM_1_", y="_DIM_2_", hue='Label', style="Label", data=df)
    # plt.title('t-SNE Results: A2W', weight='bold').set_fontsize('14')
    plt.xlabel('Dimension 1', weight='bold').set_fontsize('10')
    plt.ylabel('Dimension 2', weight='bold').set_fontsize('10')
    plt.savefig(save_path)

def show_tsne2(in_output, save_path, id_labels):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn import manifold
    import seaborn as sns
    from pandas import DataFrame
    concat = lambda x: np.concatenate(x, axis=0)
    X, y = in_output, id_labels
    X_embedded = manifold.TSNE(n_components=2).fit_transform(X)
    data = np.column_stack((X_embedded, y))
    df = DataFrame(data, columns=['_DIM_1_', '_DIM_2_', 'Label'])
    df = df.astype({'Label': 'int'})
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")
    sns.lmplot(x='_DIM_1_',
               y='_DIM_2_',
               data=df,
               fit_reg=False,
               legend=True,
               size=9,
               hue='Label',
               # markers=['o', 'o']
               markers=['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']  # ,
               # scatter_kws={"s": 200, "alpha": 0.3}
               )
    # sns.scatterplot(x="_DIM_1_", y="_DIM_2_", hue='Label', style="Label", data=df)
    # plt.title('t-SNE Results: A2W', weight='bold').set_fontsize('14')
    plt.xlabel('Dimension 1', weight='bold').set_fontsize('10')
    plt.ylabel('Dimension 2', weight='bold').set_fontsize('10')
    plt.savefig(save_path)