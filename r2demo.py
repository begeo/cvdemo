import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit


def run_r2_demo():
    N = 200  # count per cluster

    def y(x):
        return (-x ** 2 + 13 * x - 12) / 3.

    xmeans = np.array([2., 4., 6.])
    ymeans = y(xmeans)

    np.random.seed(42)

    cluster0 = np.random.multivariate_normal([xmeans[0], ymeans[0]], [[1, 0], [0, 1]], N)
    cluster1 = np.random.multivariate_normal([xmeans[1], ymeans[1]], [[1, 0], [0, 1]], N)
    cluster2 = np.random.multivariate_normal([xmeans[2], ymeans[2]], [[1, 0], [0, 1]], N)

    cluster_index = np.concatenate([np.zeros(N, ), np.ones(N, ), 2 * np.ones(N, )]).astype(int)
    cluster_data = np.concatenate([cluster0, cluster1, cluster2])

    df = pd.DataFrame(data=cluster_data, columns=['x', 'y'], index=cluster_index)
    df.index.name = 'cluster'

    ax = df.reset_index().plot.scatter(x='x', y='y', c='cluster', alpha=0.5, cmap='Set1')
    xeqn = np.arange(0.9 * np.min(xmeans), np.max(xmeans * 1.1), 0.2)
    plt.plot(xeqn, y(xeqn), linestyle='--', color='k')
    plt.plot(xmeans, ymeans, 'o', color='k')

    rng = np.random.RandomState(23)
    m = linear_model.LinearRegression()
    X = df['x'].values
    y = df['y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng, test_size=0.2)
    m.fit(X_train.reshape(-1, 1), y_train)
    y_pred = m.predict(X_test.reshape(-1, 1))
    print(f'R2 = {r2_score(y_test, y_pred)}')
    plt.plot(xeqn, m.predict(xeqn.reshape(-1, 1)), linestyle='--', color='gray')
    plt.savefig('figs/demo_data.png')
    plt.show()

    # Emulate random sampling
    kf = KFold(n_splits=5, shuffle=True)
    cv_scores = cross_val_score(m, X.reshape(-1, 1), y, cv=kf, )
    print(f'CV Scores: {cv_scores}')

    # Do some shuffle-splitting to get the distribution of R2
    n_splits, test_size = 200, 0.2
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=rng)

    scores = []
    for train_ix, test_ix in ss.split(X, y):
        y_train = y[train_ix]
        X_train = X.reshape(-1, 1)[train_ix, :]
        y_test = y[test_ix]
        X_test = X.reshape(-1, 1)[test_ix, :]
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        scores.append(r2_score(y_test, y_pred))

    ax = sns.kdeplot(scores, cumulative=True, color='k')
    ax2 = ax.twinx()
    g = sns.distplot(scores, bins=20, kde=False, rug=True, axlabel='$R^2$', rug_kws={'alpha': 0.5}, ax=ax2)

    ax.set_title(f'Distribution of $R^2$ scores ({n_splits} splits; test fraction={test_size})')
    plt.savefig('figs/demo_histr2.png')
    plt.show()


if __name__ == '__main__':
    run_r2_demo()
