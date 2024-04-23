# 残差正态性分析

本repositories主要用于时间序列分析任务中模型诊断中的残差正态性分析，应用Shapiro-Wilk正态检验得到统计量和对应的p值，并对拟合的残差分别绘制标准残差直方图和分位数-分位数(QQ)图。这两组图显示了数据的正态分布性。

> 一个有名的正态分布性检验是Shapiro-Wilk检验。该检验的本质是计算残差与其相应的正态分布分位数之间的相关系数，相关性越小，就越有理由否定正态性。

<img src="images/Hist_QQplot.jpg" alt="Hist_QQplot" style="zoom:33%;" />

## Demo

~~~python
from shaprio_wilk import ResNormCheck
import numpy as np
import statsmodels.api as sm

# 生成随机漫步数据
np.random.seed(0)
data = np.random.randn(100).cumsum()
# 拟合模型
model = sm.tsa.ARIMA(data, order=(1, 0, 1))
results = model.fit()
# 获取残差
residuals = results.resid

# 绘制图像
ResNormCheck(residuals, save=False)
~~~

## 标准残差直方图+分位数-分位数图

~~~python
def ResQQplot(residuals, save=False, dpi=1500):
    import numpy as np
    from statsmodels.graphics.gofplots import qqplot
    from scipy.stats import shapiro
    from matplotlib.pylab import mpl
    import seaborn as sns
    sns.set_theme(style="ticks")

    stat, p_value = shapiro(residuals)
    print(f'Shapiro-Wilk Test Stat={stat}, p-value={p_value}')

    fig = qqplot(residuals, line='s',
                 markerfacecolor='white', markeredgecolor='k', markersize=7.5)
    ax = fig.axes[0]

    for line in ax.get_lines():
        if line.get_linestyle() == '-':
            line.set_color('#DC143C')
            line.set_linewidth(2.1)

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    ax.set_xlabel("理论分位数", fontsize=12.5)
    ax.set_ylabel("样本分位数", fontsize=12.5)
    ax.grid(which='major', color='gray', linestyle='--', lw=0.5, alpha=0.8)

    stat = np.round(stat, 4)
    p_value = np.round(p_value, 4)
    ax.text(0.05, 0.95, f'$W$ = {stat}\n$p$ = {p_value}', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left', fontsize=13.5, color='k')

    if save:
        fig.savefig("QQplot.jpg", dpi=dpi, bbox_inches='tight')
~~~

## 分位数-分位数图

~~~python
def ResNormCheck(residuals, bins=13, save=False, dpi=1500):
    import numpy as np
    from statsmodels.graphics.gofplots import qqplot
    from scipy.stats import shapiro
    from matplotlib.pylab import mpl
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set_theme(style="ticks")

    stat, p_value = shapiro(residuals)
    print(f'Shapiro-Wilk Test Stat={stat}, p-value={p_value}')

    fig, ax = plt.subplots(1, 2, figsize=(12.1, 5))
    fig.subplots_adjust(wspace=0.16)

    ax[0].hist(residuals, bins=bins, alpha=1, color='w', edgecolor='k', lw=1.2)
    qqplot(residuals, line='s', ax=ax[1],
           markerfacecolor='white', markeredgecolor='k', markersize=7.5)
    for line in ax[1].get_lines():
        if line.get_linestyle() == '-':
            line.set_color('#DC143C')
            line.set_linewidth(2.1)

    ax[0].grid(which='major', color='gray', linestyle='--', lw=0.5, alpha=0.8)
    ax[1].grid(which='major', color='gray', linestyle='--', lw=0.5, alpha=0.8)

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    ax[0].set_xlabel("标准残差", fontsize=12.5)
    ax[0].set_ylabel("频率", fontsize=12.5)

    ax[1].set_xlabel("理论分位数", fontsize=12.5)
    ax[1].set_ylabel("样本分位数", fontsize=12.5)

    mean = np.round(np.mean(residuals), 4)
    std = np.round(np.std(residuals), 4)
    stat = np.round(stat, 4)
    p_value = np.round(p_value, 4)

    ax[0].text(0.05, 0.95, f'$\mu$ = {mean}\n$\sigma$ = {std}',
               transform=ax[0].transAxes, verticalalignment='top',
               horizontalalignment='left', fontsize=13.5, color='k')
    ax[1].text(0.05, 0.95, f'$W$ = {stat}\n$p$ = {p_value}', transform=ax[1].transAxes,
               verticalalignment='top', horizontalalignment='left', fontsize=13.5, color='k')

    if save:
        fig.savefig("Hist_QQplot.jpg", dpi=dpi, bbox_inches='tight')
~~~