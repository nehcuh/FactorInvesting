# Table of Contents

1.  [因子组合排序法 &#x2013; 因子投资方法与实践之案例实现之一](#orgcaa39e2)
    1.  [引言](#orgedbfe4c)
    2.  [因子模拟投资组合](#org4768382)
    3.  [排序法及其检验](#org66ba619)
        1.  [1. 基本假设](#org1673a38)
        2.  [2. 因子生成](#org6d5955f)
        3.  [3. 因子排序 (分位处理)](#org3b361c2)
        4.  [4. 对组合进行加权处理 (这里按照市值进行加权)](#org1626995)
        5.  [5. t-检验](#org125e86e)
    4.  [结语](#org5371375)

<a id="orgcaa39e2"></a>

# 因子组合排序法 &#x2013; 因子投资方法与实践之案例实现之一

<a id="orgedbfe4c"></a>

## 引言

石川博士的新书 《因子投资方法与实践》 出版已经有段时间，不少量化圈内的朋友都人手一本，看书快的朋友甚至
已经把全书过完，写了读书笔记。

相比这些朋友，笔者更喜欢慢工出细活，慢慢咀嚼石川这本书。实际上，市面上讲因子投资的书很多，最为有名的国外
的两本书，分别是 《主动投资组合管理：创造高收益并控制风险的量化投资方法》和
《量化股票组合管理：积极型投资组合构建和管理的方法》，笔者也建议感兴趣的读者从这两本书读起。至于石川这本
书，可能不太合适零基础的朋友，不过相比另外两本专业书，石川博士的书体系性很强，而且加入了挺多学界的最新研
究成果，加上书中主要针对 A 股市场进行论述，对于做多因子分析而言，还是非常具有参考和借鉴意义的。既然是方法
论的书籍，书中的很多案例还是非常值得我们去复现一遍的，毕竟，说不如做嘛。

之前在微信群里也问了下，貌似很多人对复现书中案例无感。笔者认为，复现的意义，不仅仅是对方法论的细化理解，更
能在实践过程中，锻炼自己抽象能力，编程能力，以及对于细节的处理能力。而对于细节的考虑和处理，至少从笔者个人
经验来说，貌似挺多人都是比较薄弱的。

本文将按照石川博士的书中第二章用到的案例，通过复现，试图对一些案例能够更深的理解，同时，对于一些想做多因子
分析，但是对 Python 的高级使用比较差的朋友，笔者非常建议跟随笔者的代码，一些去学习一些 pandas 等常用数据
分析库的用法。

<a id="org4768382"></a>

## 因子模拟投资组合

在正式进入代码实战之前，首先对相关的概念进行简单介绍。首先是 \***\*因子模拟投资组合\*\***

> 因子模拟投资组合是使用股票资产，围绕某目标因子构建的投资组合：该投资组合需满足以下两个条件：
>
> - 该投资组合仅在目标因子上有大于 0 的暴露，在其他因子上的暴露为 0
> - 在所有满足条件一的投资组合中，该投资组合的特质性风险最小

当有了因子模拟投资组合后，就可以相应地计算因子收益率。从定义出发，因子模拟投资组合就是针对某目标因子构建的
投资组合：在条件一和条件二的约束下，该投资组合的收益率尽可能地仅由目标因子驱动，因此相应的投资组合收益就是
对应的因子收益率。

但是，这里有个问题，构建因子模拟投资组合需要知道股票在不同因子上的暴露，但是根据多因子模型<sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>，股票在某因子 \(i\)
上的暴露 \(\beta_i\) 反映的是在控制了其他因子后，该目标因子的收益率变化对股票超额收益变化的影响程度。这意味着
首先需要知道因子收益率才能计算因子暴露，但是因子收益率又需要构建模拟投资组合得到，而构建模拟投资组合又需要
知道因子暴露，由此，陷入了 “先鸡先蛋” 的悖论中。

为此，一个变通的做法就是用排序法来绕过这样的矛盾。譬如下面介绍的账面市值比 (book-to-market value, BM, 市净率倒数),
BM 是一个估值指标，但是学术界惯例称之为价值因子，下面将通过因子构建，分位处理，收益率处理，加权处理，对排序
法进行说明。

<a id="org66ba619"></a>

## 排序法及其检验

<a id="org1673a38"></a>

### 1. 基本假设

> 排序法中最核心的思想是使用个股在该变量上的取值大小来代替个股在该因子上的暴露的高低。但是，该方法并没有假设
> 变量的取值等于因子暴露，也没有假设两者满足某种特定的数学关系。该方法仅仅是假设变量和因子暴露是相关的。

<a id="org6d5955f"></a>

### 2. 因子生成

1.  BM 的计算公式 (每股净资产 / 股价)

    \begin{equation}
    BM = 1/PS = netAssetsPerShare / PricePerShare
    \end{equation}

2.  财务数据处理
    - 直接使用 QUANTAXIS 处理过的通达信财务数据，具体数据保存可以参考 [QUANTAXIS github 地址](https://github.com/QUANTAXIS/QUANTAXIS)
    - 考虑到不同公司财报公布时间不一致，这里采用了一个假设，即下个季报公布日期为上个财报公布截止日期，
      举例来说，&ldquo;2020-06-30&rdquo; 这一天才可以得到全市场的 &ldquo;2020-03-31&rdquo; 的财务数据，因此，最终得到的
      财务数据，统一做一个往下平移处理
    - 股价原则上应该采用不复权的价格，因为仅考虑某个因子暴露，不应该做前复权处理；此外，假设做模拟
      投资组合，在历史上的截面上做资产组合时，用到的是不复权的真实价格
    - 简单起见，笔者做了几层过滤：
      (1) 在研究的历史阶段内，所有股票必须保证都有财务数据
      (2) 在对应时间截面，往前追溯最近一个交易日的股价数据，如果没有的话，对应的股票应该是处于停牌阶段，
      也许对应资产重组等事件，很可能对应 “异象”，应该予以剔除
3.  因子构建的 Python 代码

        import QUANTAXIS as QA
        import pandas as pd
        from functools import partial

        # 全市场股票代码
        code_list_1 = QA.QA_fetch_stock_list().index.tolist()

        # 上市时间过滤
        # 1. 这里用了一点小技巧，通过公司财报发布数量直接进行过滤
        # 2. 公司财报真实数量按照所有财报中最大值可以确定
        # 3. 取 2015/01/01 到 2019/09/26 截止的财务数据
        # 4. 财报实际发布日期与对应的会计日期不一致, 一个简单的处理方式,
        # 对所有的财报日期进行一个 shift, 这样, 基本可以保证对应的截面
        # 财报都已经公布
        df_finance = QA.QA_fetch_financial_report_adv(
            code=code_list_1, start="2015-12-30", end="2020-09-26").data
        max_report_num = df_finance.groupby(level=1).apply(len).max()
        filter_se = df_finance.groupby(level=1).apply(len) == max_report_num
        code_list_2 = filter_se.loc[filter_se].index.tolist()

        # 过滤后每股净资产
        se_value = df_finance.loc[(slice(None), code_list_2), "netAssetsPerShare"]

        # 股价获取
        dates_list = se_value.index.levels[0].unique().map(
            str).str.slice(0, 10).tolist()
        df_price = pd.DataFrame()
        drop_codes = set()
        for report_date in dates_list:
            real_trade_date = QA.QA_util_get_real_date(report_date)
            df_local = QA.QA_fetch_stock_day_adv(
                code=code_list_2, start=real_trade_date, end=real_trade_date).data
            df_price = df_price.append(df_local)
            if set(code_list_2).difference(set(df_local.index.remove_unused_levels().levels[1].unique())):
                drop_codes = drop_codes.union(set(code_list_2).difference(
                    set(df_local.index.remove_unused_levels().levels[1].unique())))
        code_list_3 = sorted(list(set(code_list_2).difference(drop_codes)))

        df_tmp_1 = df_price.loc[(slice(None), code_list_3), "close"].unstack(level=1)
        df_tmp_2 = se_value.loc[(slice(None), code_list_3)].unstack(level=1)
        df_tmp_1.index = df_tmp_2.index

        # 市净率，注意，财务数据需要进行 shift，股价则不用
        factor_bm = (df_tmp_2.shift(1) / df_tmp_1).dropna()

<a id="org3b361c2"></a>

### 3. 因子排序 (分位处理)

在得到所有股票的 BM 数据后，按照 BM 取值高低，对所有股票进行分组，一般的做法是分为 10 组，然后做多
分组最高的股票组合，做空分组最低的股票组合。

    # 按照对应的 report_date 进行分位处理
    bm_quantiles = factor_bm.apply(partial(pd.qcut, q=10, labels=False), axis=1)

<a id="org1626995"></a>

### 4. 对组合进行加权处理 (这里按照市值进行加权)

    # 获取市值，方便进行加权处理，注意： QA 默认在市值计算中都进行了前复权处理，复权不影响市值计算
    df_market = pd.DataFrame()
    for report_date in dates_list[1:]:  # 注意: 因子计算完毕, 第 1 期因子起始时间已经从 2015-3-31 开始
        real_trade_date = QA.QA_util_get_real_date(report_date)
        df_market = df_market.append(
            QA.QAAnalysis_block(
                code=code_list_3, start=real_trade_date, end=real_trade_date
            )
            .market_value["mv"]
            .unstack(level=1)
        )

    # 索引重新设置
    df_market.index = bm_quantiles.index
    df_weights = df_market.apply(lambda x: x / x.sum(), axis=1)

    # 计算收益率, 对应时间截面的收益率需要 shift(-1) 来进行对应
    pct = df_tmp_1.pct_change().shift(-1).dropna()

    # 合并所需的数据, 方便计算
    se_1 = bm_quantiles.loc[pct.index].stack()
    se_2 = df_weights.loc[pct.index].stack()
    se_3 = pct.stack()
    df = pd.concat(
        [se_1.rename("quantiles"), se_2.rename("weights"), se_3.rename("pct_change")],
        axis=1,
    )
    df["weighted_pct"] = df["weights"] * df["pct_change"]
    pct_quantiles = df.groupby(level=0).apply(
        lambda x: x.groupby("quantiles").apply(lambda y: y.weighted_pct.sum())
    )

<a id="org125e86e"></a>

### 5. t-检验

对应的样本均值除以样本的标准差就是 t-检验的 t 值，可以通过查表看对应的 p 值。下面的代码是利用 stats 对某个
分位进行 t 检验

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy import stats
    sns.set_palette('hls')
    # 验证是否服从t分布
    # returns degree of freedom,loc,scale
    dof,loc,scale=stats.t.fit(pct_quantiles[0])
    ks=stats.t.rvs(df=dof,loc=loc,scale=scale,size=pct.shape[0])
    # p>0.05, 不拒绝原假设，认为这两个样本的分布是相同的
    stats.ks_2samp(pct_quantiles[0],ks)

    # 画图
    plt.figure(figsize=(10,6))
    pct_quantiles[0].plot(kind='kde')
    t=stats.t(dof,loc,scale)
    x=np.linspace(t.ppf(.01),t.ppf(.99),100)
    plt.plot(x,t.pdf(x),c='orange')
    plt.xlabel('Age')
    plt.title('Age on T Dist')
    plt.legend()

<a id="org5371375"></a>

## 结语

行文还是仓促，代码的细节处理，比自己想象中要复杂许多，而最终得到的因子排序法结果，有点出乎医疗意料，所谓的
低 BM 的组合，收益率反而还挺不错，即便已经做了市值加权处理，也许哪里做的不对？欢迎评论！

不知道有没有小伙伴一起来复现书中案例，如果有兴趣加我微信吧，一起搞搞？

# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> APT 套利定价理论：不同资产收益率并非由单一市场因子决定，而是同时受到其他因子影响，由此，Ross 提出了套利定价理论
(Arbitrage Pricing Theroy, APT), 在 CAPM 基础上进一步延伸，构建了线性多因子定价模型 (简称多因子模型)，
\(E[R_i^e] = \beta_i^\prime\lambda\).
