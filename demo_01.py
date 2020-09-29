from functools import partial

import pandas as pd

import QUANTAXIS as QA

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
    code=code_list_1, start="2015-12-30", end="2020-09-26"
).data
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
        code=code_list_2, start=real_trade_date, end=real_trade_date
    ).data
    df_price = df_price.append(df_local)
    if set(code_list_2).difference(
        set(df_local.index.remove_unused_levels().levels[1].unique())
    ):
        drop_codes = drop_codes.union(
            set(code_list_2).difference(
                set(df_local.index.remove_unused_levels().levels[1].unique())
            )
        )
code_list_3 = sorted(list(set(code_list_2).difference(drop_codes)))

df_tmp_1 = df_price.loc[(slice(None), code_list_3), "close"].unstack(level=1)
df_tmp_2 = se_value.loc[(slice(None), code_list_3)].unstack(level=1)
df_tmp_1.index = df_tmp_2.index

# 市净率
factor_bm = (df_tmp_2.shift(1) / df_tmp_1).dropna()

# 按照对应的 report_date 进行分位处理
bm_quantiles = factor_bm.apply(partial(pd.qcut, q=10, labels=False), axis=1)

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
    [se_1.rename("quantiles"), se_2.rename(
        "weights"), se_3.rename("pct_change")],
    axis=1,
)
df["weighted_pct"] = df["weights"] * df["pct_change"]
pct_quantiles = df.groupby(level=0).apply(
    lambda x: x.groupby("quantiles").apply(lambda y: y.weighted_pct.sum())
)
