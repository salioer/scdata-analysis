"""策略中心价值分析的统计功效（最小可检测差异 MDE）估算。

用法示例：
  python tools/power_calc.py rct --p 0.15 --n1 30000 --n2 30000
  python tools/power_calc.py rdd --p 0.17 --n1 200000 --n2 200000
  python tools/power_calc.py holdout --p 0.17 --treated 3990000 --holdout 210000
  python tools/power_calc.py rdd --sd 12.5 --n1 200000 --n2 200000   # 连续指标（人均笔数/金额）
  加 --periods 3 表示合并 3 个周期（样本量按周期数放大）。
  比例指标（交易用户率）用 --p；连续指标（人均交易笔数、人均金额，含 0）用 --sd 给出标准差，
  金额建议先按 P99 截尾再算标准差。
"""
import argparse
import math
from statistics import NormalDist

# 局部线性 RDD 的方差约为同样本量随机实验的 3 倍（三角核，常用经验值）
RDD_VARIANCE_INFLATION = 3.0


def z_factor(alpha=0.05, power=0.8):
    nd = NormalDist()
    return nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)


def mde_two_prop(p, n1, n2, alpha=0.05, power=0.8, inflation=1.0):
    """两组比例比较的 MDE（绝对差，单位为比例）。"""
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2) * inflation)
    return z_factor(alpha, power) * se


def mde_mean(sd, n1, n2, alpha=0.05, power=0.8, inflation=1.0):
    """两组均值比较的 MDE（绝对差，单位同指标）。"""
    se = sd * math.sqrt((1 / n1 + 1 / n2) * inflation)
    return z_factor(alpha, power) * se


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("design", choices=["rct", "rdd", "holdout"],
                        help="rct=随机桶内比较；rdd=选人阈值断点；holdout=可选的入选用户留出组")
    metric = parser.add_mutually_exclusive_group(required=True)
    metric.add_argument("--p", type=float, help="比例指标：基础转化率，如 0.15")
    metric.add_argument("--sd", type=float, help="连续指标：人均笔数或人均金额的标准差（含 0）")
    parser.add_argument("--n1", type=float, help="组 1 人数（rct/rdd：阈值上方或 F 组）")
    parser.add_argument("--n2", type=float, help="组 2 人数（rct/rdd：阈值下方或 H 组）")
    parser.add_argument("--treated", type=float, help="holdout：正常投放的入选人数")
    parser.add_argument("--holdout", type=float, help="holdout：留出不投放的入选人数")
    parser.add_argument("--periods", type=int, default=1, help="合并的周期数")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    args = parser.parse_args()

    if args.design == "holdout":
        if args.treated is None or args.holdout is None:
            parser.error("holdout 需要 --treated 和 --holdout")
        n1, n2 = args.treated, args.holdout
    else:
        if args.n1 is None or args.n2 is None:
            parser.error(f"{args.design} 需要 --n1 和 --n2")
        n1, n2 = args.n1, args.n2

    n1 *= args.periods
    n2 *= args.periods
    inflation = RDD_VARIANCE_INFLATION if args.design == "rdd" else 1.0
    if args.p is not None:
        mde = mde_two_prop(args.p, n1, n2, args.alpha, args.power, inflation)
        print(f"设计={args.design}  基础转化率={args.p:.2%}  n1={n1:,.0f}  n2={n2:,.0f}  周期数={args.periods}")
        print(f"MDE ≈ {mde * 100:.2f}pp（alpha={args.alpha}, power={args.power}）")
    else:
        mde = mde_mean(args.sd, n1, n2, args.alpha, args.power, inflation)
        print(f"设计={args.design}  标准差={args.sd:g}  n1={n1:,.0f}  n2={n2:,.0f}  周期数={args.periods}")
        print(f"MDE ≈ {mde:.4g}（与指标同单位；alpha={args.alpha}, power={args.power}）")


if __name__ == "__main__":
    main()
