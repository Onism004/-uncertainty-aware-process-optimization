# -*- coding: utf-8 -*-
"""
不确定性感知的硅酸铝纤维工艺质量预测与鲁棒参数优化 (v2)
Uncertainty-Aware Quality Prediction and Robust Process Optimization
=================================================================
研究链（对应提案 V1.0）：Data -> Prediction -> Uncertainty -> Explanation -> Optimization -> Decision

v2 修订说明（相对 v1 的两个方法学修正）：
  1. λ-鲁棒目标使用【认知不确定性 σf】而非含观测噪声的 σ_pred：
     偶然噪声(aleatoric)不随决策变量 x 改变，进入目标函数相当于常数惩罚，不改变 argmax；
     只有认知不确定性(epistemic)才对"该不该去那里"有决策意义 —— 这是白板答辩的关键区分。
  2. 采样更集中（模拟真实工业数据特征：数据密集于当前工况附近，最优工况区数据稀缺），
     使"不确定性改变决策"(RQ4) 的考察具有实际意义。

数据说明（诚实声明）：
  本脚本使用【仿真数据】（提案 Route B 原型验证模式）。企业真实数据涉密不可外带，
  因此按真实系统的工艺区间与评分结构构造仿真数据：
    - 特征：熔融温度 T (1600-2000 ℃)、拉丝速度 v (5-15 m/s)、固化时间 t (10-60 min)
    - 目标：质量综合得分 y，权重沿用真实系统的 0.4/0.3/0.2/0.1 加权评分结构
    - 采样：围绕"当前产线设定区" (T~1770, v~10.5, t~44) 集中采样，
      真实最优工况 (1850, 8.0, 35.0) 位于数据覆盖边缘（外推区）
作者：李明珊 (Mingshan Li)   日期：2026-10-02   许可：仅用于本人研究与申请材料
"""

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\桌面\UQ研究项目"
RESULTS = {}

# ============================================================
# 1. 仿真数据生成
# ============================================================
rng = np.random.default_rng(42)
n_samples = 300

T = np.clip(rng.normal(1770, 25, n_samples), 1600, 2000)   # 熔融温度 ℃（当前设定1770附近）
v = np.clip(rng.normal(10.5, 0.8, n_samples), 5, 15)       # 拉丝速度 m/s（当前设定10.5附近）
t = np.clip(rng.normal(44, 3.5, n_samples), 10, 60)        # 固化时间 min（当前设定44附近）
X_raw = np.column_stack([T, v, t])

# 质量评分面：权重沿用真实系统加权综合评分结构 0.4/0.3/0.2/0.1
# 真实最优工况 (T=1850, v=8.0, t=35) 位于当前数据覆盖区之外（约3σ外推区）
s_thermal = np.clip(100 - 0.004 * (T - 1850) ** 2, 0, 100)
s_tensile = np.clip(100 - 0.8 * (v - 8.0) ** 2, 0, 100)
s_length = np.clip(100 - 0.08 * (t - 35.0) ** 2, 0, 100)
s_density = np.clip(100 - 0.5 * (v - 8.5) ** 2 - 0.05 * (t - 38) ** 2, 0, 100)
noise_sigma = 4.0
y = 0.4 * s_thermal + 0.3 * s_tensile + 0.2 * s_length + 0.1 * s_density \
    + rng.normal(0, noise_sigma, n_samples)


def true_quality(Tv, vv, tv):
    """真实（无噪声）质量评分面：仿真研究中作为事后验证的 oracle。
    实际生产中不可得——这正是必须用不确定性量化替代事后验证的原因。"""
    s1 = np.clip(100 - 0.004 * (Tv - 1850.0) ** 2, 0, 100)
    s2 = np.clip(100 - 0.8 * (vv - 8.0) ** 2, 0, 100)
    s3 = np.clip(100 - 0.08 * (tv - 35.0) ** 2, 0, 100)
    s4 = np.clip(100 - 0.5 * (vv - 8.5) ** 2 - 0.05 * (tv - 38.0) ** 2, 0, 100)
    return 0.4 * s1 + 0.3 * s2 + 0.2 * s3 + 0.1 * s4

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
y_mean, y_std = y_train.mean(), y_train.std()
z_train = (y_train - y_mean) / y_std

print(f"n_train={len(y_train)} n_test={len(y_test)} y_train_mean={y_mean:.2f} y_train_std={y_std:.2f}")

# ============================================================
# 2. 模型阶梯（提案5.1）：Linear -> Ridge -> RF -> GPR，5折CV + 留出集
# ============================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def gpr_model(restarts=10):
    kernel = C(1.0, (0.01, 1e4)) * RBF(length_scale=np.array([1.0, 1.0, 1.0]),
                                       length_scale_bounds=[(0.05, 50.0)] * 3) \
        + WhiteKernel(noise_level=0.3, noise_level_bounds=(1e-4, 30.0))
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=restarts,
                                     normalize_y=False, random_state=42)


models = {"LinearRegression": LinearRegression(), "Ridge": Ridge(alpha=1.0),
          "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
          "GPR": gpr_model(restarts=5)}

cv_summary = {}
for name, model in models.items():
    cvr = cross_validate(model, X_train_s, y_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_summary[name] = {"rmse_mean": float(-cvr["test_rmse"].mean()), "rmse_std": float(cvr["test_rmse"].std()),
                        "mae_mean": float(-cvr["test_mae"].mean()), "mae_std": float(cvr["test_mae"].std()),
                        "r2_mean": float(cvr["test_r2"].mean()), "r2_std": float(cvr["test_r2"].std())}
    print(f"[CV] {name}: RMSE={cv_summary[name]['rmse_mean']:.2f}+-{cv_summary[name]['rmse_std']:.2f} "
          f"MAE={cv_summary[name]['mae_mean']:.2f} R2={cv_summary[name]['r2_mean']:.3f}")
RESULTS["cv"] = cv_summary

test_scores = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    yp = model.predict(X_test_s)
    test_scores[name] = {"rmse": float(np.sqrt(mean_squared_error(y_test, yp))),
                         "mae": float(mean_absolute_error(y_test, yp)),
                         "r2": float(r2_score(y_test, yp))}
    print(f"[Test] {name}: RMSE={test_scores[name]['rmse']:.2f} MAE={test_scores[name]['mae']:.2f} "
          f"R2={test_scores[name]['r2']:.3f}")
RESULTS["test"] = test_scores

# ============================================================
# 3. GPR 不确定性量化与校准分析（提案5.2）
# ============================================================
gpr = gpr_model(restarts=10)
gpr.fit(X_train_s, z_train)
mu_z, std_z = gpr.predict(X_test_s, return_std=True)
mu_y = y_mean + y_std * mu_z
std_f = y_std * std_z                       # 认知不确定性 σf
noise_z = float(gpr.kernel_.k2.noise_level)
noise_y = float(y_std * np.sqrt(noise_z))   # 观测噪声估计（y空间）
std_pred = np.sqrt(std_f ** 2 + noise_y ** 2)  # 预测区间：认知+偶然
print(f"kernel={gpr.kernel_}")
print(f"noise_z={noise_z:.4f} noise_y={noise_y:.3f}")

gpr_metrics = {"rmse": float(np.sqrt(mean_squared_error(y_test, mu_y))),
               "mae": float(mean_absolute_error(y_test, mu_y)),
               "r2": float(r2_score(y_test, mu_y)), "noise_y": noise_y}
ls = np.exp(np.log(gpr.kernel_.k1.k2.length_scale)) if hasattr(gpr.kernel_.k1, "k2") else None
gpr_metrics["length_scales"] = [float(x) for x in np.atleast_1d(gpr.kernel_.k1.k2.length_scale)]
gpr_metrics["amplitude_y"] = float(np.sqrt(float(gpr.kernel_.k1.k1.constant_value)) * y_std)
RESULTS["gpr_test"] = gpr_metrics


def coverage(sv, zc):
    lo, hi = mu_y - zc * sv, mu_y + zc * sv
    return float(np.mean((y_test >= lo) & (y_test <= hi)))


LEVELS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
ZCRIT = [0.674, 0.842, 1.036, 1.282, 1.440, 1.645, 1.960, 2.576]
cal_lat = [coverage(std_f, z) for z in ZCRIT]
cal_pre = [coverage(std_pred, z) for z in ZCRIT]
gpr_metrics["cov95_latent"] = cal_lat[6]
gpr_metrics["cov95_predictive"] = cal_pre[6]
print(f"coverage95 latent={cal_lat[6]*100:.1f}% predictive={cal_pre[6]*100:.1f}%")
RESULTS["calibration"] = {"levels": LEVELS, "latent": cal_lat, "predictive": cal_pre}

# ============================================================
# 4. 优化：GA + λ 扫描（提案5.4；算子结构沿用实习项目）
#    目标：x* = argmax[ mu(x) - lam * sigma_f(x) ]  （认知不确定性）
# ============================================================
BOUNDS = [(1600.0, 2000.0), (5.0, 15.0), (10.0, 60.0)]


def gpr_predict_batch(Xq):
    Xq_s = scaler.transform(np.asarray(Xq, dtype=float))
    mz, sz = gpr.predict(Xq_s, return_std=True)
    mu = y_mean + y_std * mz
    sf = y_std * sz
    sp = np.sqrt(sf ** 2 + noise_y ** 2)
    return mu, sf, sp


def genetic_optimize(lmbda, pop_size=60, generations=80, seed=42):
    r = np.random.default_rng(seed)
    lo = np.array([b[0] for b in BOUNDS]); hi = np.array([b[1] for b in BOUNDS])
    pop = r.uniform(lo, hi, size=(pop_size, 3))
    for _ in range(generations):
        mu_q, sf_q, _ = gpr_predict_batch(pop)
        fits = mu_q - lmbda * sf_q          # 认知不确定性进目标；偶然噪声与x无关，进目标=常数偏移
        parents = pop[np.argsort(-fits)[: pop_size // 3]]
        kids = np.empty((parents.shape[0], 3))
        for i in range(parents.shape[0]):
            p1, p2 = parents[i], parents[(i + 1) % parents.shape[0]]
            kids[i] = 0.7 * p1 + 0.3 * p2   # 凸组合交叉（沿用实习GA）
            if r.random() < 0.1:            # 10%变异（沿用实习GA）
                j = int(r.integers(0, 3))
                kids[i, j] += r.normal(0, 0.05 * (hi[j] - lo[j]))
        kids = np.clip(kids, lo, hi)
        pop = np.vstack([parents, kids])
    mu_q, sf_q, sp_q = gpr_predict_batch(pop)
    fits = mu_q - lmbda * sf_q
    b = pop[int(np.argmax(fits))]
    mu_b, sf_b, sp_b = gpr_predict_batch(b.reshape(1, -1))
    return {"T": round(float(b[0]), 1), "v": round(float(b[1]), 2), "t": round(float(b[2]), 1),
            "mu": round(float(mu_b[0]), 2), "sigma_f": round(float(sf_b[0]), 2),
            "sigma_pred": round(float(sp_b[0]), 2),
            "true_q": round(float(true_quality(b[0], b[1], b[2])), 2)}


lambdas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
sweep = [dict(lmbda=L, **genetic_optimize(L)) for L in lambdas]
for s in sweep:
    print(f"lam={s['lmbda']}: T={s['T']} v={s['v']} t={s['t']} mu={s['mu']} sf={s['sigma_f']} sp={s['sigma_pred']}")

cur_pt = np.array([[1770.0, 10.5, 44.0]])
mu_c, sf_c, sp_c = gpr_predict_batch(cur_pt)
current_setting = {"T": 1770.0, "v": 10.5, "t": 44.0, "mu": round(float(mu_c[0]), 2),
                   "sigma_f": round(float(sf_c[0]), 2), "sigma_pred": round(float(sp_c[0]), 2),
                   "true_q": round(float(true_quality(1770.0, 10.5, 44.0)), 2)}
RESULTS["ga"] = {"current_setting": current_setting, "lambda_sweep": sweep}
print(f"current: mu={current_setting['mu']} sf={current_setting['sigma_f']}")

# 三工况点对比（提案5.5：Current / Prediction-Optimal / Robust）
pred_opt = next(s for s in sweep if s["lmbda"] == 0.0)
robust = next(s for s in sweep if s["lmbda"] == 1.0)
RESULTS["operating_points"] = {"current": current_setting, "prediction_optimal": pred_opt, "robust": robust}

# 认知不确定性的空间结构（RQ3）：四个代表点
probe = {"current_center(1770,10.5,44)": current_setting["sigma_f"]}
for nm, pt in [("target_true_opt(1850,8,35)", [1850, 8, 35]),
               ("midway(1810,9.25,39.5)", [1810, 9.25, 39.5]),
               ("far_corner(1780,6.0,20)", [1780, 6.0, 20])]:
    mu_p, sf_p, sp_p = gpr_predict_batch(np.array([pt], dtype=float))
    probe[nm] = round(float(sf_p[0]), 2)
RESULTS["sensitivity_points"] = probe
print("sigma_f probe:", probe)

# ============================================================
# 5. 图表
# ============================================================
# fig0 模型阶梯（CV RMSE）
fig, ax = plt.subplots(figsize=(8, 5))
names = list(cv_summary.keys())
mm = [cv_summary[n]["rmse_mean"] for n in names]
ss = [cv_summary[n]["rmse_std"] for n in names]
ax.bar(names, mm, yerr=ss, capsize=5, color=["#8fb8de", "#8fb8de", "#7aa6d9", "#1a3c6e"])
ax.set_ylabel("5折交叉验证 RMSE（分）")
ax.set_title("模型阶梯对比（提案5.1）：复杂度换来了什么")
ax.grid(axis="y", alpha=0.3)
for i, m in enumerate(mm):
    ax.text(i, m + 0.05, f"{m:.2f}", ha="center")
fig.tight_layout(); fig.savefig(f"{OUT}\\fig0_模型阶梯对比.png", dpi=150); plt.close(fig)

# fig1 预测区间
order = np.argsort(mu_y)
xs = np.arange(len(y_test))
fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(xs, mu_y[order] - 1.96 * std_pred[order], mu_y[order] + 1.96 * std_pred[order],
                alpha=0.22, color="#4a7ebb", label="95%预测区间（认知+观测噪声）")
ax.plot(xs, mu_y[order], "o-", ms=4, lw=1, color="#1a3c6e", label="GPR预测均值")
ax.plot(xs, y_test[order], "k^", ms=5, alpha=0.6, label="真实质量得分")
ax.set_xlabel("测试样本（按预测均值排序）"); ax.set_ylabel("质量得分")
ax.set_title(f"GPR预测区间 vs 真实值（95%经验覆盖率 = {cal_pre[6]*100:.1f}%）")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\fig1_预测区间.png", dpi=150); plt.close(fig)

# fig2 校准曲线
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0.4, 1.0], [0.4, 1.0], "k--", lw=1, label="理想校准")
ax.plot(LEVELS, cal_pre, "o-", color="#1a3c6e", label="预测区间（认知+观测噪声）")
ax.plot(LEVELS, cal_lat, "s--", color="#cc7a29", label="仅认知不确定性σf")
ax.set_xlabel("标称覆盖率"); ax.set_ylabel("经验覆盖率")
ax.set_title("校准分析（提案5.2）：区间可靠吗？")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\fig2_校准曲线.png", dpi=150); plt.close(fig)

# fig3 温度切片：认知带 vs 预测带
T_line = np.linspace(1600, 2000, 120)
slice_pts = np.column_stack([T_line, np.full_like(T_line, 10.5), np.full_like(T_line, 44.0)])
mu_l, sf_l, sp_l = gpr_predict_batch(slice_pts)
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.fill_between(T_line, mu_l - 1.96 * sp_l, mu_l + 1.96 * sp_l, alpha=0.18, color="#4a7ebb",
                label="95%预测区间（含观测噪声）")
ax.fill_between(T_line, mu_l - 1.96 * sf_l, mu_l + 1.96 * sf_l, alpha=0.35, color="#1a3c6e",
                label="95%认知不确定带（模型可学习部分）")
ax.plot(T_line, mu_l, color="#0f2a50", lw=2, label="GPR预测均值")
ax.axvspan(1740, 1800, alpha=0.12, color="green", label="当前数据集中区")
ax.axvline(1850, color="#cc3333", ls=":", lw=1.5, label="真实最优温度（数据稀疏区）")
ax.set_xlabel("熔融温度 T（℃）  [v=10.5 m/s, t=44 min]")
ax.set_ylabel("预测质量得分")
ax.set_title("灵敏度分析（RQ3）：不确定性在数据稀疏区显著升高")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\fig3_温度切片不确定性.png", dpi=150); plt.close(fig)

# fig4 λ权衡
fig, ax = plt.subplots(figsize=(9.5, 6))
lams = [s["lmbda"] for s in sweep]
mus = [s["mu"] for s in sweep]
sfs = [s["sigma_f"] for s in sweep]
ax.plot(lams, mus, "o-", color="#1a3c6e", label="推荐工况预测质量 μ")
ax.set_xlabel("风险偏好 λ（认知不确定性惩罚）")
ax.set_ylabel("预测质量得分 μ", color="#1a3c6e")
ax2 = ax.twinx()
ax2.plot(lams, sfs, "s--", color="#cc7a29", label="推荐工况认知不确定性 σf")
ax2.set_ylabel("认知不确定性 σf（分）", color="#cc7a29")
for s in sweep:
    ax.annotate(f"({s['T']:.0f}℃, {s['v']:.1f}, {s['t']:.0f}min)", (s["lmbda"], s["mu"]),
                textcoords="offset points", xytext=(8, -16), fontsize=9)
ax.set_title("λ扫描（RQ4/RQ5）：质量-可靠性权衡与推荐工况迁移")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\fig4_lambda权衡.png", dpi=150); plt.close(fig)

# ============================================================
# 6. 保存
# ============================================================
RESULTS["data"] = {"n": n_samples, "noise_sigma_true": noise_sigma, "y_mean": float(y_mean),
                   "y_std": float(y_std), "kernel_fitted": str(gpr.kernel_)}
with open(f"{OUT}\\uq_results.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2)
print("ALL DONE")
