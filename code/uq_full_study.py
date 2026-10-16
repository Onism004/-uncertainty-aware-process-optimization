# -*- coding: utf-8 -*-
"""
不确定性感知的硅酸铝纤维工艺质量预测与鲁棒参数优化 —— 全量研究脚本
覆盖提案12周计划的全部实验内容：
  Week 1-2  数据审计与问题形式化（数据字典、可复现划分、文献地图见论文）
  Week 3-4  模型阶梯与基线（Linear/Ridge/RF/GPR，5折CV+留出集）
  Week 5-6  GPR不确定性量化与校准（预测区间、覆盖率、校准曲线）
  Week 7-8  不确定性分析与灵敏度（2D热图、逐变量切片、区域分析）
  Week 9-10 鲁棒优化（λ扫描、σf预算准则、Pareto前沿、三工况对比）
  Week 11   稳健性与消融（多种子、样本量、噪声水平、主动学习预研、失败案例）
  Week 12   研究报告与答辩材料（见Word文档）

数据（Route B 仿真原型）：企业真实数据涉密不可外带；按真实系统工艺区间与
评分权重(0.4/0.3/0.2/0.1)构造仿真数据，围绕当前产线设定采样，真实最优工况位于
数据覆盖外推区。全部结果可复现（种子固定）。
作者：李明珊 (Mingshan Li)
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
BOUNDS = [(1600.0, 2000.0), (5.0, 15.0), (10.0, 60.0)]
RESULTS = {"week_plan": "12-week flagship plan, all experiments"}

# ================= 数据与真值评分面 =================

def true_quality_vec(X):
    T, v, t = X[:, 0], X[:, 1], X[:, 2]
    s1 = np.clip(100 - 0.004 * (T - 1850.0) ** 2, 0, 100)
    s2 = np.clip(100 - 0.8 * (v - 8.0) ** 2, 0, 100)
    s3 = np.clip(100 - 0.08 * (t - 35.0) ** 2, 0, 100)
    s4 = np.clip(100 - 0.5 * (v - 8.5) ** 2 - 0.05 * (t - 38.0) ** 2, 0, 100)
    return 0.4 * s1 + 0.3 * s2 + 0.2 * s3 + 0.1 * s4


def make_dataset(seed=42, n=300, noise=4.0):
    """围绕当前产线设定(T=1770, v=10.5, t=44)集中采样；真实最优(1850,8,35)在外推区"""
    rng = np.random.default_rng(seed)
    T = np.clip(rng.normal(1770, 25, n), 1600, 2000)
    v = np.clip(rng.normal(10.5, 0.8, n), 5, 15)
    t = np.clip(rng.normal(44, 3.5, n), 10, 60)
    X = np.column_stack([T, v, t])
    y = true_quality_vec(X) + rng.normal(0, noise, n)
    return X, y


def fit_gpr(X_train, y_train, restarts=10):
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    ym, ys = y_train.mean(), y_train.std()
    z = (y_train - ym) / ys
    kernel = C(1.0, (0.01, 1e4)) * RBF(length_scale=np.array([1.0, 1.0, 1.0]),
                                       length_scale_bounds=[(0.05, 50.0)] * 3) \
        + WhiteKernel(noise_level=0.3, noise_level_bounds=(1e-4, 30.0))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=restarts,
                                   random_state=42).fit(Xs, z)
    noise_y = float(ys * np.sqrt(gpr.kernel_.k2.noise_level))
    return {"scaler": scaler, "ym": float(ym), "ys": float(ys), "gpr": gpr, "noise_y": noise_y}


def gpr_predict(ctx, Xq):
    Xq_s = ctx["scaler"].transform(np.asarray(Xq, dtype=float))
    mz, sz = ctx["gpr"].predict(Xq_s, return_std=True)
    mu = ctx["ym"] + ctx["ys"] * mz
    sf = ctx["ys"] * sz
    sp = np.sqrt(sf ** 2 + ctx["noise_y"] ** 2)
    return mu, sf, sp


def ga_optimize(ctx, lmbda, pop_size=60, generations=80, seed=42, return_pop=False):
    """GA算子结构沿用实习项目：保留前1/3 + 70:30凸组合交叉 + 10%变异"""
    r = np.random.default_rng(seed)
    lo = np.array([b[0] for b in BOUNDS]); hi = np.array([b[1] for b in BOUNDS])
    pop = r.uniform(lo, hi, size=(pop_size, 3))
    for _ in range(generations):
        mu_q, sf_q, _ = gpr_predict(ctx, pop)
        fits = mu_q - lmbda * sf_q
        parents = pop[np.argsort(-fits)[: pop_size // 3]]
        kids = np.empty((parents.shape[0], 3))
        for i in range(parents.shape[0]):
            p1, p2 = parents[i], parents[(i + 1) % parents.shape[0]]
            kids[i] = 0.7 * p1 + 0.3 * p2
            if r.random() < 0.1:
                j = int(r.integers(0, 3))
                kids[i, j] += r.normal(0, 0.05 * (hi[j] - lo[j]))
        kids = np.clip(kids, lo, hi)
        pop = np.vstack([parents, kids])
    mu_q, sf_q, sp_q = gpr_predict(ctx, pop)
    fits = mu_q - lmbda * sf_q
    b = pop[int(np.argmax(fits))]
    mu_b, sf_b, sp_b = gpr_predict(ctx, b.reshape(1, -1))
    rec = {"T": round(float(b[0]), 1), "v": round(float(b[1]), 2), "t": round(float(b[2]), 1),
           "mu": round(float(mu_b[0]), 2), "sigma_f": round(float(sf_b[0]), 2),
           "sigma_pred": round(float(sp_b[0]), 2),
           "true_q": round(float(true_quality_vec(b.reshape(1, -1))[0]), 2)}
    if return_pop:
        return rec, pop, mu_q, sf_q
    return rec


def save_json():
    with open(f"{OUT}\\uq_full_results.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)


# ================= Week 3-4：模型阶梯（基线，5折CV） =================
print("=== Week 3-4: model ladder ===")
X, y = make_dataset(42, 300, 4.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def gpr_model(restarts=10):
    kernel = C(1.0, (0.01, 1e4)) * RBF(length_scale=np.array([1.0, 1.0, 1.0]),
                                       length_scale_bounds=[(0.05, 50.0)] * 3) \
        + WhiteKernel(noise_level=0.3, noise_level_bounds=(1e-4, 30.0))
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=restarts,
                                    normalize_y=False, random_state=42)


ladder = {"LinearRegression": LinearRegression(), "Ridge": Ridge(alpha=1.0),
          "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
          "GPR": gpr_model(restarts=5)}
cv_summary, test_summary = {}, {}
for name, model in ladder.items():
    cvr = cross_validate(model, X_train_s, y_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_summary[name] = {"rmse_mean": float(-cvr["test_rmse"].mean()), "rmse_std": float(cvr["test_rmse"].std()),
                        "mae_mean": float(-cvr["test_mae"].mean()), "mae_std": float(cvr["test_mae"].std()),
                        "r2_mean": float(cvr["test_r2"].mean()), "r2_std": float(cvr["test_r2"].std())}
    model.fit(X_train_s, y_train)
    yp = model.predict(X_test_s)
    test_summary[name] = {"rmse": float(np.sqrt(mean_squared_error(y_test, yp))),
                          "mae": float(mean_absolute_error(y_test, yp)),
                          "r2": float(r2_score(y_test, yp))}
    print(f"  {name}: CV RMSE={cv_summary[name]['rmse_mean']:.2f}+-{cv_summary[name]['rmse_std']:.2f} "
          f"R2={cv_summary[name]['r2_mean']:.3f} | Test RMSE={test_summary[name]['rmse']:.2f} R2={test_summary[name]['r2']:.3f}")
RESULTS["cv"], RESULTS["test"] = cv_summary, test_summary

# ================= Week 5-6：GPR + 校准 =================
print("=== Week 5-6: GPR & calibration ===")
ctx = fit_gpr(X_train, y_train, restarts=10)
mu_y, std_f, std_p = gpr_predict(ctx, X_test)
noise_y = ctx["noise_y"]
gm = {"rmse": float(np.sqrt(mean_squared_error(y_test, mu_y))),
      "mae": float(mean_absolute_error(y_test, mu_y)),
      "r2": float(r2_score(y_test, mu_y)),
      "noise_y": noise_y,
      "length_scales": [float(x) for x in np.atleast_1d(ctx["gpr"].kernel_.k1.k2.length_scale)],
      "amplitude_y": float(np.sqrt(float(ctx["gpr"].kernel_.k1.k1.constant_value)) * ctx["ys"]),
      "kernel_str": str(ctx["gpr"].kernel_)}
LEVELS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
ZCRIT = [0.674, 0.842, 1.036, 1.282, 1.440, 1.645, 1.960, 2.576]


def coverage(sv, zc):
    lo, hi = mu_y - zc * sv, mu_y + zc * sv
    return float(np.mean((y_test >= lo) & (y_test <= hi)))


cal_lat = [coverage(std_f, z) for z in ZCRIT]
cal_pre = [coverage(std_p, z) for z in ZCRIT]
gm["cov95_latent"], gm["cov95_predictive"] = cal_lat[6], cal_pre[6]
RESULTS["gpr"], RESULTS["calibration"] = gm, {"levels": LEVELS, "latent": cal_lat, "predictive": cal_pre}
print(f"  kernel={gm['kernel_str']}")
print(f"  noise_est={noise_y:.2f} (true 4.0) | cov95 latent={cal_lat[6]:.1%} predictive={cal_pre[6]:.1%}")

# 失败案例分析（Week 11）：95%认知区间未覆盖的测试点
lo95, hi95 = mu_y - 1.96 * std_f, mu_y + 1.96 * std_f
miss_mask = (y_test < lo95) | (y_test > hi95)
miss_pts = X_test[miss_mask]
RESULTS["failure_cases"] = {"n_missed": int(miss_mask.sum()), "n_test": len(y_test),
                             "missed_points": miss_pts.tolist(),
                             "missed_T_mean": float(miss_pts[:, 0].mean()) if len(miss_pts) else None}
print(f"  latent-95% interval misses: {int(miss_mask.sum())}/{len(y_test)}, "
      f"missed mean T={RESULTS['failure_cases']['missed_T_mean']}")
save_json()

# ================= Week 7-8：灵敏度与不确定性结构 =================
print("=== Week 7-8: sensitivity ===")
CURRENT = np.array([1770.0, 10.5, 44.0])
TARGET = np.array([1850.0, 8.0, 35.0])
VAR_NAMES = ["熔融温度 T（℃）", "拉丝速度 v（m/s）", "固化时间 t（min）"]

# 2D热图：(T,v) @ t=44；(T,t) @ v=10.5；(v,t) @ T=1770
pairs = [(0, 1, 44.0, "T", "v"), (0, 2, 10.5, "T", "t"), (1, 2, 1770.0, "v", "t")]
pair_axes = {0: (1600, 2000), 1: (5, 15), 2: (10, 60)}
pair_units = {0: "℃", 1: "m/s", 2: "min"}
fig_names = ["fig5_sensitivity_2D_Tv.png", "fig6_sensitivity_2D_Tt.png", "fig7_sensitivity_2D_vt.png"]
heat_data = {}
for (ia, ib, fix, na, nb), fname in zip(pairs, fig_names):
    ga = np.linspace(*pair_axes[ia], 60)
    gb = np.linspace(*pair_axes[ib], 60)
    A, B = np.meshgrid(ga, gb)
    Xq = np.zeros((A.size, 3))
    Xq[:, ia], Xq[:, ib] = A.ravel(), B.ravel()
    ic = [k for k in range(3) if k not in (ia, ib)][0]
    Xq[:, ic] = fix
    mu_q, sf_q, _ = gpr_predict(ctx, Xq)
    heat_data[f"{na}{nb}"] = {"mu_max": float(mu_q.max()), "sf_max": float(sf_q.max()),
                              "sf_argmax": [float(Xq[int(np.argmax(sf_q)), 0]),
                                            float(Xq[int(np.argmax(sf_q)), 1]),
                                            float(Xq[int(np.argmax(sf_q)), 2])]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    c1 = axes[0].contourf(A, B, mu_q.reshape(A.shape), levels=20, cmap="viridis")
    axes[0].set_title(f"预测均值 μ（固定第三变量={fix}）")
    axes[0].set_xlabel(VAR_NAMES[ia]); axes[0].set_ylabel(VAR_NAMES[ib])
    fig.colorbar(c1, ax=axes[0])
    c2 = axes[1].contourf(A, B, sf_q.reshape(A.shape), levels=20, cmap="magma")
    axes[1].plot(CURRENT[ia], CURRENT[ib], "g^", ms=10, label="当前设定")
    axes[1].plot(TARGET[ia], TARGET[ib], "r*", ms=12, label="真实最优")
    axes[1].set_title("认知不确定性 σf（不确定性热图）")
    axes[1].set_xlabel(VAR_NAMES[ia]); axes[1].set_ylabel(VAR_NAMES[ib])
    axes[1].legend(loc="upper left", fontsize=9)
    fig.colorbar(c2, ax=axes[1])
    fig.tight_layout(); fig.savefig(f"{OUT}\\{fname}", dpi=150); plt.close(fig)
    print(f"  heatmap {na}x{nb}: mu_max={mu_q.max():.1f} sf_max={sf_q.max():.2f} at {heat_data[f'{na}{nb}']['sf_argmax']}")
RESULTS["heatmaps"] = heat_data

# 逐变量1D切片
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
slice_stats = {}
for k in range(3):
    xs = np.linspace(*pair_axes[k], 120)
    pts = np.tile(CURRENT, (120, 1)).astype(float)
    pts[:, k] = xs
    mu_l, sf_l, sp_l = gpr_predict(ctx, pts)
    axes[k].fill_between(xs, mu_l - 1.96 * sp_l, mu_l + 1.96 * sp_l, alpha=0.18, color="#4a7ebb", label="95%预测区间")
    axes[k].fill_between(xs, mu_l - 1.96 * sf_l, mu_l + 1.96 * sf_l, alpha=0.35, color="#1a3c6e", label="95%认知带")
    axes[k].plot(xs, mu_l, color="#0f2a50", lw=2, label="均值")
    axes[k].axvline(CURRENT[k], color="green", ls=":", label="当前设定")
    axes[k].set_xlabel(VAR_NAMES[k]); axes[k].grid(alpha=0.3)
    if k == 0:
        axes[k].set_ylabel("预测质量得分"); axes[k].legend(fontsize=8)
    slice_stats[VAR_NAMES[k]] = {"sf_min": float(sf_l.min()), "sf_max": float(sf_l.max()),
                                  "sf_at_current": float(gpr_predict(ctx, CURRENT.reshape(1, -1))[1][0])}
    print(f"  slice {VAR_NAMES[k]}: sf_min={sf_l.min():.2f} sf_max={sf_l.max():.2f}")
axes[0].set_title("过当前设定点的逐变量切片：均值与不确定性")
fig.tight_layout(); fig.savefig(f"{OUT}\\fig8_variable_slices.png", dpi=150); plt.close(fig)
RESULTS["slices"] = slice_stats

# 区域分析：σf vs 标准化距离
dist = np.linalg.norm(ctx["scaler"].transform(X_test), axis=1)
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.scatter(dist, std_f, s=22, alpha=0.65, color="#1a3c6e", edgecolor="none")
bins = np.quantile(dist, [0, 0.25, 0.5, 0.75, 1.0])
bx, by = [], []
for i in range(4):
    m = (dist >= bins[i]) & (dist <= bins[i + 1])
    bx.append(float(dist[m].mean())); by.append(float(std_f[m].mean()))
ax.plot(bx, by, "r--o", lw=2, label="分位均值")
ax.set_xlabel("测试点到数据中心的标准化距离"); ax.set_ylabel("认知不确定性 σf（分）")
ax.set_title("区域分析（RQ3）：离数据越远，模型越不确定")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\fig9_uncertainty_vs_distance.png", dpi=150); plt.close(fig)
RESULTS["region"] = {"bin_dist": bx, "bin_sigma_f": by}
print(f"  region: bins {['%.2f' % b for b in by]}")
save_json()

# ================= Week 9-10：鲁棒优化（λ扫描/预算准则/Pareto） =================
print("=== Week 9-10: robust optimization ===")
lambdas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
sweep = []
for L in lambdas:
    rec = ga_optimize(ctx, L)
    sweep.append(dict(lmbda=L, **rec))
    print(f"  lam={L}: ({rec['T']},{rec['v']},{rec['t']}) mu={rec['mu']} sf={rec['sigma_f']} true_q={rec['true_q']}")
mu_c, sf_c, sp_c = gpr_predict(ctx, CURRENT.reshape(1, -1))
current_setting = {"T": 1770.0, "v": 10.5, "t": 44.0, "mu": round(float(mu_c[0]), 2),
                   "sigma_f": round(float(sf_c[0]), 2), "sigma_pred": round(float(sp_c[0]), 2),
                   "true_q": round(float(true_quality_vec(CURRENT.reshape(1, -1))[0]), 2)}
RESULTS["ga"] = {"current_setting": current_setting, "lambda_sweep": sweep}

# Pareto前沿（λ=0种群 + 各λ推荐点）
rec0, pop0, mu_pop, sf_pop = ga_optimize(ctx, 0.0, return_pop=True)
order = np.argsort(-mu_pop)
pareto_x, pareto_mu, pareto_sf = [], [], []
best_sf = np.inf
for i in order:
    if sf_pop[i] < best_sf:
        pareto_x.append(pop0[i]); pareto_mu.append(mu_pop[i]); pareto_sf.append(sf_pop[i])
        best_sf = sf_pop[i]
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.scatter(mu_pop, sf_pop, s=14, alpha=0.45, color="#8fb8de", label="GA种群（λ=0，按μ寻优）")
ax.plot(pareto_mu, pareto_sf, "g-", lw=1.6, label="经验Pareto前沿（μ最优的不确定性下界）")
for s in sweep:
    ax.scatter(s["mu"], s["sigma_f"], marker="*", s=140,
               color=("#cc3333" if s["lmbda"] == 0 else "#e8a33d" if s["lmbda"] < 10 else "#1a7a1a"),
               zorder=5)
    ax.annotate(f"λ={s['lmbda']}", (s["mu"], s["sigma_f"]), textcoords="offset points",
               xytext=(6, 6), fontsize=9)
ax.scatter(current_setting["mu"], current_setting["sigma_f"], marker="^", s=110, color="green", zorder=5)
ax.annotate("当前设定", (current_setting["mu"], current_setting["sigma_f"]),
            textcoords="offset points", xytext=(8, -14), fontsize=9, color="green")
ax.axvline(100, color="grey", ls="--", lw=1, alpha=0.7)
ax.annotate("满分线 μ=100（外推幻觉区在右侧）", (100.4, ax.get_ylim()[0] + 0.2), fontsize=8, color="grey")
ax.set_xlabel("预测质量 μ（分）"); ax.set_ylabel("认知不确定性 σf（分）")
ax.set_title("质量-可靠性空间：Pareto前沿与λ推荐的迁移路径")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\fig10_pareto_front.png", dpi=150); plt.close(fig)
RESULTS["pareto"] = {"n_points": len(pareto_mu),
                     "mu_range": [round(float(min(pareto_mu)), 2), round(float(max(pareto_mu)), 2)],
                     "sf_range": [round(float(min(pareto_sf)), 2), round(float(max(pareto_sf)), 2)]}

# σf预算准则（λ选择的形式化）：密集λ扫描 -> 给定σf预算b，选σf<=b中μ最大的推荐
dense_lams = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
dense = [dict(lmbda=L, **ga_optimize(ctx, L, generations=60)) for L in dense_lams]
budget_table = []
for b in [3.5, 4.0, 4.5, 5.0, 6.0]:
    cands = [d for d in dense if d["sigma_f"] <= b]
    if not cands:
        continue
    best = max(cands, key=lambda d: d["mu"])
    budget_table.append({"budget": b, "chosen_lambda": best["lmbda"], "T": best["T"], "v": best["v"],
                        "t": best["t"], "mu": best["mu"], "sigma_f": best["sigma_f"],
                        "true_q": best["true_q"]})
    print(f"  budget sigma_f<={b}: lambda*={best['lmbda']} -> true_q={best['true_q']}")
RESULTS["dense_lambda"] = dense
RESULTS["budget_table"] = budget_table
save_json()

# ================= Week 11：稳健性与消融 =================
print("=== Week 11: ablations ===")
# F1 多种子稳健性
seed_res = []
for sd in [42, 7, 123, 2024, 555]:
    Xs_, ys_ = make_dataset(sd, 300, 4.0)
    Xtr, Xte, ytr, yte = train_test_split(Xs_, ys_, test_size=0.2, random_state=42)
    c2 = fit_gpr(Xtr, ytr, restarts=5)
    row = {"seed": sd}
    for L in [0.0, 5.0, 10.0]:
        r = ga_optimize(c2, L, generations=40)
        row[f"lam{int(L)}"] = {"mu": r["mu"], "sigma_f": r["sigma_f"], "true_q": r["true_q"],
                               "T": r["T"], "v": r["v"], "t": r["t"],
                               "hallucination": bool(r["mu"] > 100)}
    seed_res.append(row)
    print(f"  seed={sd}: lam0 true_q={row['lam0']['true_q']} (mu={row['lam0']['mu']}, "
          f"hallu={row['lam0']['hallucination']}) lam10 true_q={row['lam10']['true_q']}")
lam0_tq = np.array([r["lam0"]["true_q"] for r in seed_res])
lam10_tq = np.array([r["lam10"]["true_q"] for r in seed_res])
hall_count = sum(1 for r in seed_res if r["lam0"]["hallucination"])
RESULTS["seed_robustness"] = {"runs": seed_res, "n_hallucination": hall_count, "n_seeds": len(seed_res),
                              "lam0_true_q_mean": float(lam0_tq.mean()), "lam0_true_q_std": float(lam0_tq.std()),
                              "lam10_true_q_mean": float(lam10_tq.mean()), "lam10_true_q_std": float(lam10_tq.std())}

fig, ax = plt.subplots(figsize=(8, 5.2))
sds = [r["seed"] for r in seed_res]
ax.plot(sds, [r["lam0"]["true_q"] for r in seed_res], "ro-", label="λ=0（点预测最优）真实质量")
ax.plot(sds, [r["lam5"]["true_q"] for r in seed_res], "ms--", label="λ=5 真实质量")
ax.plot(sds, [r["lam10"]["true_q"] for r in seed_res], "g^-", label="λ=10（稳健）真实质量")
ax.axhline(current_setting["true_q"], color="grey", ls=":", label="当前设定真实质量")
ax.set_xlabel("数据生成种子"); ax.set_ylabel("推荐工况的真实质量（分）")
ax.set_title(f"多种子稳健性（n={len(seed_res)}）：幻觉与稳健结论是否可重复")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\fig11_seed_robustness.png", dpi=150); plt.close(fig)

# F2 样本量消融
n_res = []
for nn in [150, 300, 600]:
    Xs_, ys_ = make_dataset(42, nn, 4.0)
    Xtr, Xte, ytr, yte = train_test_split(Xs_, ys_, test_size=0.2, random_state=42)
    c3 = fit_gpr(Xtr, ytr, restarts=5)
    mu_t, sf_t, _ = gpr_predict(c3, TARGET.reshape(1, -1))
    r0 = ga_optimize(c3, 0.0, generations=40)
    n_res.append({"n": nn, "sigma_f_target": round(float(sf_t[0]), 2),
                  "lam0_mu": r0["mu"], "lam0_true_q": r0["true_q"],
                  "lam0_hallucination": bool(r0["mu"] > 100)})
    print(f"  n={nn}: sf@target={sf_t[0]:.2f}, lam0 true_q={r0['true_q']} (mu={r0['mu']})")
RESULTS["sample_size_ablation"] = n_res

fig, ax = plt.subplots(figsize=(8, 5.2))
ns = [r["n"] for r in n_res]
ax.plot(ns, [r["sigma_f_target"] for r in n_res], "o-", color="#1a3c6e", label="目标工况(1850,8,35)的σf")
ax.set_xlabel("训练样本数 n"); ax.set_ylabel("认知不确定性 σf（分）", color="#1a3c6e")
ax2 = ax.twinx()
ax2.plot(ns, [r["lam0_true_q"] for r in n_res], "s--", color="#cc3333", label="λ=0推荐的真实质量")
ax2.set_ylabel("λ=0推荐的真实质量（分）", color="#cc3333")
ax.set_title("样本量消融：数据越多，外推幻觉越小")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\fig12_sample_size.png", dpi=150); plt.close(fig)

# F3 噪声水平消融
noise_res = []
for nz in [2.0, 4.0, 8.0]:
    Xs_, ys_ = make_dataset(42, 300, nz)
    Xtr, Xte, ytr, yte = train_test_split(Xs_, ys_, test_size=0.2, random_state=42)
    c4 = fit_gpr(Xtr, ytr, restarts=5)
    mu_y2, sf_y2, sp_y2 = gpr_predict(c4, Xte)
    cov = float(np.mean((yte >= mu_y2 - 1.96 * sp_y2) & (yte <= mu_y2 + 1.96 * sp_y2)))
    noise_res.append({"noise_true": nz, "noise_est": round(c4["noise_y"], 2), "cov95_predictive": cov})
    print(f"  noise={nz}: est={c4['noise_y']:.2f}, cov95={cov:.1%}")
RESULTS["noise_ablation"] = noise_res

# F4 主动学习预研：在目标工况区补采m个样本
al_res = []
X_base, y_base = make_dataset(42, 300, 4.0)
Xtr0, Xte0, ytr0, yte0 = train_test_split(X_base, y_base, test_size=0.2, random_state=42)
for m in [0, 10, 20, 40]:
    if m == 0:
        Xtr_m, ytr_m = Xtr0, ytr0
    else:
        rg = np.random.default_rng(1000 + m)
        Tm = np.clip(rg.normal(1820, 15, m), 1600, 2000)
        vm = np.clip(rg.normal(8.7, 0.8, m), 5, 15)
        tm = np.clip(rg.normal(35, 3, m), 10, 60)
        Xm = np.column_stack([Tm, vm, tm])
        ym = true_quality_vec(Xm) + rg.normal(0, 4.0, m)
        Xtr_m = np.vstack([Xtr0, Xm]); ytr_m = np.concatenate([ytr0, ym])
    c5 = fit_gpr(Xtr_m, ytr_m, restarts=5)
    sf_tgt = float(gpr_predict(c5, TARGET.reshape(1, -1))[1][0])
    _, sf_corner, _ = gpr_predict(c5, np.array([[1835.4, 5.0, 10.0]]))
    r0 = ga_optimize(c5, 0.0, generations=40)
    al_res.append({"m_added": m, "sigma_f_target": round(sf_tgt, 2),
                   "sigma_f_corner": round(float(sf_corner[0]), 2),
                   "lam0_mu": r0["mu"], "lam0_true_q": r0["true_q"],
                   "lam0_hallucination": bool(r0["mu"] > 100)})
    print(f"  m={m}: sf@target={sf_tgt:.2f}, sf@corner={float(sf_corner[0]):.2f}, lam0 true_q={r0['true_q']}")
RESULTS["active_learning"] = al_res

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ms = [r["m_added"] for r in al_res]
ax.plot(ms, [r["sigma_f_target"] for r in al_res], "o-", color="#1a3c6e", label="目标工况σf")
ax.plot(ms, [r["sigma_f_corner"] for r in al_res], "s--", color="#cc7a29", label="幻觉角落σf")
ax.set_xlabel("在目标工况区补充采样的样本数 m"); ax.set_ylabel("认知不确定性 σf（分）")
ax2 = ax.twinx()
ax2.plot(ms, [r["lam0_true_q"] for r in al_res], "^:", color="#cc3333", label="λ=0推荐的真实质量")
ax2.set_ylabel("λ=0推荐的真实质量（分）", color="#cc3333")
ax.set_title("主动学习预研：目标区补数据如何消除外推幻觉")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\fig13_active_learning.png", dpi=150); plt.close(fig)

RESULTS["data"] = {"n": 300, "noise_sigma_true": 4.0, "route": "B(仿真原型)",
                   "scoring_weights": [0.4, 0.3, 0.2, 0.1],
                   "sampling_center": [1770, 10.5, 44], "true_optimum": [1850, 8.0, 35]}
save_json()
print("ALL DONE (12-week experiments complete)")
