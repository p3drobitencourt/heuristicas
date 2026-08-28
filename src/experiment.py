# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import sys

"""
experiment.py
=============
Executa o grid de parametros completo para Busca Tabu e PSO Binario
sobre as 9 instancias, 10 execucoes por combinacao.

Saidas:
  results/results_detailed.csv    -- uma linha por execucao
  results/results_summary.xlsx    -- aba Detalhado + aba Resumo
  results/plots/convergence_*.png -- graficos de convergencia (P06, P07, knapPI)

Uso:
  cd heuristica
  python -u src/experiment.py
"""

import itertools
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from parser import load_all_instances, Instance
from tabu import tabu_search, TabuResult
from pso import binary_pso, PSOResult

warnings.filterwarnings("ignore")

# Logger global -- configurado no __main__
log = logging.getLogger("experiment")

def _log(msg: str = "") -> None:
    """Wrapper conveniente para log.info."""
    log.info(msg)

# ---------------------------------------------------------------------------
# Configuração dos grids de parâmetros
# ---------------------------------------------------------------------------

TABU_GRID = {
    "tenure":       [5, 10, 20],
    "max_iter":     [200, 500, 1000],
    "no_improve":   [50, 100],
    "neighborhood": ["flip", "swap"],
}

PSO_GRID = {
    "n_particles": [20, 50, 100],
    "max_iter":    [100, 200, 500],
    "w":           [0.4, 0.7, 0.9],
    "c1_c2":       [(1.5, 1.5), (2.0, 2.0)],
    "v_max":       [4, 6],
}

N_RUNS   = 10     # execuções por combinação
BASE_SEED = 42    # semente base; cada execução usa BASE_SEED + run_idx

PLOT_INSTANCES = {"P06", "P07", "KNAPPI_1_100"}   # instâncias com gráfico de convergência

# ---------------------------------------------------------------------------
# Geração de combinações de parâmetros
# ---------------------------------------------------------------------------

def _tabu_combinations() -> list[dict]:
    keys   = list(TABU_GRID.keys())
    values = [TABU_GRID[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def _pso_combinations() -> list[dict]:
    combos = []
    for np_, mi, w, c1c2, vm in itertools.product(
        PSO_GRID["n_particles"],
        PSO_GRID["max_iter"],
        PSO_GRID["w"],
        PSO_GRID["c1_c2"],
        PSO_GRID["v_max"],
    ):
        combos.append({
            "n_particles": np_,
            "max_iter":    mi,
            "w":           w,
            "c1":          c1c2[0],
            "c2":          c1c2[1],
            "v_max":       vm,
        })
    return combos


# ---------------------------------------------------------------------------
# Execução de uma combinação (Tabu)
# ---------------------------------------------------------------------------

def _run_tabu_combo(instance: Instance, params: dict, run: int) -> dict:
    seed = BASE_SEED + run
    res  = tabu_search(
        instance,
        tenure=params["tenure"],
        max_iter=params["max_iter"],
        no_improve=params["no_improve"],
        neighborhood=params["neighborhood"],
        seed=seed,
    )
    gap = instance.gap(res.best_value)
    return {
        "instancia":           instance.name,
        "heuristica":          "BuscaTabu",
        "tenure":              params["tenure"],
        "max_iter_param":      params["max_iter"],
        "no_improve_param":    params["no_improve"],
        "neighborhood":        params["neighborhood"],
        "n_particles":         None,
        "w":                   None,
        "c1":                  None,
        "c2":                  None,
        "v_max":               None,
        "execucao":            run + 1,
        "semente":             seed,
        "melhor_valor":        res.best_value,
        "valor_otimo":         instance.optimal_value,
        "gap_pct":             gap,
        "tempo_s":             round(res.elapsed_seconds, 4),
        "iter_convergencia":   res.convergence_iter,
        "total_iteracoes":     res.total_iterations,
        "_history":            res.history,   # usado para gráficos (removido antes de salvar)
    }


# ---------------------------------------------------------------------------
# Execução de uma combinação (PSO)
# ---------------------------------------------------------------------------

def _run_pso_combo(instance: Instance, params: dict, run: int) -> dict:
    seed = BASE_SEED + run
    res  = binary_pso(
        instance,
        n_particles=params["n_particles"],
        max_iter=params["max_iter"],
        w=params["w"],
        c1=params["c1"],
        c2=params["c2"],
        v_max=params["v_max"],
        seed=seed,
    )
    gap = instance.gap(res.best_value)
    return {
        "instancia":           instance.name,
        "heuristica":          "PSO_Binario",
        "tenure":              None,
        "max_iter_param":      params["max_iter"],
        "no_improve_param":    None,
        "neighborhood":        None,
        "n_particles":         params["n_particles"],
        "w":                   params["w"],
        "c1":                  params["c1"],
        "c2":                  params["c2"],
        "v_max":               params["v_max"],
        "execucao":            run + 1,
        "semente":             seed,
        "melhor_valor":        res.best_value,
        "valor_otimo":         instance.optimal_value,
        "gap_pct":             gap,
        "tempo_s":             round(res.elapsed_seconds, 4),
        "iter_convergencia":   res.convergence_iter,
        "total_iteracoes":     res.total_iterations,
        "_history":            res.history,
    }


# ---------------------------------------------------------------------------
# Gráficos de convergência
# ---------------------------------------------------------------------------

PLOT_COLORS = {
    "BuscaTabu":  "#E07B54",
    "PSO_Binario": "#5B8DB8",
}

def _plot_convergence(
    inst_name: str,
    tabu_histories: list[list[int]],
    pso_histories: list[list[int]],
    optimal_value: Optional[int],
    out_dir: str,
) -> None:
    """
    Gera gráfico de convergência (média das 10 execuções) para uma instância.
    Usa a melhor configuração (maior média de melhor_valor nas 10 execuções).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")

    for label, histories, color in [
        ("Busca Tabu",  tabu_histories,  PLOT_COLORS["BuscaTabu"]),
        ("PSO Binário", pso_histories,   PLOT_COLORS["PSO_Binario"]),
    ]:
        if not histories:
            continue
        # Truncar / padronizar comprimento das histórias (pad com último valor)
        max_len = max(len(h) for h in histories)
        padded  = [h + [h[-1]] * (max_len - len(h)) for h in histories]
        arr  = np.array(padded, dtype=float)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        xs   = np.arange(max_len)

        ax.plot(xs, mean, label=label, color=color, linewidth=2)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)

    if optimal_value is not None:
        ax.axhline(
            optimal_value, color="#aaffaa", linewidth=1.5,
            linestyle="--", label=f"Ótimo ({optimal_value:,})"
        )

    ax.set_xlabel("Iteração / Geração", color="#cccccc", fontsize=11)
    ax.set_ylabel("Melhor Valor",       color="#cccccc", fontsize=11)
    ax.set_title(f"Convergência — {inst_name}", color="#ffffff", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1e2030", edgecolor="#444444", labelcolor="#dddddd", fontsize=10)
    ax.grid(True, color="#222233", linewidth=0.5)

    fname = os.path.join(out_dir, f"convergence_{inst_name.lower()}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    _log(f"    Grafico salvo: {fname}")


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def run_experiments(data_dir: str = "data", results_dir: str = "results") -> None:
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    _log("=" * 60)
    _log("Carregando instancias...")
    _log("=" * 60)
    instances = load_all_instances(data_dir=data_dir)

    tabu_combos = _tabu_combinations()
    pso_combos  = _pso_combinations()

    _log(f"\nCombinacoes Tabu : {len(tabu_combos)}")
    _log(f"Combinacoes PSO  : {len(pso_combos)}")
    _log(f"Execucoes por combo : {N_RUNS}")
    _log(f"Total estimado de execucoes : "
          f"{(len(tabu_combos) + len(pso_combos)) * N_RUNS * len(instances):,}\n")

    all_rows: list[dict] = []

    # Para gráficos: guardar histórias por instância × heurística
    # (usaremos apenas a melhor combinação de parâmetros após rodar tudo)
    conv_data: dict[str, dict[str, list[list[int]]]] = {}

    total_start = time.perf_counter()

    for inst in instances:
        _log("\n" + "="*60)
        _log(f"  INSTANCIA: {inst.name}  (n={inst.n}, C={inst.capacity}, otimo={inst.optimal_value})")
        _log("="*60)

        conv_data[inst.name] = {"BuscaTabu": [], "PSO_Binario": []}

        # === BUSCA TABU ===
        _log(f"  [Busca Tabu]  {len(tabu_combos)} combinacoes x {N_RUNS} execucoes ...")
        tabu_best_mean = -1
        tabu_best_histories: list[list[int]] = []

        for cidx, params in enumerate(tabu_combos, 1):
            combo_histories: list[list[int]] = []
            combo_vals: list[int] = []

            for run in range(N_RUNS):
                row = _run_tabu_combo(inst, params, run)
                hist = row.pop("_history")
                combo_histories.append(hist)
                combo_vals.append(row["melhor_valor"])
                all_rows.append(row)

            mean_val = np.mean(combo_vals)
            if mean_val > tabu_best_mean:
                tabu_best_mean = mean_val
                tabu_best_histories = combo_histories

            if cidx % 6 == 0 or cidx == len(tabu_combos):
                elapsed = time.perf_counter() - total_start
                _log(f"    Tabu combo {cidx}/{len(tabu_combos)} | {elapsed:.0f}s decorridos")

        conv_data[inst.name]["BuscaTabu"] = tabu_best_histories

        # === PSO BINARIO ===
        _log(f"  [PSO Binario]  {len(pso_combos)} combinacoes x {N_RUNS} execucoes ...")
        pso_best_mean = -1
        pso_best_histories: list[list[int]] = []

        for cidx, params in enumerate(pso_combos, 1):
            combo_histories: list[list[int]] = []
            combo_vals: list[int] = []

            for run in range(N_RUNS):
                row = _run_pso_combo(inst, params, run)
                hist = row.pop("_history")
                combo_histories.append(hist)
                combo_vals.append(row["melhor_valor"])
                all_rows.append(row)

            mean_val = np.mean(combo_vals)
            if mean_val > pso_best_mean:
                pso_best_mean = mean_val
                pso_best_histories = combo_histories

            if cidx % 18 == 0 or cidx == len(pso_combos):
                elapsed = time.perf_counter() - total_start
                _log(f"    PSO  combo {cidx}/{len(pso_combos)} | {elapsed:.0f}s decorridos")

        conv_data[inst.name]["PSO_Binario"] = pso_best_histories

    # ---------------------------------------------------------------------------
    # Salvar resultados detalhados
    # ---------------------------------------------------------------------------
    _log("\n" + "="*60)
    _log("Salvando resultados...")
    _log("="*60)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(results_dir, "results_detailed.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _log(f"  CSV salvo: {csv_path}  ({len(df):,} linhas)")

    # Aba de resumo: média + desvio por instância × heurística × parâmetros
    group_cols_tabu = ["instancia", "heuristica", "tenure", "max_iter_param",
                       "no_improve_param", "neighborhood"]
    group_cols_pso  = ["instancia", "heuristica", "n_particles", "max_iter_param",
                       "w", "c1", "c2", "v_max"]

    df_tabu = df[df["heuristica"] == "BuscaTabu"]
    df_pso  = df[df["heuristica"] == "PSO_Binario"]

    def _summarize(dff: pd.DataFrame, gcols: list[str]) -> pd.DataFrame:
        agg = dff.groupby(gcols, dropna=False).agg(
            media_melhor_valor=("melhor_valor", "mean"),
            std_melhor_valor=("melhor_valor", "std"),
            melhor_valor_absoluto=("melhor_valor", "max"),
            media_gap_pct=("gap_pct", "mean"),
            media_tempo_s=("tempo_s", "mean"),
            media_iter_convergencia=("iter_convergencia", "mean"),
            media_total_iteracoes=("total_iteracoes", "mean"),
            n_execucoes=("execucao", "count"),
        ).reset_index()
        agg["media_melhor_valor"]    = agg["media_melhor_valor"].round(2)
        agg["std_melhor_valor"]      = agg["std_melhor_valor"].round(2)
        agg["media_gap_pct"]         = agg["media_gap_pct"].round(4)
        agg["media_tempo_s"]         = agg["media_tempo_s"].round(4)
        agg["media_iter_convergencia"] = agg["media_iter_convergencia"].round(1)
        return agg

    summary_tabu = _summarize(df_tabu, group_cols_tabu)
    summary_pso  = _summarize(df_pso,  group_cols_pso)
    summary_all  = pd.concat([summary_tabu, summary_pso], ignore_index=True)

    xlsx_path = os.path.join(results_dir, "results_summary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer,          sheet_name="Detalhado", index=False)
        summary_all.to_excel(writer, sheet_name="Resumo",    index=False)

        # Formatar larguras de coluna
        for sheet_name in ["Detalhado", "Resumo"]:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)

    _log(f"  XLSX salvo: {xlsx_path}")

    # ---------------------------------------------------------------------------
    # Gráficos de convergência
    # ---------------------------------------------------------------------------
    _log("\nGerando graficos de convergencia...")
    for inst in instances:
        if inst.name.upper() not in PLOT_INSTANCES:
            continue
        _log(f"  {inst.name}")
        _plot_convergence(
            inst_name=inst.name,
            tabu_histories=conv_data[inst.name]["BuscaTabu"],
            pso_histories=conv_data[inst.name]["PSO_Binario"],
            optimal_value=inst.optimal_value,
            out_dir=plots_dir,
        )

    total_elapsed = time.perf_counter() - total_start
    _log("\n" + "="*60)
    _log(f"Experimentos concluidos em {total_elapsed/60:.1f} min")
    _log("="*60)

    # Tabela resumo
    _print_final_table(df)


def _print_final_table(df: pd.DataFrame) -> None:
    """Loga tabela comparativa final."""
    _log("\n=== Melhor resultado por instancia x heuristica ===")
    best = (
        df.groupby(["instancia", "heuristica"])
        .agg(melhor=("melhor_valor", "max"), media=("melhor_valor", "mean"),
             gap=("gap_pct", "mean"), tempo=("tempo_s", "mean"))
        .reset_index()
    )
    _log(best.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "run.log")

    # Configurar logging: arquivo + console (nao redireciona sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8", mode="w"),
            logging.StreamHandler(sys.stdout),   # mantem pipe original
        ],
    )

    run_experiments(
        data_dir=os.path.join(root, "data"),
        results_dir=results_dir,
    )
