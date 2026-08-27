"""
pso.py
======
Implementação do PSO Binário (Binary Particle Swarm Optimization)
para o Problema da Mochila 0/1.

Formulação: Kennedy & Eberhart (1997) para espaço binário.

  v_i(t+1) = w·v_i(t) + c1·r1·(pbest_i − x_i(t)) + c2·r2·(gbest_i − x_i(t))
  S(v)     = 1 / (1 + exp(−v))
  x_i      = 1  se  rand() < S(v_i),  senão 0

Reparo de infactibilidade:
  Se a solução gerada exceder a capacidade, remove itens com pior razão
  valor/peso até voltar a ser factível.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from parser import Instance, greedy_solution


# ---------------------------------------------------------------------------
# Estrutura de resultado
# ---------------------------------------------------------------------------

@dataclass
class PSOResult:
    best_value: int
    best_solution: np.ndarray
    history: list[int]          # melhor valor global por geração
    total_iterations: int
    convergence_iter: int       # geração em que o melhor global foi encontrado
    elapsed_seconds: float
    instance_name: str
    params: dict


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def _sigmoid(v: np.ndarray) -> np.ndarray:
    """Sigmoide vetorizada, numericamente estável."""
    return 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))


def _repair(x: np.ndarray, instance: Instance, rng: np.random.Generator) -> np.ndarray:
    """
    Repara uma solução infactível removendo itens com pior razão valor/peso
    (entre os itens selecionados) até a capacidade ser respeitada.
    """
    x = x.copy()
    peso = int(instance.weights @ x)
    if peso <= instance.capacity:
        return x

    # Razões valor/peso dos itens selecionados; ordenar do pior ao melhor
    ratios = instance.values / np.maximum(instance.weights, 1e-9)
    selected = np.where(x == 1)[0]
    # Ordenar do pior ratio para o melhor (remover os piores primeiro)
    order = selected[np.argsort(ratios[selected])]

    for i in order:
        x[i]  = 0
        peso -= instance.weights[i]
        if peso <= instance.capacity:
            break

    return x


def _evaluate_batch(X: np.ndarray, instance: Instance) -> np.ndarray:
    """Avalia um lote de soluções (matriz n_particles × n). Retorna array de valores."""
    return X @ instance.values


# ---------------------------------------------------------------------------
# PSO Binário
# ---------------------------------------------------------------------------

def binary_pso(
    instance: Instance,
    n_particles: int = 30,
    max_iter: int = 200,
    w: float = 0.7,
    c1: float = 2.0,
    c2: float = 2.0,
    v_max: float = 6.0,
    seed: Optional[int] = None,
) -> PSOResult:
    """
    Executa o PSO Binário sobre `instance`.

    Parâmetros
    ----------
    instance    : instância do problema
    n_particles : tamanho do enxame
    max_iter    : número máximo de gerações
    w           : peso de inércia
    c1          : coeficiente cognitivo
    c2          : coeficiente social
    v_max       : limite de velocidade (simetrico: [-v_max, +v_max])
    seed        : semente aleatória

    Retorna
    -------
    PSOResult com melhor solução global, histórico de convergência e métricas.
    """
    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()
    n   = instance.n

    # --- Inicialização das posições ---
    # Metade das partículas inicia com a solução gulosa + perturbação;
    # a outra metade com posições aleatórias reparadas.
    X = np.zeros((n_particles, n), dtype=int)
    greedy = greedy_solution(instance)
    for p in range(n_particles):
        if p < n_particles // 2:
            xp = greedy.copy()
            # Perturbar aleatoriamente ~10% dos bits
            flip_mask = rng.random(n) < 0.1
            xp[flip_mask] = 1 - xp[flip_mask]
            xp = _repair(xp, instance, rng)
        else:
            # Solução aleatória viável
            prob = rng.random(n)
            xp = (rng.random(n) < prob).astype(int)
            xp = _repair(xp, instance, rng)
        X[p] = xp

    # --- Inicialização das velocidades ---
    V = rng.uniform(-v_max / 2, v_max / 2, size=(n_particles, n))

    # --- Melhor pessoal (pbest) ---
    pbest_X   = X.copy()
    pbest_val = _evaluate_batch(X, instance).tolist()

    # --- Melhor global (gbest) ---
    gbest_idx = int(np.argmax(pbest_val))
    gbest_X   = pbest_X[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    history: list[int] = [gbest_val]
    conv_iter = 0

    for gen in range(1, max_iter + 1):
        # --- Atualizar velocidades ---
        r1 = rng.random((n_particles, n))
        r2 = rng.random((n_particles, n))

        V = (
            w  * V
            + c1 * r1 * (pbest_X - X)
            + c2 * r2 * (gbest_X  - X)
        )
        # Clamp da velocidade
        V = np.clip(V, -v_max, v_max)

        # --- Atualizar posições via sigmoide ---
        S = _sigmoid(V)
        rand_matrix = rng.random((n_particles, n))
        X_new = (rand_matrix < S).astype(int)

        # --- Reparar infactíveis ---
        for p in range(n_particles):
            X_new[p] = _repair(X_new[p], instance, rng)

        X = X_new

        # --- Avaliar e atualizar pbest / gbest ---
        vals = _evaluate_batch(X, instance)
        for p in range(n_particles):
            if vals[p] > pbest_val[p]:
                pbest_val[p] = vals[p]
                pbest_X[p]   = X[p].copy()
                if vals[p] > gbest_val:
                    gbest_val = vals[p]
                    gbest_X   = X[p].copy()
                    conv_iter = gen

        history.append(gbest_val)

    elapsed = time.perf_counter() - t0

    return PSOResult(
        best_value=gbest_val,
        best_solution=gbest_X,
        history=history,
        total_iterations=max_iter,
        convergence_iter=conv_iter,
        elapsed_seconds=elapsed,
        instance_name=instance.name,
        params={
            "n_particles": n_particles,
            "max_iter": max_iter,
            "w": w,
            "c1": c1,
            "c2": c2,
            "v_max": v_max,
        },
    )


# ---------------------------------------------------------------------------
# CLI de teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from parser import load_all_instances

    instances = load_all_instances()
    inst = next(i for i in instances if i.name == "P07")
    print(f"Instância: {inst.name}  (ótimo={inst.optimal_value})")

    res = binary_pso(inst, n_particles=30, max_iter=200, w=0.7, c1=2.0, c2=2.0, seed=42)
    gap = inst.gap(res.best_value)
    print(f"Melhor valor : {res.best_value}")
    print(f"Gap          : {gap:.2f}%" if gap is not None else "Gap: N/A")
    print(f"Gerações     : {res.total_iterations}  (conv. em {res.convergence_iter})")
    print(f"Tempo        : {res.elapsed_seconds:.3f}s")
