"""
tabu.py
=======
Implementação da Busca Tabu para o Problema da Mochila 0/1.

Representação : vetor binário  x ∈ {0,1}^n
Vizinhança    : flip-1-bit (padrão) ou swap (1 item dentro ↔ 1 fora)
Lista tabu    : armazena os últimos `tenure` índices alterados
Aspiração     : aceita movimento tabu se produz novo melhor global
Parada        : max_iter atingido OU no_improve iterações sem melhora
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from parser import Instance, greedy_solution


# ---------------------------------------------------------------------------
# Estrutura de resultado
# ---------------------------------------------------------------------------

@dataclass
class TabuResult:
    best_value: int
    best_solution: np.ndarray
    history: list[int]          # melhor valor por iteração
    total_iterations: int
    convergence_iter: int       # iteração em que o melhor global foi encontrado
    elapsed_seconds: float
    instance_name: str
    params: dict


# ---------------------------------------------------------------------------
# Busca Tabu
# ---------------------------------------------------------------------------

def _flip_neighbors(x: np.ndarray, instance: Instance) -> list[tuple[int, np.ndarray, int, int]]:
    """
    Gera todos os vizinhos por flip de 1 bit.
    Retorna lista de (índice_do_flip, solução_vizinha, valor, peso).
    """
    neighbors = []
    cap = instance.capacity
    val_base, peso_base = instance.evaluate(x)

    for i in range(instance.n):
        xi_new = 1 - x[i]
        delta_peso = instance.weights[i] * (xi_new - x[i])
        delta_val  = instance.values[i]  * (xi_new - x[i])
        novo_peso  = peso_base + delta_peso
        novo_val   = val_base  + delta_val

        if novo_peso <= cap:               # apenas movimentos factíveis
            xn = x.copy()
            xn[i] = xi_new
            neighbors.append((i, xn, novo_val, novo_peso))

    return neighbors


def _swap_neighbors(x: np.ndarray, instance: Instance) -> list[tuple[int, np.ndarray, int, int]]:
    """
    Gera vizinhos por swap: troca um item que está dentro (x_i=1) por um que
    está fora (x_j=0), se o resultado for factível.
    Retorna lista de (índice_pivô_codificado, solução_vizinha, valor, peso).
    O "índice pivô" é codificado como i*n + j para rastrear na lista tabu.
    """
    n = instance.n
    inside  = np.where(x == 1)[0]
    outside = np.where(x == 0)[0]
    cap     = instance.capacity
    val_base, peso_base = instance.evaluate(x)
    neighbors = []

    for i in inside:
        for j in outside:
            novo_peso = peso_base - instance.weights[i] + instance.weights[j]
            novo_val  = val_base  - instance.values[i]  + instance.values[j]
            if novo_peso <= cap:
                xn = x.copy()
                xn[i] = 0
                xn[j] = 1
                pivot = i * n + j
                neighbors.append((pivot, xn, novo_val, novo_peso))

    return neighbors


def tabu_search(
    instance: Instance,
    tenure: int = 10,
    max_iter: int = 500,
    no_improve: int = 100,
    neighborhood: str = "flip",  # "flip" ou "swap"
    seed: Optional[int] = None,
) -> TabuResult:
    """
    Executa a Busca Tabu sobre `instance`.

    Parâmetros
    ----------
    tenure       : tamanho da lista tabu (número de iterações que um movimento fica proibido)
    max_iter     : número máximo de iterações
    no_improve   : critério de parada por estagnação
    neighborhood : "flip" (padrão) ou "swap"
    seed         : semente do gerador aleatório

    Retorna
    -------
    TabuResult com melhor solução encontrada, histórico de convergência e métricas.
    """
    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()

    # --- Solução inicial: gulosa, com perturbação aleatória pequena se seed for dado
    x_curr = greedy_solution(instance).copy()
    val_curr, _ = instance.evaluate(x_curr)

    x_best    = x_curr.copy()
    val_best  = val_curr
    conv_iter = 0

    tabu_list: deque[int] = deque(maxlen=tenure)
    history: list[int] = [val_best]
    no_imp_count = 0

    get_neighbors = _flip_neighbors if neighborhood == "flip" else _swap_neighbors

    for iteration in range(1, max_iter + 1):
        neighbors = get_neighbors(x_curr, instance)

        if not neighbors:
            break   # sem vizinhos factíveis (instância muito restrita)

        # Ordenar vizinhos por valor decrescente
        neighbors.sort(key=lambda t: t[2], reverse=True)

        moved = False
        for pivot, xn, val_n, _ in neighbors:
            is_tabu = pivot in tabu_list
            aspiration = val_n > val_best     # critério de aspiração

            if not is_tabu or aspiration:
                x_curr   = xn
                val_curr = val_n
                tabu_list.append(pivot)

                if val_curr > val_best:
                    x_best    = x_curr.copy()
                    val_best  = val_curr
                    conv_iter = iteration
                    no_imp_count = 0
                else:
                    no_imp_count += 1

                moved = True
                break

        if not moved:
            # Todos os movimentos estão proibidos: aceitar o melhor mesmo sendo tabu
            pivot, xn, val_n, _ = neighbors[0]
            x_curr   = xn
            val_curr = val_n
            tabu_list.append(pivot)
            no_imp_count += 1

        history.append(val_best)

        if no_imp_count >= no_improve:
            break

    elapsed = time.perf_counter() - t0

    return TabuResult(
        best_value=val_best,
        best_solution=x_best,
        history=history,
        total_iterations=len(history) - 1,
        convergence_iter=conv_iter,
        elapsed_seconds=elapsed,
        instance_name=instance.name,
        params={
            "tenure": tenure,
            "max_iter": max_iter,
            "no_improve": no_improve,
            "neighborhood": neighborhood,
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

    res = tabu_search(inst, tenure=10, max_iter=500, no_improve=100, seed=42)
    gap = inst.gap(res.best_value)
    print(f"Melhor valor : {res.best_value}")
    print(f"Gap          : {gap:.2f}%" if gap is not None else "Gap: N/A")
    print(f"Iterações    : {res.total_iterations}  (conv. em {res.convergence_iter})")
    print(f"Tempo        : {res.elapsed_seconds:.3f}s")
