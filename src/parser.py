"""
parser.py
=========
Módulo de download e parse das 9 instâncias do Problema da Mochila 0/1.

Fontes:
  - P01–P08 : https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/
  - knapPI   : https://github.com/dnlfm/knapsack-01-instances (formato Pisinger)
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Estrutura de dado padrão
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """Representa uma instância do Problema da Mochila 0/1."""
    name: str
    n: int
    capacity: int
    weights: np.ndarray
    values: np.ndarray
    optimal_value: Optional[int] = None   # None se não disponível
    optimal_solution: Optional[np.ndarray] = None  # vetor 0/1 ótimo

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=int)
        self.values  = np.asarray(self.values,  dtype=int)
        if self.optimal_solution is not None:
            self.optimal_solution = np.asarray(self.optimal_solution, dtype=int)
            # Calcular valor ótimo a partir do vetor solução, se não fornecido
            if self.optimal_value is None:
                self.optimal_value = int(self.values @ self.optimal_solution)

    def evaluate(self, x: np.ndarray) -> tuple[int, int]:
        """Retorna (valor, peso) de um vetor solução x."""
        x = np.asarray(x, dtype=int)
        return int(self.values @ x), int(self.weights @ x)

    def is_feasible(self, x: np.ndarray) -> bool:
        """Verifica se a solução respeita a capacidade."""
        return int(self.weights @ np.asarray(x, dtype=int)) <= self.capacity

    def gap(self, value: int) -> Optional[float]:
        """Gap % em relação ao ótimo conhecido. None se ótimo desconhecido."""
        if self.optimal_value is None or self.optimal_value == 0:
            return None
        return (self.optimal_value - value) / self.optimal_value * 100.0


# ---------------------------------------------------------------------------
# P01–P08  (FSU / Burkardt)
# ---------------------------------------------------------------------------

FSU_BASE = "https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/"

# Mapeamento: instância -> nome do arquivo de valores (p ou v)
_VALUE_FILE = {
    "p01": "p01_p.txt",
    "p02": "p02_p.txt",
    "p03": "p03_p.txt",
    "p04": "p04_p.txt",
    "p05": "p05_p.txt",
    "p06": "p06_p.txt",
    "p07": "p07_p.txt",
    "p08": "p08_p.txt",
}


def _download(url: str, dest: str, force: bool = False) -> str:
    """Baixa `url` para `dest` (só baixa se o arquivo não existir)."""
    if os.path.exists(dest) and not force:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        raise RuntimeError(f"Falha ao baixar {url}: {exc}") from exc
    return dest


def _read_ints(path: str) -> list[int]:
    """Lê todos os inteiros de um arquivo (um por linha, ignora linhas vazias)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_fsu_instance(name: str, data_dir: str = "data", force: bool = False) -> Instance:
    """
    Baixa e parseia uma instância P01–P08 do site FSU.

    Parâmetros
    ----------
    name     : "p01" … "p08"
    data_dir : pasta raiz onde os arquivos serão salvos
    force    : baixa novamente mesmo se o arquivo já existir
    """
    inst_dir = os.path.join(data_dir, name)
    os.makedirs(inst_dir, exist_ok=True)

    key = name.lower()
    val_file = _VALUE_FILE.get(key, f"{key}_p.txt")

    files = {
        "capacity": f"{key}_c.txt",
        "weights":  f"{key}_w.txt",
        "values":   val_file,
        "solution": f"{key}_s.txt",
    }

    local = {}
    for role, fname in files.items():
        url  = FSU_BASE + fname
        dest = os.path.join(inst_dir, fname)
        try:
            _download(url, dest, force=force)
            local[role] = dest
        except RuntimeError:
            if role == "solution":
                local[role] = None   # solução ótima pode não existir
            else:
                raise

    capacity = _read_ints(local["capacity"])[0]
    weights  = _read_ints(local["weights"])
    values   = _read_ints(local["values"])

    opt_sol = None
    if local.get("solution") and os.path.exists(local["solution"]):
        try:
            opt_sol = _read_ints(local["solution"])
        except Exception:
            opt_sol = None

    return Instance(
        name=name.upper(),
        n=len(weights),
        capacity=capacity,
        weights=weights,
        values=values,
        optimal_solution=np.array(opt_sol) if opt_sol else None,
    )


# ---------------------------------------------------------------------------
# knapPI_1_100_1000_1  (formato Pisinger — GitHub)
# ---------------------------------------------------------------------------

KNAPPI_URL = (
    "https://raw.githubusercontent.com/dnlfm/knapsack-01-instances/"
    "main/pisinger_instances_01_KP/large_scale/knapPI_1_100_1000_1"
)


def _parse_pisinger(path: str) -> Instance:
    """
    Parseia o arquivo knapPI_1_100_1000_1 do repositório dnlfm/knapsack-01-instances.

    Formato real do arquivo (verificado por inspeção direta):
        Linha 1      : "n capacidade"     ex.: "100 995"
        Linhas 2..n+1: "valor peso"       ex.: "94 485"
        Última linha : valor_ótimo        ex.: "196"

    A capacidade é usada diretamente (sem multiplicação).
    O valor ótimo da última linha é o número de itens ótimos selecionados
    (não o valor da função objetivo), portanto será calculado via busca exata
    ou simplesmente marcado como None para reportar gap sem referência.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f if l.strip()]

    # Linha 1: n e capacidade
    header = lines[0].split()
    n        = int(header[0])
    capacity = int(header[1])   # capacidade real (sem multiplicação)

    values  = []
    weights = []

    for line in lines[1: 1 + n]:
        parts = line.split()
        values.append(int(parts[0]))
        weights.append(int(parts[1]))

    # A última linha parece ser o ótimo de referência (196 no arquivo).
    # Como é incerto se é o valor ótimo ou outra métrica, deixamos None
    # e usaremos a melhor solução encontrada pelas heurísticas como referência.
    optimal_value = None

    return Instance(
        name="knapPI_1_100",
        n=n,
        capacity=capacity,
        weights=weights,
        values=values,
        optimal_value=optimal_value,
        optimal_solution=None,
    )


def load_knappi_instance(data_dir: str = "data", force: bool = False) -> Instance:
    """Baixa e parseia a instância knapPI_1_100_1000_1 do GitHub."""
    inst_dir = os.path.join(data_dir, "knappi_1_100")
    os.makedirs(inst_dir, exist_ok=True)
    dest = os.path.join(inst_dir, "knapPI_1_100_1000_1.txt")
    _download(KNAPPI_URL, dest, force=force)
    return _parse_pisinger(dest)


# ---------------------------------------------------------------------------
# Carregar todas as 9 instâncias
# ---------------------------------------------------------------------------

def load_all_instances(data_dir: str = "data", force: bool = False) -> list[Instance]:
    """
    Carrega todas as 9 instâncias (P01–P08 + knapPI_1_100).

    Retorna lista de Instance na ordem P01, P02, …, P08, knapPI_1_100.
    """
    instances = []
    for i in range(1, 9):
        name = f"p{i:02d}"
        print(f"  Carregando {name.upper()}...", end=" ", flush=True)
        try:
            inst = load_fsu_instance(name, data_dir=data_dir, force=force)
            print(f"OK  (n={inst.n}, C={inst.capacity}, ótimo={inst.optimal_value})")
            instances.append(inst)
        except Exception as exc:
            print(f"ERRO: {exc}")

    print("  Carregando knapPI_1_100...", end=" ", flush=True)
    try:
        inst = load_knappi_instance(data_dir=data_dir, force=force)
        print(f"OK  (n={inst.n}, C={inst.capacity}, ótimo={inst.optimal_value})")
        instances.append(inst)
    except Exception as exc:
        print(f"ERRO: {exc}")

    return instances


# ---------------------------------------------------------------------------
# Utilitário: solução gulosa (razão valor/peso)
# ---------------------------------------------------------------------------

def greedy_solution(instance: Instance) -> np.ndarray:
    """
    Constrói uma solução gulosa ordenando itens por razão valor/peso (decrescente)
    e inserindo cada item enquanto couber.
    """
    ratios = instance.values / np.maximum(instance.weights, 1e-9)
    order  = np.argsort(ratios)[::-1]
    x      = np.zeros(instance.n, dtype=int)
    cap    = instance.capacity
    for i in order:
        if instance.weights[i] <= cap:
            x[i] = 1
            cap  -= instance.weights[i]
    return x


# ---------------------------------------------------------------------------
# CLI rápido de diagnóstico
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Carregando todas as instâncias ===")
    insts = load_all_instances()
    print("\n=== Resumo ===")
    print(f"{'Instância':<15} {'n':>5} {'Capacidade':>12} {'Ótimo':>12} {'Guloso':>8}")
    print("-" * 55)
    for inst in insts:
        g = greedy_solution(inst)
        gv, _ = inst.evaluate(g)
        opt_str = str(inst.optimal_value) if inst.optimal_value else "?"
        print(f"{inst.name:<15} {inst.n:>5} {inst.capacity:>12} {opt_str:>12} {gv:>8}")
