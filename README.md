# Problema da Mochila 0/1 com Meta-heurísticas

Este repositório contém a implementação e a avaliação de duas meta-heurísticas para o Problema da Mochila 0/1 (0/1 Knapsack Problem):
- **Busca Tabu** (Tabu Search)
- **PSO Binário** (Binary Particle Swarm Optimization)

## Estrutura do Projeto

- `src/parser.py`: Leitura e carregamento das instâncias.
- `src/tabu.py`: Implementação da Busca Tabu.
- `src/pso.py`: Implementação do PSO Binário.
- `src/experiment.py`: Script principal que roda o grid de hiperparâmetros.
- `src/generate_report_table.py`: Atualiza a tabela de resultados no relatório automaticamente.
- `data/`: Contém as instâncias do problema (P01 a P08 e knapPI_1_100).
- `results/`: Arquivos gerados pelo experimento (CSV, XLSX, gráficos).
- `report.md`: Relatório comparativo final e análise dos resultados.

## Requisitos

- Python 3.9+
- Dependências listadas em `requirements.txt` (ou instale manualmente: `numpy`, `pandas`, `matplotlib`, `openpyxl`).

Instalação rápida das dependências:
```bash
pip install numpy pandas matplotlib openpyxl
```

## Como Executar

Para rodar a bateria de experimentos e gerar os relatórios completos:

1. **Rodar os Experimentos:**
   O script executará a Busca Tabu e o PSO Binário nas 9 instâncias variando os parâmetros, salvando tabelas detalhadas e gráficos:
   ```bash
   python -u src/experiment.py
   ```
   *Nota: Este comando pode levar de 2 a 5 minutos, dependendo da máquina.*

2. **Gerar a Tabela Resumo no Relatório:**
   Após a conclusão dos experimentos, atualize o `report.md` com os novos valores preenchidos:
   ```bash
   python src/generate_report_table.py
   ```

## Onde Encontrar os Resultados

- **CSV com todas as execuções:** `results/results_detailed.csv`
- **Tabela Resumo Agrupada (Excel):** `results/results_summary.xlsx`
- **Gráficos de Convergência:** `results/plots/`
- **Relatório Final:** `report.md`
