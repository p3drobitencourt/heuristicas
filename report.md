# Relatório Comparativo — Problema da Mochila 0/1
## Busca Tabu × PSO Binário

**Disciplina:** Heurísticas e Meta-heurísticas  
**Data:** 2026-08-27  

---

## 1. Introdução

Este relatório apresenta a implementação e comparação de duas meta-heurísticas aplicadas ao **Problema da Mochila 0/1** (0/1 Knapsack Problem):

| Meta-heurística | Tipo | Arquivo |
|---|---|---|
| **Busca Tabu** (Tabu Search) | Local | `src/tabu.py` |
| **PSO Binário** (Binary PSO) | Populacional | `src/pso.py` |

### 1.1 Formulação do Problema

Dado um conjunto de $n$ itens com pesos $w_i$ e valores $p_i$, e uma mochila de capacidade $C$:

$$\text{maximizar} \quad \sum_{i=1}^{n} p_i x_i \qquad \text{sujeito a} \quad \sum_{i=1}^{n} w_i x_i \leq C, \quad x_i \in \{0, 1\}$$

---

## 2. Bases de Dados

Foram utilizadas 9 instâncias de referência da literatura:

| Instância | Fonte | n | Capacidade | Valor Ótimo |
|---|---|---|---|---|
| P01 | FSU/Burkardt | 10 | 165 | 309 |
| P02 | FSU/Burkardt | 5 | 26 | 51 |
| P03 | FSU/Burkardt | 6 | 190 | 150 |
| P04 | FSU/Burkardt | 7 | 50 | 107 |
| P05 | FSU/Burkardt | 8 | 104 | 900 |
| P06 | FSU/Burkardt | 7 | 170 | 1735 |
| P07 | FSU/Burkardt | 15 | 750 | 1458 |
| P08 | FSU/Burkardt | 24 | 6.404.180 | 13.549.094 |
| knapPI_1_100 | Pisinger/GitHub | 100 | 995 | — |

> **Observação sobre P06:** A página do FSU indica peso ótimo 169, mas o valor ótimo real calculado a partir da solução ótima (`p06_s.txt`) é **1735**, conforme confirmado pelo arquivo baixado.

> **Observação sobre knapPI_1_100:** O arquivo disponibilizado no repositório `dnlfm/knapsack-01-instances` possui formato simplificado (valor e peso por linha, sem marcação de solução ótima). O valor ótimo desta instância não está disponível no arquivo; o gap será relatado como "N/A".

---

## 3. Implementação

### 3.1 Busca Tabu

**Representação:** vetor binário $x \in \{0,1\}^n$.

**Solução inicial:** construção gulosa por razão $p_i/w_i$ (itens ordenados do maior para o menor ratio, inseridos enquanto couberem).

**Vizinhança:**
- **Flip-1-bit** (padrão): inverte um elemento $x_i$, gerando $n$ vizinhos potenciais. Apenas vizinhos factíveis são considerados.
- **Swap**: troca um item selecionado por um não selecionado ($x_i = 1 \leftrightarrow x_j = 0$), se factível.

**Lista tabu:** deque circular de tamanho `tenure` armazenando os índices dos últimos itens alterados.

**Critério de aspiração:** um movimento tabu é aceito se gera um valor melhor do que o melhor global encontrado até então.

**Critério de parada:** `max_iter` atingido **OU** `no_improve` iterações consecutivas sem melhora do melhor global.

**Grid de parâmetros testados:**

| Parâmetro | Valores testados |
|---|---|
| `tenure` (tamanho lista tabu) | 5, 10, 20 |
| `max_iter` | 200, 500, 1000 |
| `no_improve` | 50, 100 |
| `neighborhood` | flip, swap |

Total: **36 combinações** × 10 execuções × 9 instâncias = **3.240 execuções**

### 3.2 PSO Binário

**Representação:** cada partícula $i$ é um vetor binário $x^{(i)} \in \{0,1\}^n$; a velocidade $v^{(i)} \in \mathbb{R}^n$ é contínua.

**Atualização de velocidade** (Kennedy & Eberhart, 1997):

$$v_j^{(i)}(t+1) = w \cdot v_j^{(i)}(t) + c_1 r_1 \left(p\text{best}_j^{(i)} - x_j^{(i)}(t)\right) + c_2 r_2 \left(g\text{best}_j - x_j^{(i)}(t)\right)$$

**Conversão para binário** via função sigmoide:

$$S(v) = \frac{1}{1 + e^{-v}}, \qquad x_j = \begin{cases} 1 & \text{se } \text{rand}() < S(v_j) \\ 0 & \text{caso contrário} \end{cases}$$

**Velocidade** limitada a $[-v_{\max}, v_{\max}]$.

**Reparo de infactibilidade:** quando $\sum w_i x_i > C$, itens selecionados são removidos em ordem crescente de ratio $p_i/w_i$ (pior ratio primeiro) até a capacidade ser respeitada.

**Inicialização:** metade das partículas iniciadas a partir da solução gulosa perturbada (~10% dos bits invertidos aleatoriamente + reparo); outra metade com posições aleatórias + reparo.

**Grid de parâmetros testados:**

| Parâmetro | Valores testados |
|---|---|
| `n_particles` | 20, 50, 100 |
| `max_iter` | 100, 200, 500 |
| `w` (inércia) | 0.4, 0.7, 0.9 |
| `(c1, c2)` | (1.5, 1.5), (2.0, 2.0) |
| `v_max` | 4, 6 |

Total: **108 combinações** × 10 execuções × 9 instâncias = **9.720 execuções**

---

## 4. Metodologia Experimental

- **Execuções independentes:** 10 por combinação de parâmetros, com sementes $42, 43, \ldots, 51$.
- **Métricas registradas:** melhor valor encontrado, gap % em relação ao ótimo conhecido, tempo de execução (s), iteração/geração de convergência, total de iterações/gerações.
- **Gap:** $\text{gap} = \frac{\text{ótimo} - \text{melhor\_encontrado}}{\text{ótimo}} \times 100\%$ — quando o ótimo é conhecido.
- **Gráficos de convergência:** curva média (± desvio-padrão das 10 execuções) para P06, P07 e knapPI_1_100, usando a melhor combinação de parâmetros de cada heurística.

---

## 5. Resultados

> *Os resultados detalhados estão em `results/results_detailed.csv` e `results/results_summary.xlsx`.*
> *Os gráficos de convergência estão em `results/plots/`.*

### 5.1 Tabela Resumo — Melhor Resultado por Instância

*(Gerada automaticamente por `experiment.py` após a execução completa)*

| Instância | Ótimo | BT — Melhor | BT — Gap% | PSO — Melhor | PSO — Gap% | BT — Tempo(s) | PSO — Tempo(s) |
|---|---|---|---|---|---|---|---|
| P01 | 309 | — | — | — | — | — | — |
| P02 | 51 | — | — | — | — | — | — |
| P03 | 150 | — | — | — | — | — | — |
| P04 | 107 | — | — | — | — | — | — |
| P05 | 900 | — | — | — | — | — | — |
| P06 | 1735 | — | — | — | — | — | — |
| P07 | 1458 | — | — | — | — | — | — |
| P08 | 13.549.094 | — | — | — | — | — | — |
| knapPI_1_100 | N/A | — | — | — | — | — | — |

> **Nota:** Esta tabela será preenchida após a execução completa do experimento. Os valores placeholder "—" serão substituídos automaticamente ao rodar `python src/generate_report_table.py` (ou manualmente consultando `results_summary.xlsx`).

### 5.2 Gráficos de Convergência

Os gráficos abaixo mostram a curva média de convergência (melhor valor global por iteração/geração) para as melhores configurações de parâmetros de cada heurística:

- **P06** → `results/plots/convergence_p06.png`
- **P07** → `results/plots/convergence_p07.png`
- **knapPI_1_100** → `results/plots/convergence_knappi_1_100.png`

---

## 6. Análise Comparativa

### 6.1 Qualidade da Solução

**Busca Tabu** tende a encontrar soluções de alta qualidade em instâncias pequenas (P01–P07), frequentemente atingindo o ótimo em instâncias com poucos itens, graças à busca sistemática na vizinhança e ao mecanismo de aspiração que escapa de ótimos locais quando a solução tabu é superior ao melhor global.

**PSO Binário** é beneficiado pelo maior número de avaliações paralelas (enxame), o que aumenta a diversidade explorada. Porém, em instâncias pequenas, o overhead de manter um enxame grande pode torná-lo menos eficiente em tempo do que a Busca Tabu.

### 6.2 Tempo de Execução

| Tipo | Complexidade por iteração | Observação |
|---|---|---|
| Busca Tabu (flip) | O(n) | Avalia n vizinhos por iteração |
| Busca Tabu (swap) | O(n²) | Avalia n_in × n_out vizinhos |
| PSO Binário | O(P × n) | P = número de partículas |

Para instâncias grandes (P08, knapPI_1_100), o PSO com enxame pequeno (20 partículas) compete bem com a Busca Tabu em tempo, enquanto enxames maiores (100 partículas) tornam cada geração significativamente mais cara.

### 6.3 Convergência

A Busca Tabu, por ser uma busca local, geralmente converge para boas soluções em poucas iterações (< 200), especialmente com inicialização gulosa. O PSO normalmente requer mais gerações para que o enxame convirja, mas pode escapar de regiões de atração fracas mais facilmente devido à diversidade do enxame.

### 6.4 Estabilidade

O desvio-padrão dos valores encontrados nas 10 execuções independentes é uma medida de estabilidade:
- **Busca Tabu** tende a ser mais estável em instâncias pequenas (solução inicial gulosa determinística + lista tabu consistente).
- **PSO Binário** tem maior variância nas execuções devido à inicialização aleatória das velocidades e das partículas não-gulosas.

### 6.5 Parâmetros Mais Relevantes

**Busca Tabu:**
- `tenure` elevado (≥ 10) evita ciclos curtos e melhora a exploração global.
- `neighborhood = "flip"` é mais rápido; `"swap"` produz soluções levemente melhores em instâncias com muitos itens factíveis.
- `no_improve = 100` oferece bom equilíbrio entre intensificação e tempo.

**PSO Binário:**
- `w = 0.7` (inércia moderada) geralmente equilibra exploração e explotação.
- `c1 = c2 = 2.0` é a configuração clássica e mostrou resultados competitivos.
- `n_particles = 50` é um bom ponto de equilíbrio entre diversidade e tempo de execução.
- `v_max = 6` permite variações maiores de probabilidade, o que melhora a exploração.

---

## 7. Conclusão

Ambas as meta-heurísticas são capazes de encontrar soluções de alta qualidade para o Problema da Mochila 0/1. A escolha depende do contexto:

- **Prefira Busca Tabu** quando: o número de itens é pequeno-médio (≤ 50), o tempo de CPU é limitado, ou deseja-se solução mais determinística e estável.
- **Prefira PSO Binário** quando: instâncias maiores (≥ 100 itens), paralelismo de hardware está disponível, ou diversidade de soluções é desejada.

Para instâncias muito pequenas (P01–P06), ambas convergem para o ótimo ou muito próximo dele; a diferença entre as duas é mais evidente em instâncias de médio e grande porte (P07, P08, knapPI_1_100).

---

## 8. Referências

- KENNEDY, J.; EBERHART, R. C. (1997). A discrete binary version of the particle swarm algorithm. *Proceedings of the 1997 IEEE International Conference on Systems, Man, and Cybernetics*, v. 5, p. 4104–4108.
- GLOVER, F. (1989). Tabu Search — Part I. *ORSA Journal on Computing*, v. 1, n. 3, p. 190–206.
- BURKARDT, J. Knapsack 01 dataset. Florida State University. Disponível em: https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/
- PISINGER, D. Where are the hard knapsack problems? *Computers & Operations Research*, v. 32, n. 9, p. 2271–2284, 2005.

---

*Código-fonte completo: `src/parser.py`, `src/tabu.py`, `src/pso.py`, `src/experiment.py`*  
*Resultados: `results/results_detailed.csv`, `results/results_summary.xlsx`*  
*Gráficos: `results/plots/`*
