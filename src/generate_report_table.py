import os
import pandas as pd
import re

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(root, "results", "results_detailed.csv")
    report_path = os.path.join(root, "report.md")
    
    if not os.path.exists(results_path):
        print("Arquivo results_detailed.csv não encontrado. Rode o experimento primeiro.")
        return

    df = pd.read_csv(results_path)
    
    # Ótimos conhecidos
    otimos = {
        "P01": 309, "P02": 51, "P03": 150, "P04": 107, 
        "P05": 900, "P06": 1735, "P07": 1458, "P08": 13549094, 
        "knapPI_1_100": None
    }
    
    # Agrupar e extrair melhor valor e tempo médio
    agg = df.groupby(["instancia", "heuristica"]).agg(
        melhor=("melhor_valor", "max"),
        tempo=("tempo_s", "mean")
    ).reset_index()
    
    # Construir linhas da tabela
    table_lines = [
        "| Instância | Ótimo | BT — Melhor | BT — Gap% | PSO — Melhor | PSO — Gap% | BT — Tempo(s) | PSO — Tempo(s) |",
        "|---|---|---|---|---|---|---|---|"
    ]
    
    for inst, opt in otimos.items():
        bt_row = agg[(agg["instancia"] == inst) & (agg["heuristica"] == "BuscaTabu")]
        pso_row = agg[(agg["instancia"] == inst) & (agg["heuristica"] == "PSO_Binario")]
        
        bt_melhor = int(bt_row["melhor"].values[0]) if not bt_row.empty else "—"
        pso_melhor = int(pso_row["melhor"].values[0]) if not pso_row.empty else "—"
        
        bt_tempo = f"{bt_row['tempo'].values[0]:.4f}" if not bt_row.empty else "—"
        pso_tempo = f"{pso_row['tempo'].values[0]:.4f}" if not pso_row.empty else "—"
        
        if opt is not None:
            bt_gap = f"{max(0, (opt - bt_melhor) / opt * 100):.2f}%" if bt_melhor != "—" else "—"
            pso_gap = f"{max(0, (opt - pso_melhor) / opt * 100):.2f}%" if pso_melhor != "—" else "—"
            opt_str = str(opt)
        else:
            bt_gap = "N/A"
            pso_gap = "N/A"
            opt_str = "N/A"
            
        table_lines.append(f"| {inst} | {opt_str} | {bt_melhor} | {bt_gap} | {pso_melhor} | {pso_gap} | {bt_tempo} | {pso_tempo} |")
        
    new_table = "\n".join(table_lines)
    
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Substituir tabela no report
    pattern = r"\| Instância \| Ótimo \| BT — Melhor.*?(\n\|.*)+\n"
    content_new = re.sub(pattern, new_table + "\n", content, flags=re.MULTILINE)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content_new)
        
    print("Tabela atualizada no report.md com sucesso!")

if __name__ == "__main__":
    main()
