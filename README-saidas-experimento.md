# 📄 README — Explicação das Saídas do Experimento (CSV)

Este documento descreve detalhadamente **todas as colunas** geradas pelo experimento que avalia reescrita de queries SQL com o modelo **Mistral**, medindo **desempenho**, **correção semântica** e **emissões de CO₂** via CodeCarbon.

O CSV principal é:

```
results_mistral_<prompt_technique>.csv
```

---

# 🧭 Estrutura Geral do CSV

O arquivo contém informações em quatro blocos:

1. **Identificação do experimento**
2. **Métricas da query original**
3. **Métricas da query reescrita pelo LLM**
4. **Comparação semântica e energética**

A seguir, cada coluna é documentada.

---

# 1) 🔹 Identificação do Experimento

| Coluna | Tipo | Descrição |
|--------|------|------------|
| **db** | string | Nome lógico do banco utilizado. Ex.: `webshopdb`. |
| **llm** | string | Modelo LLM utilizado. Ex.: `mistral-small-latest`. |
| **prompt_technique** | string | Estratégia de prompt usada: `zero-shot`, `few-shot` ou `chain-of-thought`. |
| **query_id** | int | Índice da query dentro do arquivo `queries.txt` (1, 2, 3, ...). |

---

# 2) 🕒 Métricas da Query Original

Essas colunas representam o desempenho da query **não reescrita**.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| **original_ms** | string (float-format) | Tempo total de execução da query original (ms), medido via `time.time()` em Python. |
| **execution_ms_original** | string (float) | Tempo de execução reportado pelo `EXPLAIN ANALYZE` (`Execution Time`). |
| **planning_ms_original** | string (float) | Tempo de planejamento do `EXPLAIN ANALYZE` (`Planning Time`). |
| **buffers_plan_original** | int / vazio | Valor de `Shared Hit Blocks` (quantidade de blocos retornados do cache). Pode ser vazio se o plano não possuir essas métricas. |

---

# 3) 🔁 Métricas da Query Reescrita pelo LLM

Essas colunas são iguais às anteriores, porém aplicadas à **query gerada pelo Mistral**.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| **rewritten_ms** | string (float) | Tempo total de execução da query reescrita. Será `NaN` se a reescrita for inválida (erro de sintaxe, etc.). |
| **execution_ms_rewritten** | string (float) | Execution Time do `EXPLAIN ANALYZE` para a query reescrita. |
| **planning_ms_rewritten** | string (float) | Planning Time da query reescrita. |
| **buffers_plan_rewritten** | int / vazio | Shared Hit Blocks da reescrita. |

---

# 4) 🌱 Métricas de Energia (CodeCarbon)

Cada query é executada dentro de um `EmissionsTracker`, gerando estimativas de carbono emitido.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| **emissions_original** | float | Emissões estimadas de CO₂ (kg CO₂e) associadas à execução da query original. |
| **emissions_rewritten** | float | Emissões estimadas de CO₂ (kg CO₂e) associadas à execução da query reescrita. |

### ⚠ Observação importante sobre energia no Windows:

O CodeCarbon **não consegue medir energia nativamente no Windows**, então:

- Ele usa um **modelo estimado baseado no TDP da CPU**
- As emissões são **aproximadas**, porém **consistentes** para comparar *original vs reescrita*

---

# 5) 🧠 Métricas de Correção / Comparação Semântica

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| **speedup** | string (float) | `original_ms / rewritten_ms`. Ex.: `2.0` = reescrita **2x mais rápida**. |
| **same_rowcount** | bool | `True` se as duas queries retornaram **o mesmo número de linhas**. |
| **same_signature (same_sig)** | bool | Indica se **os resultados são exatamente iguais**. Calculado por hash MD5 das linhas retornadas. |

`same_sig = True` significa **equivalência semântica total**.

---

# 6) 📄 Sobre o arquivo `emissions_mistral_<prompt>.csv`

Gerado automaticamente pelo CodeCarbon. Contém medições de energia para cada execução:

- Query original → `project_name = mistral_<prompt>_original`
- Query reescrita → `project_name = mistral_<prompt>_rewritten`

Colunas típicas:

- `timestamp`
- `project_name`
- `duration`
- `emissions`
- `energy_consumed`
- estimativas de CPU/GPU/RAM

---

# 📌 Conclusão

As métricas permitem avaliar cada reescrita do LLM em três dimensões:

- **Correção semântica** (`same_sig`)
- **Desempenho** (speedup, tempos do EXPLAIN)
- **Consumo energético** (emissões)

