import os
import time
import csv
import json
import hashlib
import psycopg2
import ollama  

# ====== CONFIGURAÇÕES ======
PG_HOST = "localhost"
PG_PORT = '5432'          
PG_DB = "shopmall"        # troque para "goods" quando quiser
PG_USER = "postgres"
PG_PASS = "postgres"

QUERIES_FILE = "dbgpt_exp/queries.txt"
RESULTS_CSV = "dbgpt_exp/results_ollama.csv"

OLLAMA_MODEL = "llama3.1:8b"  

# Prompt base no estilo do artigo (Instruction / Example / Input)
PROMPT_INSTRUCTION = (
    "Rewrite the input SQL query to produce an equivalent query that can be executed "
    "on a PostgreSQL database with decreased latency. Return ONLY the SQL."
)

PROMPT_EXAMPLE = """Example Input:
select ... from t1 where t1.a=(select avg(a) from t3 where t1.b=t3.b);

Example Output:
select ... from t1 inner join (
    select avg(a) avg, t3.b from t3 group by t3.b
) as t3
on (t1.a = avg and t1.b = t3.b);
"""

# ====== CONEXÃO PG ======
def pg_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def fetch_all(sql: str):
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        if cur.description:
            return cur.fetchall()
        return []

def run_timed(sql: str):
    t0 = time.time()
    rows = fetch_all(sql)
    dt = (time.time() - t0) * 1000.0  # ms
    return rows, dt

def explain_json(sql: str):
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql)
        data = cur.fetchall()[0][0][0]  # JSON root obj (list with one element)
        # Times em ms, se disponíveis
        planning = data.get("Planning Time", None)
        execution = data.get("Execution Time", None)
        plan = data.get("Plan", {})
        return data, planning, execution, plan

def rows_signature(rows):
    # assinatura simples para checar equivalência de resultado
    # (ordem pode variar; se necessário, ajuste sua query para ORDER BY)
    h = hashlib.md5()
    for r in rows:
        h.update(json.dumps(r, default=str).encode("utf-8"))
    return h.hexdigest()

# ====== OLLAMA ======
def rewrite_sql_via_ollama(original_sql: str, schema_hint: str = "") -> str:
    # Crie o prompt no mesmo formato
    parts = [
        PROMPT_INSTRUCTION,
        "",
        PROMPT_EXAMPLE,
        ""
    ]
    if schema_hint:
        parts.append("Schema:")
        parts.append(schema_hint)
        parts.append("")  # linha em branco

    parts.append("Input:")
    parts.append(original_sql.strip())

    user_content = "\n".join(parts)

    # Gere a resposta usando o Ollama
    retry_count = 0
    while retry_count < 5:
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": user_content}],
                options={
                    "temperature": 0.7,
                    "num_ctx": 8192
                }
            )
            rewritten_sql = response["message"]["content"].strip().strip("```").strip()
            # Remover o prefixo "sql" se estiver presente
            if rewritten_sql.lower().startswith("sql "):
                rewritten_sql = rewritten_sql[3:].strip()
            return rewritten_sql
        except Exception as e:
            print(f"Erro ao chamar Ollama: {e}. Tentando novamente...")
            retry_count += 1
            time.sleep(5)
    raise RuntimeError("Falha ao obter resposta do Ollama após várias tentativas.")

# ====== LEITURA DE QUERIES ======
def read_queries(path: str):
    # separa pelas linhas em branco
    with open(path, "r", encoding="utf-8") as f:
        blob = f.read()
    blocks = [b.strip() for b in blob.split("\n\n") if b.strip() and not b.strip().startswith("--")]
    return blocks

# ====== MAIN ======
def main():
    queries = read_queries(QUERIES_FILE)
    first_write = not os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first_write:
            w.writerow([
                "db", "query_id",
                "original_ms", "execution_ms_original", "planning_ms_original", "buffers_plan_original",
                "rewritten_ms", "execution_ms_rewritten", "planning_ms_rewritten", "buffers_plan_rewritten",
                "speedup", "same_rowcount", "same_signature",
                "original_sql", "rewritten_sql"
            ])

        for i, original in enumerate(queries, start=1):
            print(f"\n=== Query {i} ===")
            # LLM rewrite
            rewritten = rewrite_sql_via_ollama(original)

            if rewritten.lower().startswith("sql"):
                rewritten = rewritten[3:].strip()
                
            # Original
            try:
                rows_orig, t_orig = run_timed(original)
                ej_orig, plan_ms_o, exec_ms_o, plan_o = explain_json(original)
            except Exception as e:
                print("Erro original:", e)
                rows_orig, t_orig, ej_orig, plan_ms_o, exec_ms_o, plan_o = [], float("nan"), {}, None, None, {}
            
            # Rewritten
            try:
                rows_rew, t_rew = run_timed(rewritten)
                ej_rew, plan_ms_r, exec_ms_r, plan_r = explain_json(rewritten)
            except Exception as e:
                print("Erro reescrita:", e)
                rows_rew, t_rew, ej_rew, plan_ms_r, exec_ms_r, plan_r = [], float("nan"), {}, None, None, {}

            # Comparação simples de correção
            same_count = (len(rows_orig) == len(rows_rew))
            sig_o = rows_signature(rows_orig)
            sig_r = rows_signature(rows_rew)
            same_sig = (sig_o == sig_r)

            # Buffers (se desejar algo do plano)
            buffers_o = plan_o.get("Shared Hit Blocks", None) if isinstance(plan_o, dict) else None
            buffers_r = plan_r.get("Shared Hit Blocks", None) if isinstance(plan_r, dict) else None

            speedup = (t_orig / t_rew) if (isinstance(t_orig, float) and isinstance(t_rew, float) and t_rew > 0) else ""

            w.writerow([
                PG_DB, i,
                f"{t_orig:.3f}" if isinstance(t_orig, float) else "",
                f"{exec_ms_o:.3f}" if exec_ms_o else "",
                f"{plan_ms_o:.3f}" if plan_ms_o else "",
                buffers_o if buffers_o is not None else "",
                f"{t_rew:.3f}" if isinstance(t_rew, float) else "",
                f"{exec_ms_r:.3f}" if exec_ms_r else "",
                f"{plan_ms_r:.3f}" if plan_ms_r else "",
                buffers_r if buffers_r is not None else "",
                f"{speedup:.3f}" if speedup != "" else "",
                same_count, same_sig,
                original.replace("\n", " ").strip(),
                rewritten.replace("\n", " ").strip()
            ])

            print("OK → linha gravada em results_ollama.csv")

if __name__ == "__main__":
    main()