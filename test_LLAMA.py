import os
import time
import csv
import json
import hashlib
import psycopg2
import ollama
from codecarbon import EmissionsTracker

PROMPT_MODE = "zero-shot"  # "zero-shot"  ou "few-shot" ou "chain-of-thought"

# ====== CONFIGURAÇÕES ======

PG_URL = "postgresql://neondb_owner:npg_xAOBhf4MkK5C@ep-plain-sunset-aervpg8i-pooler.c-2.us-east-2.aws.neon.tech/webshopdb?sslmode=require&channel_binding=require"

SCHEMA_HINT = """
webshop.address(id, customerid, firstname, lastname, address1, address2, city, zip, created, updated)
webshop.articles(id, productid, ean, colorid, size, description, originalprice, reducedprice, taxrate, discountinpercent, currentlyactive, created, updated)
webshop.colors(id, name, rgb)
webshop.customer(id, firstname, lastname, gender, email, dateofbirth, currentaddressid, created, updated)
webshop.labels(id, name, slugname, icon)
webshop.order(id, customer, ordertimestamp, shippingaddressid, total, shippingcost, created, updated)
webshop.order_positions(id, orderid, articleid, amount, price, created, updated)
webshop.products(id, name, labelid, category, gender, currentlyactive, created, updated)
webshop.sizes(id, gender, category, size, size_us, size_uk, size_eu)
webshop.stock(id, articleid, count, created, updated)
""".strip()

OLLAMA_MODEL = "llama3.1:8b"

QUERIES_FILE = "dbgpt_exp/queries.txt"
RESULTS_CSV = f"dbgpt_exp/results_ollama_{PROMPT_MODE}.csv"
EMISSIONS_CSV = f"dbgpt_exp/emissions_ollama_{PROMPT_MODE}.csv"

# Prompt base no estilo do artigo (Instruction / Example / Input)
PROMPT_INSTRUCTION = (
    "Rewrite the input SQL query to produce an equivalent query that can be executed "
    "on a PostgreSQL database with decreased latency. Return ONLY the SQL."
)

PROMPT_EXAMPLE = """
Example Input:
SELECT c.id, c.firstname, c.lastname
FROM customer c
WHERE c.id IN (
    SELECT o.customer
    FROM "order" o
    WHERE o.total > 100
);

Example Output:
SELECT c.id, c.firstname, c.lastname
FROM customer c
JOIN (
    SELECT DISTINCT o.customer AS customer_id
    FROM "order" o
    WHERE o.total > 100
) sub ON sub.customer_id = c.id;
"""

# ====== CONSTRUÇÃO DE PROMPTS (Zero-shot / Few-shot / Prompt-Chaining) ======

def build_zero_shot_prompt(original_sql: str, schema_hint: str = "") -> str:
    schema_line = f"Schema: {schema_hint}\n" if schema_hint else ""
    prompt = (
        f"{PROMPT_INSTRUCTION}\n"
        f"{schema_line}"
        f"Input SQL:\n{original_sql.strip()}\n"
        f"Output SQL:"
    )
    return prompt

def build_few_shot_prompt(original_sql: str, schema_hint: str = "") -> str:
    schema_line = f"Schema: {schema_hint}\n" if schema_hint else ""
    prompt = (
        f"{PROMPT_INSTRUCTION}\n"
        f"{PROMPT_EXAMPLE}\n"
        f"{schema_line}"
        f"Input SQL:\n{original_sql.strip()}\n"
        f"Output SQL:"
    )
    return prompt


def build_chain_of_thought_prompt(original_sql: str, schema_hint: str = "") -> str:
    schema_line = f"Schema: {schema_hint}\n" if schema_hint else ""
    prompt = (
        f"{PROMPT_INSTRUCTION}\n"
        f"{schema_line}"
        f"Input SQL:\n{original_sql.strip()}\n"
        "First, explain step by step how you would optimize this query for lower latency in PostgreSQL. "
        "Then, provide ONLY the final optimized SQL query."
    )
    return prompt

# ====== CONEXÃO PG ======
def pg_conn():
    return psycopg2.connect(PG_URL)

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
        data = cur.fetchall()[0][0][0]  
        planning = data.get("Planning Time", None)
        execution = data.get("Execution Time", None)
        plan = data.get("Plan", {})
        return data, planning, execution, plan

def rows_signature(rows):
    h = hashlib.md5()
    for r in rows:
        h.update(json.dumps(r, default=str).encode("utf-8"))
    return h.hexdigest()

def extract_sql_from_response(raw: str) -> str:
    """
    Extrai SQL sem remover linhas essenciais. Remove apenas markdown.
    Mantém a estrutura completa do SELECT.
    """
    if not raw:
        return ""

    cleaned = raw.strip()

    # remove blocos markdown
    cleaned = cleaned.replace("```sql", "")
    cleaned = cleaned.replace("```SQL", "")
    cleaned = cleaned.replace("```", "")

    # se a resposta contém SELECT, apanha tudo a partir do primeiro SELECT
    idx = cleaned.lower().find("select")
    if idx != -1:
        return cleaned[idx:].strip()

    return cleaned.strip()

def extract_sql_from_response_cot(raw: str) -> str:
    cleaned = raw.strip()

    # Tenta encontrar bloco ```sql ... ```
    lower = cleaned.lower()
    if "```sql" in lower:
        idx = lower.index("```sql")
        segment = cleaned[idx + len("```sql"):]
        # corta no próximo ```
        if "```" in segment:
            segment = segment.split("```", 1)[0]
        return segment.strip()

    # Pega a última linha que começa com SELECT
    lines = cleaned.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().upper().startswith("SELECT"):
            return "\n".join(lines[i:]).strip()

    # Retornar tudo em último caso
    return cleaned.strip()

# ====== LLAMA ======
def rewrite_sql_via_ollama(
    original_sql: str,
    schema_hint: str = "",
    prompt_mode: str = None,
) -> str:
    if prompt_mode is None:
        prompt_mode = PROMPT_MODE

    if prompt_mode == "zero_shot":
        user_prompt = build_zero_shot_prompt(original_sql, schema_hint)
    elif prompt_mode in ("cot", "chain_of_thought"):
        user_prompt = build_chain_of_thought_prompt(original_sql, schema_hint)
    else:
        user_prompt = build_few_shot_prompt(original_sql, schema_hint)

    # System message precisa variar para cot
    if prompt_mode in ("cot", "chain_of_thought"):
        system_msg = (
            "You are a PostgreSQL SQL optimization assistant. "
            "You will first briefly explain how to optimize the query, "
            "then output the final optimized SQL query. "
            "The final SQL query must start with the word SELECT."
        )
    else:
        system_msg = (
            "You are a PostgreSQL SQL optimization assistant. "
            "Return ONLY a valid SQL SELECT query, with no explanations, no comments and no markdown."
        )

    # Chamada ao Ollama — sem temperature (par com OpenAI que também não passa temperature)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        # não passar temperature para manter comportamento igual ao OpenAI
    )

    raw = response["message"]["content"]

    if prompt_mode in ("cot", "chain_of_thought"):
        sql = extract_sql_from_response_cot(raw)
    else:
        sql = extract_sql_from_response(raw)

    return sql

# ====== LEITURA DE QUERIES ======
def read_queries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        blob = f.read()
    blocks = [b.strip() for b in blob.split("\n\n") if b.strip() and not b.strip().startswith("--")]
    return blocks

def run_query_with_energy(sql: str, tag: str):
    tracker = EmissionsTracker(
        project_name=f"ollama_{PROMPT_MODE}_{tag}",
        output_dir=os.path.dirname(EMISSIONS_CSV),
        save_to_file=False,
        log_level="error",
        measure_power_secs=1,
    )
    tracker.start()
    try:
        rows, ms = run_timed(sql)
        emissions = tracker.stop()
    except Exception:
        tracker.stop()
        raise
    return rows, ms, emissions

def main():
    queries = read_queries(QUERIES_FILE)
    first_write_results = not os.path.exists(RESULTS_CSV)
    first_write_emissions = not os.path.exists(EMISSIONS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f, \
         open(EMISSIONS_CSV, "a", newline="", encoding="utf-8") as f_em:

        w = csv.writer(f)
        w_em = csv.writer(f_em)

        if first_write_results:
            w.writerow([
                "db","query_id",
                "original_ms","execution_ms_original","planning_ms_original","buffers_plan_original",
                "rewritten_ms","execution_ms_rewritten","planning_ms_rewritten","buffers_plan_rewritten",
                "emissions_original", "emissions_rewritten", 
                "speedup","buffers_ratio","same_rowcount","same_signature",
                "original_sql","rewritten_sql"
            ])

        if first_write_emissions:
            w_em.writerow([
                "db", "llm", "prompt_technique", "query_id",
                "emissions_original", "emissions_rewritten",
                "energy_ratio", "energy_saving_pct"
            ])

        for i, original in enumerate(queries, start=1):
            print(f"\n=== Query {i} ===")

            rewritten = rewrite_sql_via_ollama(original)

             # Original
            try:
                rows_orig, t_orig, em_orig = run_query_with_energy(original, "original")
                ej_orig, plan_ms_o, exec_ms_o, plan_o = explain_json(original)
            except Exception as e:
                print("Erro original:", e)
                rows_orig, t_orig, ej_orig, plan_ms_o, exec_ms_o, plan_o = [], float("nan"), {}, None, None, {}
                em_orig = float("nan")
            
            # Reescrita
            try:
                rows_rew, t_rew, em_rew = run_query_with_energy(rewritten, "rewritten")
                ej_rew, plan_ms_r, exec_ms_r, plan_r = explain_json(rewritten)
            except Exception as e:
                print("Erro reescrita:", e)
                rows_rew, t_rew, ej_rew, plan_ms_r, exec_ms_r, plan_r = [], float("nan"), {}, None, None, {}
                em_rew = float("nan")

            # Comparação simples de correção
            same_count = (len(rows_orig) == len(rows_rew))
            sig_o = rows_signature(rows_orig)
            sig_r = rows_signature(rows_rew)
            same_sig = (sig_o == sig_r)

            buffers_o = plan_o.get("Shared Hit Blocks", None) if isinstance(plan_o, dict) else None
            buffers_r = plan_r.get("Shared Hit Blocks", None) if isinstance(plan_r, dict) else None

            if (
                exec_ms_o is not None
                and exec_ms_r is not None
                and exec_ms_r > 0
            ):
                speedup = exec_ms_o / exec_ms_r
            else:
                speedup = ""
                
            if (
                isinstance(buffers_o, (int, float))
                and isinstance(buffers_r, (int, float))
                and buffers_r > 0
            ):
                buffers_ratio = buffers_o / buffers_r
            else:
                buffers_ratio = ""
                
            energy_ratio = float("nan")
            energy_saving_pct = float("nan")
            if (
                isinstance(em_orig, float)
                and isinstance(em_rew, float)
                and em_rew > 0.0
                and em_orig > 0.0
            ):
                energy_ratio = em_orig / em_rew
                energy_saving_pct = (em_orig - em_rew) / em_orig * 100.0

            w.writerow([
                "webshopdb", i,
                f"{t_orig:.3f}" if isinstance(t_orig, float) else "",
                f"{exec_ms_o:.3f}" if exec_ms_o is not None else "",
                f"{plan_ms_o:.3f}" if plan_ms_o is not None else "",
                buffers_o if buffers_o is not None else "",
                f"{t_rew:.3f}" if isinstance(t_rew, float) else "",
                f"{exec_ms_r:.3f}" if exec_ms_r is not None else "",
                f"{plan_ms_r:.3f}" if plan_ms_r is not None else "",
                buffers_r if buffers_r is not None else "",
                f"{speedup:.3f}" if speedup != "" else "",
                f"{buffers_ratio:.3f}" if buffers_ratio != "" else "",
                em_orig, em_rew,
                same_count, same_sig,
                original.replace("\n"," ").strip(),
                rewritten.replace("\n"," ").strip()
            ])

            w_em.writerow([
                "webshopdb",
                OLLAMA_MODEL,
                PROMPT_MODE,
                i,
                em_orig,
                em_rew,
                energy_ratio,
                energy_saving_pct,
            ])

            print("OK → linha gravada em results.csv")

if __name__ == "__main__":
    main()