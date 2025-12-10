import os
import time
import csv
import json
import hashlib
import psycopg2
import ollama
from codecarbon import EmissionsTracker

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

PROMPT_TECHNIQUE = "zero-shot"
OLLAMA_MODEL = "mistral-nemo"

QUERIES_FILE = "dbgpt_exp/queries.txt"
RESULTS_CSV = f"dbgpt_exp/results_{OLLAMA_MODEL}_{PROMPT_TECHNIQUE}.csv"
EMISSIONS_CSV = f"dbgpt_exp/emissions_{OLLAMA_MODEL}_{PROMPT_TECHNIQUE}.csv"

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

def pg_conn():
    return psycopg2.connect(PG_URL)

def fetch_all(sql: str):
    try:
        with pg_conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            if cur.description:
                return cur.fetchall()
            return []
    except Exception:
        return []

def run_timed(sql: str):
    t0 = time.time()
    rows = fetch_all(sql)
    dt = (time.time() - t0) * 1000.0
    return rows, dt

def explain_json(sql: str):
    try:
        with pg_conn() as conn, conn.cursor() as cur:
            cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql)
            data = cur.fetchall()[0][0][0]
            planning = data.get("Planning Time", None)
            execution = data.get("Execution Time", None)
            plan = data.get("Plan", {})
            return data, planning, execution, plan
    except Exception:
        return {}, None, None, {}

def rows_signature(rows):
    h = hashlib.md5()
    for r in rows:
        h.update(json.dumps(r, default=str).encode("utf-8"))
    return h.hexdigest()

def build_zero_shot_prompt(original_sql, schema_hint=""):
    prompt = (
        f"{PROMPT_INSTRUCTION}\n"
        f"{'Schema: ' + schema_hint if schema_hint else ''}\n"
        f"Input SQL:\n{original_sql}\n"
        f"Output SQL:"
    )
    return prompt

def build_few_shot_prompt(original_sql, schema_hint=""):
    prompt = (
        f"{PROMPT_INSTRUCTION}\n"
        f"{PROMPT_EXAMPLE}\n"
        f"{'Schema: ' + schema_hint if schema_hint else ''}\n"
        f"Input SQL:\n{original_sql}\n"
        f"Output SQL:"
    )
    return prompt

def build_chain_of_thought_prompt(original_sql, schema_hint=""):
    prompt = (
        f"{PROMPT_INSTRUCTION}\n"
        f"{'Schema: ' + schema_hint if schema_hint else ''}\n"
        f"Input SQL:\n{original_sql}\n"
        "First, explain step by step how you would optimize this query for lower latency in PostgreSQL. "
        "Then, provide ONLY the final optimized SQL query inside a markdown block."
    )
    return prompt

def rewrite_sql_via_ollama(original_sql: str, schema_hint: str = "") -> str:
    if PROMPT_TECHNIQUE == "zero-shot":
        user_content = build_zero_shot_prompt(original_sql, schema_hint)
    elif PROMPT_TECHNIQUE == "few-shot":
        user_content = build_few_shot_prompt(original_sql, schema_hint)
    elif PROMPT_TECHNIQUE == "chain-of-thought":
        user_content = build_chain_of_thought_prompt(original_sql, schema_hint)
    else:
        raise ValueError("Técnica de prompt desconhecida.")

    retry_count = 0
    while retry_count < 3:
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": user_content}],
                options={
                    "temperature": 0.1,
                    "num_ctx": 4096
                }
            )
            content = response["message"]["content"]
            
            import re
            pattern = r"```(?:sql)?\s*(.*?)\s*```"
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            
            if match:
                rewritten_sql = match.group(1).strip()
            else:
                rewritten_sql = content.strip()
                
            if rewritten_sql.lower().startswith("sql"):
                rewritten_sql = rewritten_sql[3:].strip()
                
            return rewritten_sql

        except Exception:
            retry_count += 1
            time.sleep(2)
            
    return ""

def read_queries(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        blob = f.read()
    blocks = [b.strip() for b in blob.split("\n\n") if b.strip() and not b.strip().startswith("--")]
    return blocks

def run_query_with_energy(sql: str, tag: str):
    tracker = EmissionsTracker(
        project_name=f"{OLLAMA_MODEL}_{PROMPT_TECHNIQUE}_{tag}",
        output_dir=os.path.dirname(EMISSIONS_CSV) if os.path.dirname(EMISSIONS_CSV) else ".",
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
        rows, ms, emissions = [], float("nan"), float("nan")
    return rows, ms, emissions

def fmt_float(v):
    if isinstance(v, float) and not (v != v):
        return f"{v:.3f}"
    return "NaN"

def fmt_pct(v):
    if isinstance(v, float) and not (v != v):
        return f"{v:.2f}"
    return "NaN"

def main():
    queries = read_queries(QUERIES_FILE)
    if not queries:
        return

    os.makedirs(os.path.dirname(RESULTS_CSV) or ".", exist_ok=True)
    
    first_write_results = not os.path.exists(RESULTS_CSV)
    first_write_emissions = not os.path.exists(EMISSIONS_CSV)

    print(f"Iniciando Benchmark com {OLLAMA_MODEL} ({PROMPT_TECHNIQUE})")

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f_res, \
         open(EMISSIONS_CSV, "a", newline="", encoding="utf-8") as f_em:

        w_res = csv.writer(f_res)
        w_em = csv.writer(f_em)

        if first_write_results:
            w_res.writerow([
                "db", "llm", "prompt_technique", "query_id",
                "original_ms", "execution_ms_original", "planning_ms_original", "buffers_plan_original",
                "rewritten_ms", "execution_ms_rewritten", "planning_ms_rewritten", "buffers_plan_rewritten",
                "emissions_original", "emissions_rewritten",
                "speedup", "same_rowcount", "same_signature",
                "original_sql", "rewritten_sql"
            ])

        if first_write_emissions:
            w_em.writerow([
                "db", "llm", "prompt_technique", "query_id",
                "emissions_original", "emissions_rewritten",
                "energy_ratio", "energy_saving_pct"
            ])

        for i, original in enumerate(queries, start=1):
            print(f"\n=== Query {i} ===")
            
            rewritten = rewrite_sql_via_ollama(original, SCHEMA_HINT)
            if not rewritten:
                print("Falha ao gerar SQL.")
                continue

            rows_orig, t_orig, em_orig = run_query_with_energy(original, "original")
            ej_orig, plan_ms_o, exec_ms_o, plan_o = explain_json(original)

            rows_rew, t_rew, em_rew = run_query_with_energy(rewritten, "rewritten")
            ej_rew, plan_ms_r, exec_ms_r, plan_r = explain_json(rewritten)

            same_count = (len(rows_orig) == len(rows_rew))
            sig_o = rows_signature(rows_orig)
            sig_r = rows_signature(rows_rew)
            same_sig = (sig_o == sig_r)

            buffers_o = plan_o.get("Shared Hit Blocks", None) if isinstance(plan_o, dict) else None
            buffers_r = plan_r.get("Shared Hit Blocks", None) if isinstance(plan_r, dict) else None

            speedup = (t_orig / t_rew) if (isinstance(t_orig, float) and isinstance(t_rew, float) and t_rew > 0) else float("nan")

            energy_ratio = float("nan")
            energy_saving_pct = float("nan")
            if (isinstance(em_orig, float) and isinstance(em_rew, float) and em_rew > 0.0 and em_orig > 0.0):
                energy_ratio = em_orig / em_rew
                energy_saving_pct = (em_orig - em_rew) / em_orig * 100.0

            w_res.writerow([
                "webshopdb",
                OLLAMA_MODEL,
                PROMPT_TECHNIQUE,
                i,
                fmt_float(t_orig),
                fmt_float(exec_ms_o),
                fmt_float(plan_ms_o),
                buffers_o if buffers_o is not None else "",
                fmt_float(t_rew),
                fmt_float(exec_ms_r),
                fmt_float(plan_ms_r),
                buffers_r if buffers_r is not None else "",
                f"{em_orig:.10f}", f"{em_rew:.10f}",
                fmt_float(speedup),
                same_count, same_sig,
                original.replace("\n", " ").strip(),
                (rewritten or "").replace("\n", " ").strip(),
            ])

            w_em.writerow([
                "webshopdb",
                OLLAMA_MODEL,
                PROMPT_TECHNIQUE,
                i,
                f"{em_orig:.10f}",
                f"{em_rew:.10f}",
                fmt_float(energy_ratio),
                fmt_pct(energy_saving_pct),
            ])
            

            print(f"Query {i} processada. Speedup: {fmt_float(speedup)}x | CO2 Saving: {fmt_pct(energy_saving_pct)}%")

if __name__ == "__main__":
    main()
