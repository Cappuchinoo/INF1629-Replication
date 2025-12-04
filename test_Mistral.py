import os
import time
import csv
import json
import hashlib
import re
import logging

import psycopg2
from mistralai import Mistral
from codecarbon import EmissionsTracker

# ===== DEBUG FLAGS =====
DEBUG_LLM = False       # mostra prompt, RAW da Mistral e SQL limpa
DEBUG_ENERGY = True    # mostra [ENERGY] no terminal

# Silenciar warnings do CodeCarbon
logging.getLogger("codecarbon").setLevel(logging.ERROR)

# ===== CONFIG =====
PG_URL = (
    "postgresql://neondb_owner:npg_xAOBhf4MkK5C@ep-plain-sunset-aervpg8i-pooler.c-2."
    "us-east-2.aws.neon.tech/webshopdb?sslmode=require&channel_binding=require"
)

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

# Escolha: "zero-shot", "few-shot", "chain-of-thought"
PROMPT_TECHNIQUE = "chain-of-thought"

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QUERIES_FILE = os.path.join(BASE_DIR, "dbgpt_exp", "queries.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "dbgpt_exp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    f"results_mistral_{PROMPT_TECHNIQUE}.csv"
)

# ===== MISTRAL =====
MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise RuntimeError("Defina MISTRAL_API_KEY com sua chave.")

mistral_client = Mistral(api_key=MISTRAL_API_KEY)

PROMPT_INSTRUCTION = (
    "Rewrite the input SQL query to produce an equivalent query that can be executed "
    "on a PostgreSQL database with decreased latency. "
    "Return ONLY the final SQL query, with NO explanations, NO comments and NO markdown. "
    "The answer must start directly with SELECT, WITH, INSERT, UPDATE, DELETE, CREATE or ALTER."
)

PROMPT_EXAMPLE = """Example Input:
select ... from t1 where t1.a=(select avg(a) from t3 where t1.b=t3.b);

Example Output:
select ... from t1 inner join (
    select avg(a) avg, t3.b from t3 group by t3.b
) as t3
on (t1.a = avg and t1.b = t3.b);
"""

# ===== DB FUNCS =====


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

# ===== PROMPTS =====


def build_zero_shot_prompt(original_sql, schema_hint=""):
    return (
        f"{PROMPT_INSTRUCTION}\n"
        f"{'Schema: ' + schema_hint if schema_hint else ''}\n"
        f"Input SQL:\n{original_sql}\nOutput SQL:"
    )


def build_few_shot_prompt(original_sql, schema_hint=""):
    return (
        f"{PROMPT_INSTRUCTION}\n{PROMPT_EXAMPLE}\n"
        f"{'Schema: ' + schema_hint if schema_hint else ''}\n"
        f"Input SQL:\n{original_sql}\nOutput SQL:"
    )


def build_chain_of_thought_prompt(original_sql, schema_hint=""):
    return (
        f"{PROMPT_INSTRUCTION}\n"
        f"{'Schema: ' + schema_hint if schema_hint else ''}\n"
        f"Input SQL:\n{original_sql}\n"
        "First, explain step-by-step how to optimize; then provide ONLY the final SQL."
    )

# ===== LIMPEZA DO LLM =====


def extract_sql(text: str) -> str:
    if not text:
        return ""

    m = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"```(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    upper = text.upper()
    for kw in ("WITH", "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER"):
        idx = upper.find(kw)
        if idx != -1:
            return text[idx:].strip()

    return text.strip()

# ===== LLM CALL =====


def rewrite_sql_via_mistral(original_sql: str, schema_hint: str = "") -> str:

    if PROMPT_TECHNIQUE == "zero-shot":
        user_content = build_zero_shot_prompt(original_sql, schema_hint)
    elif PROMPT_TECHNIQUE == "few-shot":
        user_content = build_few_shot_prompt(original_sql, schema_hint)
    else:
        user_content = build_chain_of_thought_prompt(original_sql, schema_hint)

    if DEBUG_LLM:
        print("\n=== ORIGINAL SQL ===")
        print(original_sql)
        print("=== FIM ORIGINAL SQL ===\n")

        print("=== PROMPT ENVIADO PARA O LLM ===")
        print(user_content)
        print("=== FIM PROMPT ENVIADO ===\n")

    for _ in range(5):
        try:
            response = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.7,
                max_tokens=1024,
            )

            raw = response.choices[0].message.content or ""

            if DEBUG_LLM:
                print("=== RAW LLM RESPONSE ===")
                print(raw)
                print("=== FIM RAW LLM RESPONSE ===\n")

            cleaned = extract_sql(raw)

            if DEBUG_LLM:
                print("=== CLEANED REWRITTEN SQL ===")
                print(cleaned)
                print("=== FIM CLEANED REWRITTEN SQL ===\n")

            return cleaned if cleaned else raw.strip()

        except Exception as e:
            print("Erro chamar LLM:", e)
            time.sleep(5)

    return ""

# ===== READ QUERIES =====


def read_queries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        blob = f.read()
    return [
        b.strip()
        for b in blob.split("\n\n")
        if b.strip() and not b.strip().startswith("--")
    ]

# ===== ENERGY WRAPPER =====


def run_query_with_energy(sql: str, tag: str):
    tracker = EmissionsTracker(
        project_name=f"mistral_{PROMPT_TECHNIQUE}_{tag}",
        output_dir=OUTPUT_DIR,
        output_file=f"emissions_mistral_{PROMPT_TECHNIQUE}.csv",
        log_level="error",
        measure_power_secs=1,
        save_to_file=True,
    )

    tracker.start()
    try:
        rows, ms = run_timed(sql)
        emissions = tracker.stop()
        if DEBUG_ENERGY:
            print(f"[ENERGY] {tag}: {emissions:.8f} kg CO2e")
    except Exception:
        tracker.stop()
        raise

    return rows, ms, emissions

# ===== HELPERS =====


def fmt_float(v):
    if isinstance(v, float) and not (v != v):
        return f"{v:.3f}"
    return "NaN"

# ===== MAIN =====


def main():

    print("Lendo queries de:", QUERIES_FILE)
    queries = read_queries(QUERIES_FILE)

    print("CSV será gravado em:", RESULTS_CSV)
    first_write = not os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        if first_write:
            w.writerow([
                "db", "llm", "prompt_technique", "query_id",
                "original_ms", "execution_ms_original", "planning_ms_original", "buffers_plan_original",
                "rewritten_ms", "execution_ms_rewritten", "planning_ms_rewritten", "buffers_plan_rewritten",
                "emissions_original", "emissions_rewritten",
                "speedup", "same_rowcount", "same_signature",
                "original_sql", "rewritten_sql"
            ])

        for i, original in enumerate(queries, start=1):
            print(f"\n=== Query {i} ({PROMPT_TECHNIQUE}, {MISTRAL_MODEL}) ===")

            rewritten = rewrite_sql_via_mistral(original)

            # ORIGINAL
            try:
                rows_orig, t_orig, em_orig = run_query_with_energy(
                    original, "original")
                ej_orig, plan_ms_o, exec_ms_o, plan_o = explain_json(original)
            except Exception as e:
                print("Erro original:", e)
                rows_orig, t_orig = [], float("nan")
                plan_ms_o = exec_ms_o = None
                plan_o = {}
                em_orig = float("nan")

            # REWRITTEN
            try:
                rows_rew, t_rew, em_rew = run_query_with_energy(
                    rewritten, "rewritten")
                ej_rew, plan_ms_r, exec_ms_r, plan_r = explain_json(rewritten)
            except Exception as e:
                print("Erro reescrita:", e)
                rows_rew, t_rew = [], float("nan")
                plan_ms_r = exec_ms_r = None
                plan_r = {}
                em_rew = float("nan")

            # VALIDATION
            same_count = (len(rows_orig) == len(rows_rew))

            sig_o = rows_signature(rows_orig)
            sig_r = rows_signature(rows_rew)
            same_sig = (sig_o == sig_r)

            buffers_o = plan_o.get("Shared Hit Blocks") if isinstance(
                plan_o, dict) else None
            buffers_r = plan_r.get("Shared Hit Blocks") if isinstance(
                plan_r, dict) else None

            speedup = (
                t_orig / t_rew
                if isinstance(t_orig, float)
                and isinstance(t_rew, float)
                and t_rew > 0
                else float("nan")
            )

            # WRITE CSV
            w.writerow([
                "webshopdb",
                MISTRAL_MODEL,
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
                em_orig, em_rew,
                fmt_float(speedup),
                same_count, same_sig,
                original.replace("\n", " ").strip(),
                (rewritten or "").replace("\n", " ").strip(),
            ])

            print(
                f"[Q{i}] orig={fmt_float(t_orig)} ms, em_orig={fmt_float(em_orig)} | "
                f"rew={fmt_float(t_rew)} ms, em_rew={fmt_float(em_rew)} | "
                f"speedup={fmt_float(speedup)} | same_sig={same_sig}"
            )

            print("OK → linha gravada em", RESULTS_CSV)


# ===== RUN =====
if __name__ == "__main__":
    main()
