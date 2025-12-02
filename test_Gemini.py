import os
import time
import csv
import json
import hashlib
import psycopg2
from google import genai   

# CONFIG

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

PROMPT_TECHNIQUE = "zero-shot"   # zero-shot | few-shot | chain-of-thought

QUERIES_FILE = "queries.txt"
RESULTS_CSV = "results_gemini.csv"


# CONFIG Gemini API v2

API_KEY = os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Defina GEMINI_API_KEY no ambiente!")

client = genai.Client(api_key=API_KEY)

GEMINI_MODEL = "gemini-2.5-flash"

# PROMPTS

PROMPT_INSTRUCTION = (
    "Rewrite the input SQL query to produce an equivalent query that can be executed "
    "on a PostgreSQL database with decreased latency. Return ONLY the SQL."
)

PROMPT_EXAMPLE = """Example Input:
select ... from t1 where t1.a=(select avg(a) from t3 where t1.b=t3.b);

Example Output:
select ... from t1 
inner join (
    select avg(a) avg, t3.b from t3 group by t3.b
) as t3
on (t1.a = avg and t1.b = t3.b);
"""

def build_zero_shot_prompt(original_sql, schema_hint=""):
    return (
        f"{PROMPT_INSTRUCTION}\n"
        f"Schema: {schema_hint}\n"
        f"Input SQL:\n{original_sql}\n"
        "Output SQL:"
    )

def build_few_shot_prompt(original_sql, schema_hint=""):
    return (
        f"{PROMPT_INSTRUCTION}\n"
        f"{PROMPT_EXAMPLE}\n"
        f"Schema: {schema_hint}\n"
        f"Input SQL:\n{original_sql}\n"
        "Output SQL:"
    )

def build_chain_of_thought_prompt(original_sql, schema_hint=""):
    return (
        f"{PROMPT_INSTRUCTION}\n"
        f"Schema: {schema_hint}\n"
        f"Input SQL:\n{original_sql}\n"
        "First explain the reasoning step-by-step, then output ONLY the rewritten SQL."
    )



# REWRITE SQL (Gemini API v2)

def rewrite_sql_via_gemini(original_sql: str, schema_hint: str = "") -> str:
    
    if PROMPT_TECHNIQUE == "zero-shot":
        prompt = build_zero_shot_prompt(original_sql, schema_hint)

    elif PROMPT_TECHNIQUE == "few-shot":
        prompt = build_few_shot_prompt(original_sql, schema_hint)

    elif PROMPT_TECHNIQUE == "chain-of-thought":
        prompt = build_chain_of_thought_prompt(original_sql, schema_hint)

    else:
        raise ValueError(f"Técnica desconhecida: {PROMPT_TECHNIQUE}")

    # Tentativas
    retries = 0
    while retries < 5:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,         
                contents=prompt
            )

            text = response.text.strip()

            # --- limpeza de markdown ---
            text = text.replace("```sql", "").replace("```", "").strip()

            # --- chain-of-thought: extrair somente o SQL ---
            if PROMPT_TECHNIQUE == "chain-of-thought":
                lines = [
                    l for l in text.splitlines()
                    if l.strip().upper().startswith(("SELECT", "WITH"))
                ]
                if lines:
                    return "\n".join(lines).strip()

            return text  # retorno padrão

        except Exception as e:
            print("Gemini error:", e)
            retries += 1
            time.sleep(2)

    raise RuntimeError("Gemini falhou após várias tentativas.")




#  POSTGRES

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
    dt = (time.time() - t0) * 1000.0
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


#  LEITURA DE QUERIES

def read_queries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        blob = f.read()
    blocks = [b.strip() for b in blob.split("\n\n") if b.strip() and not b.strip().startswith("--")]
    return blocks



# MAIN

def main():
    queries = read_queries(QUERIES_FILE)
    first = not os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if first:
            w.writerow([
                "db", "query_id",
                "original_ms", "execution_ms_original", "planning_ms_original",
                "rewritten_ms", "execution_ms_rewritten", "planning_ms_rewritten",
                "speedup", "same_rowcount", "same_signature",
                "original_sql", "rewritten_sql"
            ])

        for i, original in enumerate(queries, start=1):
            print(f"\n=== QUERY {i} ===")

            rewritten = rewrite_sql_via_gemini(original, SCHEMA_HINT)

            try:
                rows_o, t_o = run_timed(original)
                ej_o, p_o, e_o, _ = explain_json(original)
            except:
                rows_o, t_o, p_o, e_o = [], float("nan"), None, None

            try:
                rows_r, t_r = run_timed(rewritten)
                ej_r, p_r, e_r, _ = explain_json(rewritten)
            except:
                rows_r, t_r, p_r, e_r = [], float("nan"), None, None

            same_count = len(rows_o) == len(rows_r)
            same_sig = rows_signature(rows_o) == rows_signature(rows_r)

            speedup = (t_o / t_r) if (t_r > 0) else ""

            w.writerow([
                "webshopdb", i,
                f"{t_o:.3f}", f"{e_o}", f"{p_o}",
                f"{t_r:.3f}", f"{e_r}", f"{p_r}",
                speedup, same_count, same_sig,
                original.replace("\n", " "),
                rewritten.replace("\n", " ")
            ])

            print("OK → salvo no CSV")


if __name__ == "__main__":
    main()
