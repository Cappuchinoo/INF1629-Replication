import os, time, csv, json, hashlib
from openai import OpenAI
import psycopg2

PROMPT_MODE = "cot"

# ====== CONFIGURAÇÕES ======
PG_HOST = "localhost"
PG_PORT = 6543            # confirme com `docker ps`
PG_DB   = "sampledb"
PG_USER = "postgres"
PG_PASS = "postgres"

OPENAI_MODEL = "gpt-4o"    # ou "gpt-4.1" / "gpt-4o" etc.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QUERIES_FILE = "dbgpt_exp/queries.txt"
RESULTS_CSV  = "dbgpt_exp/results.csv"

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

    # 3) Fallback: retornar tudo
    return cleaned.strip()

# ====== OPENAI ======
def rewrite_sql_via_llm(
    original_sql: str,
    schema_hint: str = "",
    prompt_mode: str = None,
) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Defina OPENAI_API_KEY no ambiente.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    if prompt_mode is None:
        prompt_mode = PROMPT_MODE

    if prompt_mode == "zero_shot":
        user_prompt = build_zero_shot_prompt(original_sql, schema_hint)
    elif prompt_mode in ("cot", "chain_of_thought"):
        user_prompt = build_chain_of_thought_prompt(original_sql, schema_hint)
    else:
        user_prompt = build_few_shot_prompt(original_sql, schema_hint)

    # System message precisa varia para cot
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

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        # não passar temperature para evitar erro de unsupported_value
    )

    raw = resp.choices[0].message.content
    sql = extract_sql_from_response(raw)


    return sql

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
                "db","query_id",
                "original_ms","execution_ms_original","planning_ms_original","buffers_plan_original",
                "rewritten_ms","execution_ms_rewritten","planning_ms_rewritten","buffers_plan_rewritten",
                "speedup","buffers_ratio","same_rowcount","same_signature",
                "original_sql","rewritten_sql"
            ])

        for i, original in enumerate(queries, start=1):
            print(f"\n=== Query {i} ===")
            
            rewritten = rewrite_sql_via_llm(original)

            # Original
            try:
                rows_orig, t_orig = run_timed(original)
                ej_orig, plan_ms_o, exec_ms_o, plan_o = explain_json(original)
            except Exception as e:
                print("Erro original:", e)
                rows_orig, t_orig, ej_orig, plan_ms_o, exec_ms_o, plan_o = [], float("nan"), {}, None, None, {}
            
            # Reescrita
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

            w.writerow([
                PG_DB, i,
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
                same_count, same_sig,
                original.replace("\n"," ").strip(),
                rewritten.replace("\n"," ").strip()
            ])

            print("OK → linha gravada em results.csv")

if __name__ == "__main__":
    main()
