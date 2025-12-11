import csv
import math
import statistics

RESULTS_CSV = "dbgpt_exp/results.csv"

IGNORE_OUTLIERS = True

FIXED_THRESHOLD = True
MAX_SPEEDUP = 3.0
MAX_BUFFERS_RATIO = 10.0

def parse_float(x):
    try:
        if x is None:
            return math.nan
        x = x.strip()
        if x == "":
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def parse_bool(x):
    if x is None:
        return None
    x = str(x).strip().lower()
    if x in ("true", "t", "1", "yes"):
        return True
    if x in ("false", "f", "0", "no"):
        return False
    return None


def iqr_filter(values):

    q1 = statistics.quantiles(values, n=4, method="inclusive")[0]   # 25%
    q3 = statistics.quantiles(values, n=4, method="inclusive")[2]   # 75%
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return [v for v in values if lower <= v <= upper]

def domain_filter(values, max_value=None):
    if max_value is None:
        return values
    return [v for v in values if v <= max_value]


def safe_mean(xs):
    return statistics.mean(xs) if xs else math.nan


def safe_median(xs):
    return statistics.median(xs) if xs else math.nan


def safe_mode(xs):
    if not xs:
        return math.nan
    try:
        return statistics.mode(xs)
    except statistics.StatisticsError:
        return math.nan


def main():
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    speedups = []
    buffers = []
    same_counts = []
    sig_counts = []

    # coleta valores
    for row in rows:
        sr = parse_bool(row.get("same_rowcount"))
        ss = parse_bool(row.get("same_signature"))
        s  = parse_float(row.get("speedup"))
        b  = parse_float(row.get("buffers_ratio"))

        if sr is not None:
            same_counts.append(sr)
        if ss is not None:
            sig_counts.append(ss)

        # usamos speedup/buffers_ratio apenas quando há equivalência de resultados
        if sr is True:
            if not math.isnan(s):
                speedups.append(s)
            if not math.isnan(b):
                buffers.append(b)

    # ------------------------------
    # SEM FILTRO DE OUTLIERS
    # ------------------------------
    print("=== MÉTRICAS COM OUTLIERS ===\n")

    print(f"Speedup média   : {safe_mean(speedups):.4f}")
    print(f"Speedup mediana : {safe_median(speedups):.4f}")
    print(f"Speedup moda    : {safe_mode(speedups):.4f}\n")

    print(f"Buffers média   : {safe_mean(buffers):.4f}")
    print(f"Buffers mediana : {safe_median(buffers):.4f}")
    print(f"Buffers moda    : {safe_mode(buffers):.4f}\n")

    sr_true = same_counts.count(True) / len(same_counts) * 100 if same_counts else math.nan
    sig_true = sig_counts.count(True) / len(sig_counts) * 100 if sig_counts else math.nan

    print(f"% same_rowcount TRUE : {sr_true:.2f}%")
    print(f"% same_signature TRUE: {sig_true:.2f}%\n")

    # ------------------------------
    # COM FILTRO DE OUTLIERS
    # ------------------------------
    if IGNORE_OUTLIERS:
        print("\n=== MÉTRICAS SEM OUTLIERS (IQR FILTER) ===\n")
        
        if FIXED_THRESHOLD:
            speed_base = speedups
            buffers_base = buffers

            speedups_no = domain_filter(speed_base, MAX_SPEEDUP)
            buffers_no = domain_filter(buffers_base, MAX_BUFFERS_RATIO)
        else:
            speedups_no = iqr_filter(speedups)
            buffers_no = iqr_filter(buffers)
        
        


        print(f"Speedup média   : {safe_mean(speedups_no):.4f}")
        print(f"Speedup mediana : {safe_median(speedups_no):.4f}")
        print(f"Speedup moda    : {safe_mode(speedups_no):.4f}")
        print(f"(original N={len(speedups)} → filtrado N={len(speedups_no)})\n")

        print(f"Buffers média   : {safe_mean(buffers_no):.4f}")
        print(f"Buffers mediana : {safe_median(buffers_no):.4f}")
        print(f"Buffers moda    : {safe_mode(buffers_no):.4f}")
        print(f"(original N={len(buffers)} → filtrado N={len(buffers_no)})\n")

    print("\nConcluído.\n")


if __name__ == "__main__":
    main()
