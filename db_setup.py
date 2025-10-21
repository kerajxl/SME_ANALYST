# db_setup.py
# Generator dla case'u HD vs Endur – bez zapisu do SQLite (zwraca DataFrame'y)
# Tabele:
# - hd_readings
# - product_dict
# - endur_dump

import random
import pandas as pd
from datetime import date, timedelta
from typing import Tuple

# Daty
TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)
PERIOD_START = date(2022, 1, 1)
PERIOD_END = date(2025, 9, 30)  # ostatni miesiąc włącznie

# Konfiguracja
N_PODS = 500
DUAL_MEDIUM_FRACTION = 0.33   # ~1/3 POD-ów ma oba media
BASE_MEDIUM_POWER_WEIGHT = 0.6 # częściej power jako bazowe

# Słownik produktów (~100 w 4 kategoriach)
CATS = ["GAS_FIX", "GAS_SPOT", "POWER_FIX", "POWER_SPOT"]
PER_CAT = 25  # 4 * 25 = 100

def month_range(start: date, end: date):
    y, m = start.year, start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        yield date(y, m, 1)
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1

def last_day_of_month(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    return date(d.year + (d.month // 12), ((d.month % 12) + 1), 1) - timedelta(days=1)

MONTHS = list(month_range(PERIOD_START, PERIOD_END))  # powinno być 45 miesięcy
N_MONTHS = len(MONTHS)

def make_id(prefix: str, n: int, width: int = 6) -> str:
    return f"{prefix}{str(n).zfill(width)}"

def partition_contracts(total_months: int = 45):
    # 70% -> 2 kontrakty (każdy <= 24), 30% -> 3 kontrakty
    if random.random() < 0.7:
        first = random.randint(21, 24)
        second = total_months - first
        return [first, second]
    else:
        a = random.randint(14, 16)
        b = random.randint(14, 16)
        c = total_months - a - b
        if c < 1:
            c = 1
            if a > 14:
                a -= 1
            else:
                b -= 1
        return [a, b, c]

def generate_products():
    product_rows = []
    cat_to_ids = {c: [] for c in CATS}
    for idx, cat in enumerate(CATS):
        for j in range(1, PER_CAT + 1):
            pid = f"PRD{idx*PER_CAT + j:03d}"
            if cat == "GAS_FIX":
                pname = f"gas_fix_{j:03d}_premium" if j % 2 == 0 else f"ultra_{j:03d}_gas_fix"
            elif cat == "GAS_SPOT":
                pname = f"gas_spot_{j:03d}_indexed" if j % 2 == 0 else f"promo_{j:03d}_gas_spot"
            elif cat == "POWER_FIX":
                pname = f"power_fix_{j:03d}_pro" if j % 2 == 0 else f"smart_{j:03d}_power_fix"
            else:  # POWER_SPOT
                pname = f"power_spot_{j:03d}_float" if j % 2 == 0 else f"dynamic_{j:03d}_power_spot"
            product_rows.append({"product_id": pid, "product_name": pname})
            cat_to_ids[cat].append(pid)

    product_df = pd.DataFrame(product_rows)
    pid_to_cat = {pid: cat for cat, ids in cat_to_ids.items() for pid in ids}
    pid_to_medium = {pid: (2 if cat.startswith('GAS') else 1) for pid, cat in pid_to_cat.items()}
    return product_df, cat_to_ids, pid_to_cat, pid_to_medium

def generate_hd_dataset(cat_to_ids, pid_to_medium):
    # master data
    N_CUSTOMERS = 350
    N_ADDR = 500
    customers = [make_id("CUST", i+1) for i in range(N_CUSTOMERS)]
    addresses = [make_id("ADDR", i+1) for i in range(N_ADDR)]

    pod_to_customer = {}
    pod_to_address = {}
    for i in range(1, N_PODS+1):
        pod = make_id("POD", i)
        pod_to_customer[pod] = random.choice(customers)
        pod_to_address[pod] = random.choice(addresses)

    all_pods = [make_id("POD", i) for i in range(1, N_PODS+1)]
    random.shuffle(all_pods)
    num_dual = int(round(N_PODS * DUAL_MEDIUM_FRACTION))
    dual_set = set(all_pods[:num_dual])

    rows = []
    contract_counter = 1

    for pod in all_pods:
        base_medium = 1 if random.random() < BASE_MEDIUM_POWER_WEIGHT else 2
        mediums = [base_medium]
        if pod in dual_set:
            mediums.append(2 if base_medium == 1 else 1)

        for medium in mediums:
            parts = partition_contracts(N_MONTHS)
            current_cat = random.choice(["POWER_FIX", "POWER_SPOT"]) if medium == 1 else random.choice(["GAS_FIX", "GAS_SPOT"])
            prev_pid = None
            month_idx = 0
            for part_len in parts:
                if prev_pid is None:
                    pid = random.choice(cat_to_ids[current_cat])
                else:
                    if random.random() < 0.65:
                        if random.random() < 0.5:
                            current_cat = random.choice(["POWER_FIX", "POWER_SPOT"]) if medium == 1 else random.choice(["GAS_FIX", "GAS_SPOT"])
                        options = [x for x in cat_to_ids[current_cat] if x != prev_pid]
                        pid = random.choice(options) if options else prev_pid
                    else:
                        pid = prev_pid

                # spójność medium vs produkt
                if pid_to_medium[pid] != medium:
                    raise ValueError("Product medium mismatch")

                ctr_id = make_id("CTR", contract_counter); contract_counter += 1

                for _ in range(part_len):
                    m_first = MONTHS[month_idx]
                    m_last = last_day_of_month(m_first)
                    forecast = round(random.random() * 35.0, 3)
                    factor = random.uniform(0.6, 1.3)
                    real = round(forecast * factor, 3)
                    rows.append({
                        "loading_date": TODAY.isoformat(),
                        "pod_header_id": pod,
                        "customer_id": pod_to_customer[pod],
                        "adres_id": pod_to_address[pod],
                        "contrct_id": ctr_id,
                        "product_id": pid,
                        "medium_id": medium,  # 1=power, 2=gas
                        "usage_from_date": m_first.isoformat(),
                        "usage_to_date": m_last.isoformat(),
                        "forecast_usage_mwh": forecast,
                        "real_usage_mwh": real,
                    })
                    month_idx += 1
                prev_pid = pid

    hd_df = pd.DataFrame(rows)
    # dokładnie 100 wierszy wczoraj
    if len(hd_df) >= 100:
        idxs = hd_df.sample(n=100, random_state=777).index
        hd_df.loc[idxs, "loading_date"] = YESTERDAY.isoformat()
    return hd_df

def make_endur_dump(hd_df: pd.DataFrame, pid_to_cat: dict) -> pd.DataFrame:
    # tylko dzisiejsze wiersze HD
    hd_today = hd_df[hd_df["loading_date"] == TODAY.isoformat()].copy()
    hd_today["delivery_month"] = hd_today["usage_from_date"].str.slice(0, 7)
    hd_today["endur_product"] = hd_today["product_id"].map(pid_to_cat)
    hd_today["BOOK"] = hd_today["endur_product"].apply(lambda c: "AXPL_SME_GAS" if c.startswith("GAS") else "AXPL_SME_POWER")

    endur_agg = (hd_today.groupby(["endur_product", "BOOK", "delivery_month"], as_index=False)["real_usage_mwh"]
                 .sum()
                 .rename(columns={"real_usage_mwh": "volume"}))
    endur_agg["reporting_date"] = TODAY.isoformat()
    endur_agg = endur_agg[["reporting_date", "BOOK", "endur_product", "delivery_month", "volume"]]
    return endur_agg

# w db_setup.py

def setup_database(seed: int | None = None, shuffle: bool = True):
    if seed is not None:
        random.seed(seed)

    product_df, cat_to_ids, pid_to_cat, pid_to_medium = generate_products()
    hd_df = generate_hd_dataset(cat_to_ids, pid_to_medium)
    endur_df = make_endur_dump(hd_df, pid_to_cat)

    if shuffle:
        # losowa kolejność wierszy (bez deterministycznego random_state)
        hd_df = hd_df.sample(frac=1, random_state=None).reset_index(drop=True)
        product_df = product_df.sample(frac=1, random_state=None).reset_index(drop=True)
        endur_df = endur_df.sample(frac=1, random_state=None).reset_index(drop=True)

    return hd_df, product_df, endur_df


# Opcjonalnie: kompatybilność z wcześniejszym wzorcem "modify_random_usage"
def modify_random_usage(df: pd.DataFrame):
    """
    Dla kompatybilności z poprzednim repo:
    Przyjmuje DataFrame (tu: hd_readings), losowo modyfikuje real_usage_mwh w 1 wierszu.
    Zwraca (random_index, random_customer_id, old_value, new_value, modified_record).
    """
    if df.empty:
        return None, None, None, None, None
    idx = df.sample(n=1, random_state=None).index[0]
    row = df.loc[idx].copy()
    old_val = row["real_usage_mwh"]
    factor = random.uniform(0.8, 1.2)
    new_val = round(old_val * factor, 3)
    df.loc[idx, "real_usage_mwh"] = new_val
    return int(idx), row.get("customer_id"), float(old_val), float(new_val), df.loc[idx].to_dict()
