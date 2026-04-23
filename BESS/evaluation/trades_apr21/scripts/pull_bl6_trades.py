"""Pull BL6 confirmed trades for 2026-04-21 from production EMS DB.

DA  : isot_dt_evaluation_blocks  (evaluated/settled DA positions, QH amounts in MW)
IDM : isot_vdt_trade, tradeday='2026-04-21'  (same-session intraday fills, amounts in MW)

VPN to 10.4.1.66 must be active.
"""

import os
import psycopg2
import pandas as pd

DB = dict(host="10.4.1.66", port=5432, dbname="DB_EMS", user="pnacek", password="Kapitan4478")
TRADE_DATE = "2026-04-21"
BLOCK = "BL6"

SQL_DA = """
SELECT periodfrom, valuetime, tradetype, amount, price
FROM db_ems.isot_dt_evaluation_blocks
WHERE tradecomment = %(block)s
  AND valuetime::date = %(day)s
ORDER BY periodfrom, tradetype;
"""

SQL_IDM = """
SELECT tradetype, periodfrom, amount, price, deliverydur, datetimemodify
FROM db_ems.isot_vdt_trade
WHERE tradecomment = %(block)s
  AND tradeday::date = %(day)s
ORDER BY periodfrom, tradetype;
"""


def main():
    params = {"block": BLOCK, "day": TRADE_DATE}

    print(f"[*] Connecting to DB_EMS ...")
    conn = psycopg2.connect(**DB)
    try:
        da  = pd.read_sql(SQL_DA,  conn, params=params)
        idm = pd.read_sql(SQL_IDM, conn, params=params)
    finally:
        conn.close()

    print(f"[+] DA rows : {len(da)} | IDM trades : {len(idm)}")

    # EUR conversion: amounts are in MW
    da["eur"]  = da["amount"]  * da["price"] * 0.25               # MW × 0.25h × EUR/MWh
    idm["eur"] = idm["amount"] * idm["price"] * idm["deliverydur"] / 60.0

    da_buy  = da[da["tradetype"] == "N"]
    da_sell = da[da["tradetype"] == "P"]
    id_buy  = idm[idm["tradetype"] == "N"]
    id_sell = idm[idm["tradetype"] == "P"]

    da_buy_eur  = da_buy["eur"].sum()
    da_sell_eur = da_sell["eur"].sum()
    id_buy_eur  = id_buy["eur"].sum()
    id_sell_eur = id_sell["eur"].sum()

    print(f"\n--- {BLOCK} on {TRADE_DATE} ---")
    print(f"  DA+VDA:   buy {da_buy_eur:>8.2f} EUR | sell {da_sell_eur:>8.2f} EUR | margin {da_sell_eur - da_buy_eur:>8.2f} EUR")
    print(f"  Intraday: buy {id_buy_eur:>8.2f} EUR | sell {id_sell_eur:>8.2f} EUR | margin {id_sell_eur - id_buy_eur:>8.2f} EUR")
    print(f"  Total gross: {da_sell_eur - da_buy_eur + id_sell_eur - id_buy_eur:.2f} EUR")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    da.to_csv(os.path.join(data_dir,  f"bl6_da_{TRADE_DATE}.csv"),  index=False)
    idm.to_csv(os.path.join(data_dir, f"bl6_idm_{TRADE_DATE}.csv"), index=False)
    print(f"\n[+] Saved to data/")


if __name__ == "__main__":
    main()
