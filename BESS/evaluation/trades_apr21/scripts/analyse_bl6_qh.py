"""Per-QH evaluation of BL6 for 2026-04-21.

Mirrors the production evaluation logic exactly:
  - DA amounts in MW, converted to EUR via x0.25
  - IDM amounts in MW, converted to EUR via x deliverydur/60
  - 60-min IDM products expand to 4 QHs (periodfrom = hour index)
  - Limits: charge 1.2 MW, discharge 2.5 MW per QH, capacity 5.0 MWh
"""

import os
import pandas as pd

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
TRADE_DATE = "2026-04-21"
CHARGE_LIMIT    = 1.2    # MW
DISCHARGE_LIMIT = 2.5    # MW
CAPACITY        = 5.0    # MWh

pd.set_option("display.width", 260)
pd.set_option("display.float_format", "{:.3f}".format)


def load():
    da  = pd.read_csv(os.path.join(DATA_DIR, f"bl6_da_{TRADE_DATE}.csv"),  parse_dates=["valuetime"])
    idm = pd.read_csv(os.path.join(DATA_DIR, f"bl6_idm_{TRADE_DATE}.csv"), parse_dates=["datetimemodify"])
    return da, idm


def build_qh_grid(da, idm):
    """Return two arrays [96] for qh_buy_mw and qh_sell_mw."""
    buy_mw  = [0.0] * 96
    sell_mw = [0.0] * 96

    for _, r in da.iterrows():
        qh  = int(r["periodfrom"])
        mw  = float(r["amount"])
        if r["tradetype"] == "N":
            buy_mw[qh]  += mw
        else:
            sell_mw[qh] += mw

    for _, r in idm.iterrows():
        pf  = int(r["periodfrom"])
        amt = float(r["amount"])
        dur = int(r["deliverydur"])
        qhs = [pf] if dur == 15 else list(range(pf * 4, pf * 4 + 4))
        for qh in qhs:
            if 0 <= qh < 96:
                if r["tradetype"] == "N":
                    buy_mw[qh]  += amt
                else:
                    sell_mw[qh] += amt

    return buy_mw, sell_mw


def qh_to_time(qh):
    h, m = divmod(qh * 15, 60)
    return f"{h:02d}:{m:02d}"


def main():
    da, idm = load()

    # --- P&L ---
    da["eur"]  = da["amount"]  * da["price"] * 0.25
    idm["eur"] = idm["amount"] * idm["price"] * idm["deliverydur"] / 60.0

    da_buy_eur  = da[da["tradetype"] == "N"]["eur"].sum()
    da_sell_eur = da[da["tradetype"] == "P"]["eur"].sum()
    id_buy_eur  = idm[idm["tradetype"] == "N"]["eur"].sum()
    id_sell_eur = idm[idm["tradetype"] == "P"]["eur"].sum()
    da_margin   = da_sell_eur - da_buy_eur
    id_margin   = id_sell_eur - id_buy_eur

    # --- QH grid ---
    buy_mw, sell_mw = build_qh_grid(da, idm)

    rows = []
    soc  = CAPACITY           # start full
    phys_charge = phys_discharge = 0.0

    for qh in range(96):
        bv = buy_mw[qh]
        sv = sell_mw[qh]
        if bv == 0 and sv == 0:
            continue

        net = sv - bv           # positive = net discharge (MW)

        # physical flows capped at limits
        if net < 0:
            phys_charge    += min(abs(net), CHARGE_LIMIT) * 0.25
        else:
            phys_discharge += min(net, DISCHARGE_LIMIT) * 0.25

        soc -= net * 0.25       # MWh delta (discharge lowers SoC)

        flag = ""
        if net < 0 and abs(net) > CHARGE_LIMIT + 0.01:
            flag = f"[!] overcharge {abs(net):.2f} MW > {CHARGE_LIMIT}"
        elif net > 0 and net > DISCHARGE_LIMIT + 0.01:
            flag = f"[!] overdischarge {net:.2f} MW > {DISCHARGE_LIMIT}"

        rows.append({
            "qh":       qh,
            "time":     qh_to_time(qh),
            "buy_mw":   bv,
            "sell_mw":  sv,
            "net_mw":   net,
            "soc_mwh":  soc,
            "flag":     flag,
        })

    grid = pd.DataFrame(rows)

    print(f"\n=== BL6 QH breakdown {TRADE_DATE} ===\n")
    print(grid.to_string(index=False))

    cycles = phys_charge / CAPACITY

    print(f"\n--- P&L ---")
    print(f"  DA+VDA:   buy {da_buy_eur:>8.2f} | sell {da_sell_eur:>8.2f} | margin {da_margin:>8.2f} EUR")
    print(f"  Intraday: buy {id_buy_eur:>8.2f} | sell {id_sell_eur:>8.2f} | margin {id_margin:>8.2f} EUR")
    print(f"  Total gross: {da_margin + id_margin:.2f} EUR")
    print(f"\n--- Physical ---")
    print(f"  Charged   : {phys_charge:.3f} MWh")
    print(f"  Discharged: {phys_discharge:.3f} MWh")
    print(f"  Cycles    : {cycles:.2f}")

    anom = grid[grid["flag"] != ""]
    print(f"\n--- Anomalies ({len(anom)}) ---")
    if anom.empty:
        print("  None.")
    else:
        print(anom[["qh", "time", "buy_mw", "sell_mw", "net_mw", "flag"]].to_string(index=False))

    grid.to_csv(os.path.join(DATA_DIR, f"bl6_qh_{TRADE_DATE}.csv"), index=False)
    print(f"\n[+] Saved to data/bl6_qh_{TRADE_DATE}.csv")


if __name__ == "__main__":
    main()
