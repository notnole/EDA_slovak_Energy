"""Read a vector over a time range and print the head.

Assumes SSH tunnel is up:
    ssh -L8080:10.100.0.70:8080 noel@greenbat1.vps.wbsprt.com
"""

from vectord import VectordClient


def main():
    client = VectordClient()
    df = client.read_df(
        vector="F.B.Odchylka",
        start="2026-04-20T00:00:00Z",
        end="2026-04-20T12:00:00Z",
    )
    print(f"[+] {len(df)} points")
    print(df.head())


if __name__ == "__main__":
    main()
