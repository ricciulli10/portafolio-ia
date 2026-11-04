"""Portfolio analysis in plain SQL over SQLite. Run: python -m credit_risk.sql_analysis"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd

from .data import DATA_PATH, REPORTS

QUERIES = {
    "default_rate_by_checking": """
        SELECT checking_status,
               COUNT(*) AS loans,
               ROUND(AVG(class = 'bad'), 3) AS default_rate
        FROM loans
        GROUP BY checking_status
        ORDER BY default_rate DESC
    """,
    "riskiest_purposes_with_volume": """
        WITH by_purpose AS (
            SELECT purpose,
                   COUNT(*) AS loans,
                   AVG(class = 'bad') AS default_rate,
                   SUM(credit_amount) AS exposure
            FROM loans
            GROUP BY purpose
            HAVING COUNT(*) >= 30
        )
        SELECT purpose, loans, ROUND(default_rate, 3) AS default_rate, exposure,
               ROUND(100.0 * exposure / SUM(exposure) OVER (), 1) AS exposure_share,
               RANK() OVER (ORDER BY default_rate DESC) AS risk_rank
        FROM by_purpose
        ORDER BY risk_rank
    """,
    "default_rate_by_duration_quartile": """
        WITH q AS (
            SELECT class, duration, NTILE(4) OVER (ORDER BY duration) AS quartile FROM loans
        )
        SELECT quartile, MIN(duration) AS min_months, MAX(duration) AS max_months,
               COUNT(*) AS loans, ROUND(AVG(class = 'bad'), 3) AS default_rate
        FROM q
        GROUP BY quartile
        ORDER BY quartile
    """,
}


def run() -> dict[str, pd.DataFrame]:
    with sqlite3.connect(":memory:") as db:
        pd.read_csv(DATA_PATH).to_sql("loans", db, index=False)
        return {name: pd.read_sql(query, db) for name, query in QUERIES.items()}


def main() -> None:
    results = run()
    for name, frame in results.items():
        print(f"\n{name}\n{frame.to_string(index=False)}")
    report = {name: frame.to_dict("records") for name, frame in results.items()}
    REPORTS.joinpath("sql_insights.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
