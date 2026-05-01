from __future__ import annotations

from pathlib import Path

import duckdb


def create_ecommerce(path: Path) -> None:
    conn = duckdb.connect(str(path))
    conn.execute(
        """
        CREATE TABLE customers (
          customer_id INTEGER,
          name VARCHAR,
          signup_date DATE,
          status VARCHAR
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE orders (
          order_id INTEGER,
          customer_id INTEGER,
          order_date DATE,
          amount DOUBLE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE payments (
          payment_id INTEGER,
          customer_id INTEGER,
          payment_date DATE,
          status VARCHAR,
          amount DOUBLE
        );
        """
    )
    conn.execute(
        """
        INSERT INTO customers VALUES
        (1, 'Alice', '2024-01-01', 'active'),
        (2, 'Bob', '2024-02-01', 'active'),
        (3, 'Cara', '2024-03-01', 'inactive');
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES
        (101, 1, '2024-04-01', 120.0),
        (102, 2, '2024-04-06', 80.0),
        (103, 1, '2024-04-08', 50.0);
        """
    )
    conn.execute(
        """
        INSERT INTO payments VALUES
        (201, 1, '2024-04-01', 'paid', 120.0),
        (202, 2, '2024-04-06', 'failed', 80.0),
        (203, 1, '2024-04-08', 'paid', 50.0);
        """
    )
    conn.close()


def main() -> None:
    target_dir = Path("sample_data")
    target_dir.mkdir(parents=True, exist_ok=True)
    create_ecommerce(target_dir / "ecommerce.duckdb")


if __name__ == "__main__":
    main()

