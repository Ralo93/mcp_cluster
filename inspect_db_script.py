import psycopg2

def inspect_columns(db_url, table="speeches"):
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s;
            """, (table,))
            cols = cur.fetchall()
            print(f"📋 Columns in '{table}':")
            for col in cols:
                print("-", col[0])

if __name__ == "__main__":
    db_url = "postgresql://postgres:postgres@localhost:5432/cluster"
    inspect_columns(db_url, table="speeches")
