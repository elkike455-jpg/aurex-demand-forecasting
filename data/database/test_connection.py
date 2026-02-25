from db_connection import get_connection

def main():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        print("✅ Connected! SELECT 1 =", cur.fetchone()[0])
        cur.close()
        conn.close()
    except Exception as e:
        print("❌ Connection failed:", e)

if __name__ == "__main__":
    main()