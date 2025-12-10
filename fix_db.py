import sqlite3
conn = sqlite3.connect('data.db')
try:
    conn.execute('ALTER TABLE billing_plans ADD COLUMN billing_cycle TEXT DEFAULT "monthly"')
    conn.commit()
    print('Fixed billing_plans table!')
except Exception as e:
    print(f'Already fixed or error: {e}')
conn.close()
