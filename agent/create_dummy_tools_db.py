# agent/create_dummy_tools_db.py

import sqlite3

def create_dummy_guidelines_db(path: str = "data/acc_aha_guidelines_dummy.db"):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS guidelines
                 (id INTEGER PRIMARY KEY, condition TEXT, recommendation TEXT, risk_level TEXT)''')
    dummy_data = [
        ("hypotension", "Start fluid bolus if MAP <65, consider vasopressor if refractory", "critical"),
        ("tachycardia", "Assess for sepsis, consider beta-blocker if stable", "high"),
    ]
    c.executemany("INSERT INTO guidelines (condition, recommendation, risk_level) VALUES (?, ?, ?)", dummy_data)
    conn.commit()
    conn.close()
    print(f"Dummy ACC/AHA DB created at {path}")

def create_dummy_drugbank_db(path: str = "data/drugbank_dummy.db"):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions
                 (drug1 TEXT, drug2 TEXT, severity TEXT, description TEXT)''')
    dummy_data = [
        ("aspirin", "warfarin", "major", "Increased bleeding risk"),
        ("metformin", "cimetidine", "moderate", "Risk of lactic acidosis"),
    ]
    c.executemany("INSERT INTO interactions VALUES (?, ?, ?, ?)", dummy_data)
    conn.commit()
    conn.close()
    print(f"Dummy DrugBank DB created at {path}")

if __name__ == "__main__":
    create_dummy_guidelines_db()
    create_dummy_drugbank_db()
