
import streamlit as st
import pandas as pd
from streamlit.connections import BaseConnection
import sqlite3

class FinanceDataConnection(BaseConnection[sqlite3.Connection]):
    def _connect(self, **kwargs) -> sqlite3.Connection:
        db_path = kwargs.get("db_path", "finance_data.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    "Date" TEXT,
                    "Description" TEXT,
                    "Debit" REAL,
                    "Credit" REAL,
                    "Balance" REAL
                );
            """)
        return conn

    def get_all_data(self) -> pd.DataFrame:
        try:
            df = pd.read_sql("SELECT * FROM transactions", self._instance)
        except pd.io.sql.DatabaseError:
            df = pd.DataFrame()
        return df

    def append_data(self, df: pd.DataFrame):
        df.to_sql("transactions", self._instance, if_exists="append", index=False)

def main():
    st.title("Personal Finance Tracker")

    conn = st.connection("finance_db", type=FinanceDataConnection)

    uploaded_file = st.file_uploader("Upload your monthly bank statement (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
            for col in ["Debit", "Credit", "Balance"]:
                df[col] = df[col].fillna(0)
                df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

            conn.append_data(df)
            st.success("File uploaded and data stored successfully!")

        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.header("Account Balance Trend")
    try:
        all_data = conn.get_all_data()
        if not all_data.empty:
            all_data['Date'] = pd.to_datetime(all_data['Date'])
            all_data = all_data.sort_values(by="Date")
            st.line_chart(all_data.rename(columns={"Date": "index"}).set_index("index")["Balance"])
        else:
            st.info("No data available to display. Upload a CSV file to get started.")
    except Exception as e:
        st.error(f"Error connecting to the database or fetching data: {e}")
        st.info("No data to display. Upload a CSV file to get started.")

if __name__ == "__main__":
    main()
