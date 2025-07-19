
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS description_categories (
                    "Description" TEXT PRIMARY KEY,
                    "Category" TEXT NOT NULL
                );
            """)
        return conn

    def get_all_data(self) -> pd.DataFrame:
        try:
            df = pd.read_sql("""
                SELECT
                    t.*,
                    COALESCE(dc.Category, 'Other') as Category
                FROM
                    transactions t
                LEFT JOIN
                    description_categories dc ON t.Description = dc.Description
            """, self._instance)
        except pd.io.sql.DatabaseError:
            df = pd.DataFrame()
        return df

    def update_category(self, description: str, category: str):
        with self._instance:
            self._instance.execute("""
                INSERT OR REPLACE INTO description_categories (Description, Category)
                VALUES (?, ?)
            """, (description, category))

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

            st.header("Spending Breakdown")
            spending_data = all_data[all_data["Debit"] > 0]
            excluded_categories = ["Transfer"]
            spending_data = spending_data[~spending_data["Category"].isin(excluded_categories)]

            if not spending_data.empty:
                    category_spending = spending_data.groupby("Category")["Debit"].sum().reset_index()

                    if not category_spending.empty:
                        all_labels = ["Total Spending"] + category_spending["Category"].unique().tolist()
                        label_to_index = {label: i for i, label in enumerate(all_labels)}

                        source_indices = [label_to_index["Total Spending"]] * len(category_spending)
                        target_indices = [label_to_index[cat] for cat in category_spending["Category"]]
                        values = category_spending["Debit"].tolist()

                        link = dict(source=source_indices, target=target_indices, value=values)
                        node = dict(label=all_labels, pad=15, thickness=20, color="blue")
                        data = go.Sankey(link=link, node=node)

                        fig = go.Figure(data)
                        fig.update_layout(title_text="Sankey Diagram - Spending Breakdown", font_size=10)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No spending data to display.")
            else:
                st.info("No data available to display. Upload a CSV file to get started.")

    except Exception as e:
        st.error(f"Error connecting to the database or fetching data: {e}")
        st.info("No data to display. Upload a CSV file to get started.")

    if st.button("Categorize Transactions"):
            st.session_state.page = "categorize"

    if "page" in st.session_state and st.session_state.page == "categorize":
            categorize_transactions_page(conn)


def get_local_model_category(description: str) -> str:
    """
    Calls a local LLM to categorize a transaction and cleans the output
    to return a standardized category name.
    """
    import ollama

    # Define your standard categories
    valid_categories = ["Food", "Rent", "Bills", "Entertainment", "Other", "Investing", "Transfer"]

    # Use a more constrained prompt for the model
    prompt = f"""
    From the following list of categories: {valid_categories},
    which single category best describes the transaction: "{description}"?
    Respond with only the single category name.
    """

    try:
        response = ollama.chat(
            model='gemma:2b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        output = response['message']['content'].strip()

        # Check if the model's output contains one of the valid categories
        for category in valid_categories:
            if category.lower() in output.lower():
                return category  # Return the standardized category name

    except Exception as e:
        st.warning(f"Could not connect to local model: {e}")
        # Fallback to 'Other' if the model fails
        return "Other"

    # If no valid category is found in the response, default to 'Other'
    return "Other"


def categorize_transactions_page(conn):
    st.header("Categorize Transactions")

    all_data = conn.get_all_data()

    if all_data.empty:
        st.info("No transactions to categorize.")
        return

    # ✅ Filter out transactions already marked as 'Transfer'
    data_to_categorize = all_data[all_data["Category"] != "Transfer"]

    # Use the filtered dataframe to get descriptions
    unique_descriptions = data_to_categorize["Description"].unique()

    if st.button("Run Local Model Categorization"):
        with st.spinner("Categorizing transactions..."):
            for description in unique_descriptions:
                # Get current category from the filtered data
                current_cat = data_to_categorize[data_to_categorize["Description"] == description]["Category"].iloc[0]
                if current_cat == "Other":
                    local_category = get_local_model_category(description)
                    conn.update_category(description, local_category)
        st.success("Categorization complete!")
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Manual Categorization")

    options = ["Rent", "Bills", "Food", "Entertainment", "Investing", "Transfer", "Other"]

    for description in unique_descriptions:
        current_category = data_to_categorize[data_to_categorize["Description"] == description]["Category"].iloc[0]

        try:
            current_index = options.index(current_category)
        except ValueError:
            current_index = options.index("Other")

        new_category = st.selectbox(
            f"**{description}**",
            options,
            index=current_index,
            key=description
        )

        if new_category != current_category:
            conn.update_category(description, new_category)
            st.experimental_rerun()


if __name__ == "__main__":
    main()

