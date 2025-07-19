
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

    def append_data(self, df: pd.DataFrame) -> int:
        # Get existing transactions
        existing_data = self.get_all_data()
        
        # Count initial rows for reporting purposes
        initial_count = len(df)
        
        # If no existing data, just append all
        if existing_data.empty:
            df.to_sql("transactions", self._instance, if_exists="append", index=False)
            return initial_count
        
        # Convert date columns to string for comparison
        if 'Date' in existing_data.columns and 'Date' in df.columns:
            existing_data['Date'] = existing_data['Date'].astype(str)
            df['Date'] = df['Date'].astype(str)
        
        # Create a composite key for identifying unique transactions
        existing_data['composite_key'] = existing_data.apply(
            lambda row: f"{row['Date']}_{row['Description']}_{row['Debit']}_{row['Credit']}", axis=1
        )
        
        # Add composite key to new data
        df['composite_key'] = df.apply(
            lambda row: f"{row['Date']}_{row['Description']}_{row['Debit']}_{row['Credit']}", axis=1
        )
        
        # Filter out rows that already exist
        existing_keys = set(existing_data['composite_key'])
        new_rows = df[~df['composite_key'].isin(existing_keys)]
        
        # Remove the temporary composite key column
        new_rows = new_rows.drop(columns=['composite_key'])
        
        # Only add new unique transactions
        if not new_rows.empty:
            new_rows.to_sql("transactions", self._instance, if_exists="append", index=False)
        
        # Return number of new rows added
        return len(new_rows)

def main():
    st.title("Personal Finance Tracker")

    conn = st.connection("finance_db", type=FinanceDataConnection)
    
    # Add date filter in sidebar
    st.sidebar.header("Date Filter")
    
    # Get all data first to determine date range
    all_data = conn.get_all_data()
    
    # Default date range
    default_start_date = None
    default_end_date = None
    
    # If data exists, set default date range based on data
    if not all_data.empty:
        try:
            all_data['Date'] = pd.to_datetime(all_data['Date'])
            default_start_date = all_data['Date'].min().date()
            default_end_date = all_data['Date'].max().date()
        except Exception:
            # If date conversion fails, use None (will be handled later)
            pass
    
    # Date filters
    start_date = st.sidebar.date_input("Start Date", value=default_start_date)
    end_date = st.sidebar.date_input("End Date", value=default_end_date)

    uploaded_file = st.file_uploader("Upload your monthly bank statement (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
            for col in ["Debit", "Credit", "Balance"]:
                df[col] = df[col].fillna(0)
                df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

            new_rows_count = conn.append_data(df)
            if new_rows_count == len(df):
                st.success(f"File uploaded and all {new_rows_count} transactions stored successfully!")
            else:
                st.success(f"File uploaded! Added {new_rows_count} new unique transactions out of {len(df)} total.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

    try:
        all_data = conn.get_all_data()
        if not all_data.empty:
            all_data['Date'] = pd.to_datetime(all_data['Date'])
            
            # Apply date filter
            filtered_data = all_data
            if start_date and end_date:
                filtered_data = all_data[(all_data['Date'].dt.date >= start_date) & 
                                         (all_data['Date'].dt.date <= end_date)]
                st.info(f"Showing data from {start_date} to {end_date}")
            
            # Check if we have data after filtering
            if filtered_data.empty:
                st.warning("No data available for the selected date range.")
            else:
                filtered_data = filtered_data.sort_values(by="Date")
                
                # Calculate financial health KPIs
                st.subheader("Financial Health Overview")
                
                # Calculate total income (credits excluding transfers)
                income_data = filtered_data[filtered_data["Credit"] > 0]
                income_data = income_data[income_data["Category"] != "Transfer"]
                total_income = income_data["Credit"].sum()
                
                # Calculate total spending (debits excluding transfers and investments)
                spending_data = filtered_data[filtered_data["Debit"] > 0]
                spending_data = spending_data[~spending_data["Category"].isin(["Transfer", "Investing"])]
                total_spending = spending_data["Debit"].sum()
                
                # Calculate investments (separate from regular spending)
                investment_data = filtered_data[filtered_data["Category"] == "Investing"]
                total_investments = investment_data["Debit"].sum()
                
                # Calculate net cashflow (income - spending, excluding investments)
                net_cashflow = total_income - total_spending
                
                # Display KPI metrics in a row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Income vs. Spending", 
                        value=f"${net_cashflow:.2f}",
                        delta=f"{'+' if net_cashflow > 0 else ''}{net_cashflow/total_income*100:.1f}% of Income" if total_income > 0 else "N/A",
                        delta_color="normal" if net_cashflow >= 0 else "inverse"
                    )
                
                with col2:
                    st.metric(
                        label="Total Spending", 
                        value=f"${total_spending:.2f}",
                        delta=f"{total_spending/total_income*100:.1f}% of Income" if total_income > 0 else "N/A",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        label="Investments", 
                        value=f"${total_investments:.2f}",
                        delta=f"{total_investments/total_income*100:.1f}% of Income" if total_income > 0 else "N/A",
                        delta_color="normal"
                    )
                
                # Display account balance chart
                st.header("Account Balance Trend")
                st.line_chart(filtered_data.rename(columns={"Date": "index"}).set_index("index")["Balance"])

            st.header("Spending Breakdown")
            # Use filtered_data instead of all_data for spending analysis
            spending_data = filtered_data[filtered_data["Debit"] > 0]
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
                        
                        # Add bar chart for category spending
                        st.subheader("Category Spending Breakdown")
                        
                        # Sort by spending amount (descending)
                        category_spending = category_spending.sort_values(by="Debit", ascending=False)
                        
                        # Create bar chart with Plotly for better customization
                        bar_fig = go.Figure(data=[
                            go.Bar(
                                x=category_spending["Category"],
                                y=category_spending["Debit"],
                                text=category_spending["Debit"].apply(lambda x: f"${x:.2f}"),
                                textposition="auto",
                                marker_color="royalblue"
                            )
                        ])
                        
                        bar_fig.update_layout(
                            title="Spending by Category",
                            xaxis_title="Category",
                            yaxis_title="Amount ($)",
                            yaxis=dict(tickprefix="$"),
                            height=500
                        )
                        
                        st.plotly_chart(bar_fig, use_container_width=True)
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
            categorize_transactions_page(conn, start_date, end_date)


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


def categorize_transactions_page(conn, start_date=None, end_date=None):
    st.header("Categorize Transactions")

    all_data = conn.get_all_data()

    if all_data.empty:
        st.info("No transactions to categorize.")
        return
        
    # Convert date column to datetime
    all_data['Date'] = pd.to_datetime(all_data['Date'])
    
    # Apply date filter if provided
    filtered_data = all_data
    if start_date and end_date:
        filtered_data = all_data[(all_data['Date'].dt.date >= start_date) & 
                               (all_data['Date'].dt.date <= end_date)]
        st.info(f"Showing transactions from {start_date} to {end_date}")
    
    if filtered_data.empty:
        st.warning("No data available for the selected date range.")
        return

    # ✅ Filter out transactions already marked as 'Transfer'
    data_to_categorize = filtered_data[filtered_data["Category"] != "Transfer"]

    # Use the filtered dataframe to get descriptions
    unique_descriptions = data_to_categorize["Description"].unique()

    if st.button("Run AI Model Categorization"):
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

