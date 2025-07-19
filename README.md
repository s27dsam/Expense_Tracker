# Expense Tracker

A personal finance management application built with Streamlit that helps you track, categorize, and visualize your spending habits.

![Expense Tracker](https://img.shields.io/badge/App-Expense%20Tracker-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **CSV Import**: Upload bank statements in CSV format
- **Automatic Transaction Categorization**: Uses a local LLM (Ollama) to automatically categorize transactions
- **Manual Category Assignment**: Manually assign categories to transactions
- **Spending Visualization**: 
  - View your account balance trend
  - See spending breakdown with interactive Sankey diagrams
- **Persistent Storage**: Data is stored in SQLite database for easy access

## Requirements

- Python 3.8+
- Dependencies:
  - streamlit
  - pandas
  - plotly
  - ollama

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/expense-tracker.git
   cd expense-tracker
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Ollama (for local LLM categorization):
   - Follow the instructions at [Ollama's website](https://ollama.ai/download)
   - Make sure to pull the gemma:2b model:
     ```bash
     ollama pull gemma:2b
     ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

3. Upload a CSV file with your bank statement data
   - The CSV should have columns: Date, Description, Debit, Credit, Balance

4. Use the "Categorize Transactions" button to assign categories
   - Let the local model automatically categorize transactions
   - Or manually assign categories from the dropdown menus

5. View the visualizations of your spending patterns

## Data Structure

The application expects CSV files with the following columns:
- `Date`: Transaction date (format: DD MMM YYYY)
- `Description`: Transaction description
- `Debit`: Amount withdrawn (positive number)
- `Credit`: Amount deposited (positive number)
- `Balance`: Account balance after the transaction

## Technology Stack

- **Frontend & Backend**: Streamlit
- **Data Processing**: Pandas
- **Data Visualization**: Plotly
- **Database**: SQLite
- **Transaction Categorization**: Ollama (Local LLM)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

