# GFC Financial Chatbot

The GFC Financial Chatbot is a Python-based application designed to deliver actionable financial insights for Apple (AAPL), Microsoft (MSFT), and Tesla (TSLA) using data extracted from their SEC 10-K filings for fiscal years 2022–2024. The project encompasses three key stages: (1) extracting financial metrics, (2) cleaning, preprocessing, and analyzing the data to uncover trends, and (3) building a sophisticated chatbot to answer dynamic financial queries. The chatbot leverages advanced query parsing, fuzzy matching, and semantic similarity to provide context-rich responses, including economic insights, year-over-year (YoY) comparisons, and industry rankings.

## Project Overview

The GFC Financial Chatbot was developed to transform complex financial data into user-friendly insights, catering to financial analysts and stakeholders at GFC. The project integrates data extraction, preprocessing, trend analysis, and interactive querying, with the following components:
- **Data Extraction and Preprocessing**: Extracted key financial metrics from 10-K filings, cleaned and enriched the data with ratios, YoY growth rates, and normalized features using a Jupyter Notebook (`Financial_Analysis_MSFT_TSLA_AAPL.ipynb`).
- **Trend Analysis**: Analyzed financial trends (e.g., revenue growth, profitability) and visualized them to inform chatbot responses.
- **Chatbot Development**: Built a command-line chatbot (`financial_chatbot.py`) to handle queries about financial metrics, years, and companies, evolving from a basic script to a robust system with intent recognition and error handling.

The deliverables include:
- `financial_chatbot.py`: The chatbot script.
- `Financial_Analysis_MSFT_TSLA_AAPL.ipynb`: The Jupyter Notebook for data extraction, cleaning, and analysis.
- `Preprocessed_Financial_Data_MSFT_TSLA_AAPL_2022_2024.csv`: The preprocessed dataset.
- `README.md`: This documentation.

## Phase 1: Data Extraction

### Objective
Extract key financial metrics from the SEC 10-K filings of Apple, Microsoft, and Tesla for fiscal years 2022–2024.

### Process
- **Source**: Retrieved 10-K filings from the SEC EDGAR database for each company.
- **Metrics Extracted** (in millions of USD):
  - Total Revenue
  - Net Income
  - Total Assets
  - Total Liabilities
  - Cash Flow from Operating Activities
- **Data Collection**:
  - Manually reviewed Income Statements, Balance Sheets, and Cash Flow Statements in the 10-K filings.
  - Recorded data for each company and year, ensuring accuracy.
  - Example data points (2024):
    - Apple: Total Revenue = $391,145M, Net Income = $94,761M
    - Microsoft: Total Revenue = $245,122M, Net Income = $88,136M
    - Tesla: Total Revenue = $97,690M, Net Income = $7,091M
- **Output**: Created `Financial_Data_MSFT_TSLA_AAPL_2022_2024.csv` with 7 columns (`Company`, `Year`, and the 5 metrics) and 9 rows (3 years × 3 companies).

### Implementation
- Documented in `Financial_Analysis_MSFT_TSLA_AAPL.ipynb` (Step 1: Data Preparation and Preprocessing).
- Input CSV was manually compiled and loaded into pandas for cleaning and preprocessing.

## Phase 2: Data Cleaning and Preprocessing

### Objective
Clean, normalize, and enrich the raw financial data to create an AI-ready dataset for analysis and chatbot integration.

### Process
- **Input**: `Financial_Data_MSFT_TSLA_AAPL_2022_2024.csv`.
- **Cleaning Steps**:
  - **Missing Values**: Confirmed no NaN values using `df.isnull().sum()`.
  - **Duplicates**: Verified no duplicate rows with `df.duplicated().sum()`.
  - **Data Types**: Ensured `Company` as string, `Year` as integer, and financial metrics as numeric (`int64` or `float64`) using `pd.to_numeric`.
  - **Consistency**: Validated all metrics were in millions of USD, cross-checked against 10-K filings.
- **Transformations**:
  - **Normalization**: Applied min-max scaling within each company to normalize metrics to [0,1] using:
    ```
    (x - min(x)) / (max(x) - min(x))
    ```
    - Example: Apple’s Total Revenue (2022: $394,328M, 2023: $383,285M, 2024: $391,145M) → Normalized: 1.0, 0.0, 0.7118.
  - **Financial Ratios**:
    - **Profit Margin (%)**: `(Net Income / Total Revenue) × 100`, e.g., Apple 2022: 25.31%.
    - **Leverage Ratio**: `Total Liabilities / Total Assets`, e.g., Tesla 2024: 0.40.
  - **YoY Growth Rates**: `((Current - Previous) / Previous) × 100`, e.g., Microsoft 2024 Revenue Growth: 15.67%.
  - **Lag Features**: Added previous year’s values, e.g., Apple 2023 `Total Revenue_Lag1` = $394,328M.
  - **One-Hot Encoding**: Converted `Company` to binary columns (`Company_Apple`, `Company_Microsoft`, `Company_Tesla`).
- **Output**:
  - `Preprocessed_Financial_Data_MSFT_TSLA_AAPL_2022_2024.csv`:
    - 26 columns: Original metrics, normalized values, ratios, growth rates, lag features, one-hot columns.
    - 9 rows (3 years × 3 companies).
    - Example row (Apple 2024):
      ```
      Year: 2024, Total Revenue: 391145, Net Income: 94761, Profit Margin (%): 24.23, Leverage Ratio: 0.80, Total Revenue_YoY_Growth (%): 2.05, Company_Apple: True
      ```
- **Implementation**:
  - Executed in `Financial_Analysis_MSFT_TSLA_AAPL.ipynb` (Step 1).
  - Used `pandas` for data manipulation, `numpy` for calculations, and custom functions for normalization.
  - Saved output with `df.to_csv`.

### Validation
- Cross-checked original metrics against 10-K filings.
- Verified calculations (e.g., Tesla 2024 Profit Margin = 7.26%, Microsoft 2024 Revenue Growth = 15.67%).
- Confirmed dataset integrity (9 rows, 26 columns, no missing values).

## Phase 3: Trend Analysis

### Objective
Analyze financial trends and visualize key metrics to inform chatbot responses and provide insights into company performance.

### Process
- **Input**: `Preprocessed_Financial_Data_MSFT_TSLA_AAPL_2022_2024.csv`.
- **Analysis**:
  - **YoY Growth Rates**: Summarized average growth rates per company (e.g., Microsoft Revenue: 11.28% YoY).
  - **Financial Ratios**: Evaluated Profit Margin and Leverage Ratio for profitability and risk.
  - **Trends**:
    - **Microsoft**: Strong growth (11.3% revenue, 10.6% net income YoY), high margins (~35%).
    - **Apple**: Stable revenue (~$390B), slight net income decline (-2.6% YoY), strong services.
    - **Tesla**: Volatile, with 2024 revenue growth at 0.95% and net income drop of -52.7%.
- **Visualizations**:
  - Plotted Total Revenue, Net Income, Profit Margin, and Leverage Ratio over 2022–2024.
  - Saved as PNGs (`revenue_trend.png`, `net_income_trend.png`, `profit_margin_trend.png`, `leverage_ratio_trend.png`).
  - Example: Revenue plot showed Microsoft’s growth, Apple’s stability, Tesla’s slowdown.
- **Key Insights**:
  - **Microsoft**: Cloud and AI drove 15.67% revenue growth in 2024, with $118.5B cash flow.
  - **Apple**: Stable ~$390B revenue, improved leverage (0.80 in 2024), services growth.
  - **Tesla**: 2024 net income fell to $7.1B (-52.7%), but energy storage grew 67% (per 10-K).
- **Implementation**:
  - Documented in `Financial_Analysis_MSFT_TSLA_AAPL.ipynb` (Step 2: Trend Analysis).
  - Used `pandas` for analysis, `matplotlib` for visualizations.
  - Converted one-hot columns to integers for consistency.

### Output
- Analytical summaries (e.g., average YoY growth rates).
- Visualization PNGs for potential chatbot integration.
- Insights to guide chatbot responses (e.g., “Microsoft’s 35% margin vs. Tesla’s 7%”).

## Phase 4: Chatbot Development

### Objective
Develop a command-line chatbot to answer dynamic financial queries using the preprocessed dataset, providing accurate, context-rich responses.

### Process
- **Implementation**: `financial_chatbot.py`.
- **Features**:
  - **Data Structure**: Nested dictionary (`{company: {year: row}}`) for O(1) lookups.
  - **Query Parsing**: Extracts company, year, and metric using:
    - Exact matches (e.g., “Apple”, “revenue”).
    - Fuzzy matching (`difflib.get_close_matches`) for misspellings.
    - Synonym mappings (e.g., “AAPL” → Apple, “profit” → Net Income).
    - Simple word vectors for semantic similarity (e.g., “sales” ≈ “revenue”).
    - Regex for year detection (e.g., `\b20\d{2}\b`).
  - **Intent Recognition**: Identifies intents (metric query, comparison, trend, overview).
  - **Response Generation**:
    - Formats values (e.g., `$391,145 million`, “24.23%”).
    - Adds economic context (e.g., Apple 2024: “services-driven stability”).
    - Includes benchmarks (e.g., Profit Margin vs. 25% tech average).
    - Provides YoY insights (e.g., “increased 15.67% from 2023”).
    - Ranks companies for key metrics (e.g., “highest net income in 2024”).
  - **Error Handling**: Suggests valid inputs for errors (e.g., “Try Apple, Microsoft, or Tesla”).
  - **Logging**: Tracks operations for debugging.
- **Example Interaction**:
  ```
  === Financial Data Assistant ===
  Ask me about Apple, Microsoft, or Tesla's financial metrics for 2022-2024.
  Examples: 'What was Apple's revenue in 2023?' or 'Tell me Tesla's profit margin in 2024'
  Type 'exit' to quit.

  Your question: What is Microsoft's leverage ratio in 2023?
  Microsoft’s ratio of liabilities to assets in 2023 was 0.50, at the prudent debt threshold (0.5), reflecting its AI and cloud demand driving performance.

  Your question: Tesla’s net income in 2024?
  Tesla’s net income in 2024 was $7,091 million, reflecting its slower EV growth offset by energy storage expansion. This ranked #3 among the tech companies analyzed for 2024.
  ```
- **Dependencies**:
  - Python 3.7+
  - Libraries: `pandas`, `numpy`, `difflib`, `logging` (standard library).
  - Dataset: `Preprocessed_Financial_Data_MSFT_TSLA_AAPL_2022_2024.csv`.

### Validation
- Tested with diverse queries (e.g., “MSFT profit 2023”, “Tesla assets 2024”).
- Verified responses against dataset and 10-K filings.
- Ensured robust error handling for invalid inputs.

## Repository Structure

```
gfc-financial-chatbot/
├── financial_chatbot.py              # Chatbot script
├── Financial_Analysis_MSFT_TSLA_AAPL.ipynb  # Jupyter Notebook for data extraction and analysis
├── Preprocessed_Financial_Data_MSFT_TSLA_AAPL_2022_2024.csv  # Preprocessed dataset
├── README.md                         # Project documentation
├── .gitignore                        # Excludes Python cache files
├── revenue_trend.png                 # Visualization: Revenue trends
├── net_income_trend.png              # Visualization: Net income trends
├── profit_margin_trend.png           # Visualization: Profit margin trends
├── leverage_ratio_trend.png          # Visualization: Leverage ratio trends
```

## Installation and Usage

### Prerequisites
- Python 3.7+
- Jupyter Notebook (for running the analysis)
- Git (optional, for cloning)
- Terminal or command prompt

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourGitHubUsername/gfc-financial-chatbot.git
   cd gfc-financial-chatbot
   ```
2. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib
   ```
3. **Ensure Files are Present**:
   - Verify `Preprocessed_Financial_Data_MSFT_TSLA_AAPL_2022_2024.csv` and `Financial_Analysis_MSFT_TSLA_AAPL.ipynb` are in the directory.
   - The chatbot requires the CSV; the notebook generates visualizations.

### Running the Analysis
1. Open `Financial_Analysis_MSFT_TSLA_AAPL.ipynb` in Jupyter Notebook:
   ```bash
   jupyter notebook Financial_Analysis_MSFT_TSLA_AAPL.ipynb
   ```
2. Run all cells to:
   - Preprocess the raw data (if starting from `Financial_Data_MSFT_TSLA_AAPL_2022_2024.csv`).
   - Generate trend analysis and visualizations (PNG files).
3. Review outputs (tables, plots) in the notebook.

### Running the Chatbot
1. Navigate to the project directory:
   ```bash
   cd path/to/gfc-financial-chatbot
   ```
2. Run the chatbot:
   ```bash
   python financial_chatbot.py
   ```
3. Interact with the chatbot:
   - Enter queries like “What was Apple’s revenue in 2023?” or “Tesla’s profit margin in 2024”.
   - Type `exit` to quit.
4. Example interaction:
   ```
   Your question: What is Microsoft’s leverage ratio in 2023?
   Microsoft’s ratio of liabilities to assets in 2023 was 0.50, at the prudent debt threshold (0.5), reflecting its AI and cloud demand driving performance.
   ```

## Features

- **Comprehensive Metrics**: Supports 20+ metrics (e.g., revenue, profit margin, leverage ratio, YoY growth) for 2022–2024.
- **Flexible Query Handling**: Recognizes synonyms (e.g., “MSFT” → Microsoft), fuzzy matches, and semantic similarities.
- **Context-Rich Responses**: Includes economic context, benchmarks, YoY insights, and industry rankings.
- **Robust Error Handling**: Guides users with suggestions for invalid inputs.
- **Trend Analysis**: Provides insights into Microsoft’s growth, Apple’s stability, and Tesla’s volatility.
- **Visualizations**: Generates plots for revenue, net income, profit margin, and leverage ratio trends.

## Limitations

- **Query Complexity**: Limited to keyword-based parsing with simple word vectors, not full NLP.
- **Dataset Scope**: Covers only Apple, Microsoft, Tesla for 2022–2024, excluding other metrics (e.g., stock price).
- **Interface**: Command-line only, no GUI or web interface.
- **Conversation Memory**: Lacks query history for follow-up questions.

## Future Enhancements

- **NLP Integration**: Use spaCy or transformers for advanced query understanding.
- **Expanded Dataset**: Include more companies or metrics via SEC EDGAR APIs.
- **Web Interface**: Develop a Flask or Streamlit app with interactive visualizations.
- **Predictive Models**: Implement ARIMA or LSTM using lag features for 2025 forecasts.
- **Conversation Memory**: Add query history for contextual responses.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License (to be added as `LICENSE`).

## Acknowledgments

- Data sourced from SEC EDGAR 10-K filings.
- Built with Python, `pandas`, `numpy`, `matplotlib`, and standard libraries (`difflib`, `logging`).
- Developed for GFC to enhance financial data accessibility.

## Contact

For questions or feedback, contact [YourGitHubUsername] on GitHub or via email at [your.email@example.com].