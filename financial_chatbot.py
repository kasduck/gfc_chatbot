import pandas as pd
import re
from difflib import get_close_matches
from collections import defaultdict
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialChatbot:
    """A financial chatbot that provides information about companies' financial metrics."""
    
    def __init__(self, data_file):
        """Initialize the chatbot with data from CSV file."""
        self.df = self._load_data(data_file)
        self.companies_data = self._preprocess_data()
        self.metric_mappings = self._define_metric_mappings()
        self.economic_context = self._define_economic_context()
        self.benchmarks = self._define_benchmarks()
        
        # Extract all available companies, years, and metrics for validation
        self.available_companies = list(self.companies_data.keys())
        self.available_years = set()
        for company_data in self.companies_data.values():
            self.available_years.update(company_data.keys())
        self.available_years = sorted(list(self.available_years))
        self.available_metrics = list(self.metric_mappings.keys())
        
        # Create synonym mappings for companies and metrics
        self.company_synonyms = {
            'apple': 'Apple',
            'aapl': 'Apple',
            'microsoft': 'Microsoft',
            'msft': 'Microsoft',
            'tesla': 'Tesla',
            'tsla': 'Tesla'
        }
        
        # Word embeddings simulation (for demonstration)
        # In a real implementation, you might use pre-trained word embeddings or a small model
        self._create_simple_word_vectors()
    
    def _load_data(self, file_path):
        """Load data from CSV file with error handling."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _preprocess_data(self):
        """Create a nested dictionary for fast lookup: {company: {year: row}}."""
        companies_data = {}
        for company in ['Apple', 'Microsoft', 'Tesla']:
            company_data = {}
            company_rows = self.df[self.df[f'Company_{company}'] == True]
            for year in self.df['Year'].unique():
                row = company_rows[company_rows['Year'] == year]
                if not row.empty:
                    company_data[int(year)] = row.iloc[0]
            companies_data[company] = company_data
        return companies_data
    
    def _define_metric_mappings(self):
        """Define metric synonyms and properties."""
        return {
            'revenue': {'column': 'Total Revenue', 'unit': '$M', 'description': 'total revenue'},
            'total revenue': {'column': 'Total Revenue', 'unit': '$M', 'description': 'total revenue'},
            'net income': {'column': 'Net Income', 'unit': '$M', 'description': 'net income'},
            'income': {'column': 'Net Income', 'unit': '$M', 'description': 'net income'},
            'profit': {'column': 'Net Income', 'unit': '$M', 'description': 'net income'},
            'total assets': {'column': 'Total Assets', 'unit': '$M', 'description': 'total assets'},
            'assets': {'column': 'Total Assets', 'unit': '$M', 'description': 'total assets'},
            'total liabilities': {'column': 'Total Liabilities', 'unit': '$M', 'description': 'total liabilities'},
            'liabilities': {'column': 'Total Liabilities', 'unit': '$M', 'description': 'total liabilities'},
            'cash flow': {'column': 'Cash Flow from Operating Activities', 'unit': '$M', 'description': 'cash flow from operations'},
            'operating cash flow': {'column': 'Cash Flow from Operating Activities', 'unit': '$M', 'description': 'cash flow from operations'},
            'profit margin': {'column': 'Profit Margin (%)', 'unit': '%', 'description': 'percentage of revenue kept as profit'},
            'margin': {'column': 'Profit Margin (%)', 'unit': '%', 'description': 'percentage of revenue kept as profit'},
            'leverage ratio': {'column': 'Leverage Ratio', 'unit': '', 'description': 'ratio of liabilities to assets'},
            'debt ratio': {'column': 'Leverage Ratio', 'unit': '', 'description': 'ratio of liabilities to assets'},
            'revenue growth': {'column': 'Total Revenue_YoY_Growth (%)', 'unit': '%', 'description': 'year-over-year revenue growth'},
            'net income growth': {'column': 'Net Income_YoY_Growth (%)', 'unit': '%', 'description': 'year-over-year net income growth'},
            'assets growth': {'column': 'Total Assets_YoY_Growth (%)', 'unit': '%', 'description': 'year-over-year assets growth'},
            'liabilities growth': {'column': 'Total Liabilities_YoY_Growth (%)', 'unit': '%', 'description': 'year-over-year liabilities growth'},
            'cash flow growth': {'column': 'Cash Flow from Operating Activities_YoY_Growth (%)', 'unit': '%', 'description': 'year-over-year cash flow growth'}
        }
    
    def _define_economic_context(self):
        """Define economic context for responses."""
        return {
            'Apple': {
                2022: 'strong ecosystem growth despite global supply chain disruptions',
                2023: 'resilience amid inflation and higher R&D costs',
                2024: 'services-driven stability in a high-interest-rate environment'
            },
            'Microsoft': {
                2022: 'cloud and software growth in a recovering economy',
                2023: 'AI and cloud demand driving performance',
                2024: 'leadership in AI and cloud amid tech sector expansion'
            },
            'Tesla': {
                2022: 'rapid EV market growth with production scaling',
                2023: 'margin pressure from price cuts and competition',
                2024: 'slower EV growth offset by energy storage expansion'
            }
        }
    
    def _define_benchmarks(self):
        """Define benchmark values for comparison."""
        return {
            'Profit Margin (%)': {'value': 25, 'description': 'tech sector average'},
            'Leverage Ratio': {'value': 0.5, 'description': 'prudent debt threshold'}
        }
    
    def _create_simple_word_vectors(self):
        """Create simple word vectors for semantic similarity (demonstration only)."""
        # In a real implementation, you'd use pre-trained embeddings or a small model
        # This is a very simplistic approach for demonstration
        self.word_vectors = {
            # Company terms
            'apple': np.array([0.9, 0.1, 0.1]),
            'aapl': np.array([0.8, 0.1, 0.1]),
            'microsoft': np.array([0.1, 0.9, 0.1]),
            'msft': np.array([0.1, 0.8, 0.1]),
            'tesla': np.array([0.1, 0.1, 0.9]),
            'tsla': np.array([0.1, 0.1, 0.8]),
            
            # Financial metrics
            'revenue': np.array([0.9, 0.1, 0.1, 0.1, 0.1]),
            'sales': np.array([0.8, 0.1, 0.1, 0.1, 0.1]),
            'income': np.array([0.1, 0.9, 0.1, 0.1, 0.1]),
            'profit': np.array([0.1, 0.8, 0.1, 0.1, 0.1]),
            'earnings': np.array([0.1, 0.7, 0.1, 0.1, 0.1]),
            'assets': np.array([0.1, 0.1, 0.9, 0.1, 0.1]),
            'liabilities': np.array([0.1, 0.1, 0.1, 0.9, 0.1]),
            'debts': np.array([0.1, 0.1, 0.1, 0.8, 0.1]),
            'cash flow': np.array([0.1, 0.1, 0.1, 0.1, 0.9]),
            'cash': np.array([0.1, 0.1, 0.1, 0.1, 0.8]),
        }
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)
    
    def find_closest_term(self, query_term, vocabulary):
        """Find the closest term using semantic similarity."""
        query_term = query_term.lower()
        
        # First try exact match
        if query_term in vocabulary:
            return query_term
        
        # Then try fuzzy string matching
        matches = get_close_matches(query_term, vocabulary, n=1, cutoff=0.7)
        if matches:
            return matches[0]
        
        # Finally try simple vector similarity if we have vectors for both
        if hasattr(self, 'word_vectors') and query_term in self.word_vectors:
            best_similarity = -1
            best_match = None
            
            for term in vocabulary:
                if term in self.word_vectors:
                    similarity = self._cosine_similarity(
                        self.word_vectors[query_term], 
                        self.word_vectors[term]
                    )
                    if similarity > best_similarity and similarity > 0.7:
                        best_similarity = similarity
                        best_match = term
            
            if best_match:
                return best_match
        
        return None
    
    def parse_query(self, user_query):
        """Parse user query to extract company, year, and metric with fuzzy matching."""
        user_query = user_query.lower().strip()
        
        # Extract company using improved matching
        company = None
        # First look for exact matches of company names or ticker symbols
        for comp_term in self.company_synonyms:
            if comp_term in user_query.split():
                company = self.company_synonyms[comp_term]
                break
        
        # If no exact match, try fuzzy matching on the entire query
        if not company:
            # Extract potential company name tokens
            query_tokens = user_query.split()
            for token in query_tokens:
                match = self.find_closest_term(token, self.company_synonyms.keys())
                if match:
                    company = self.company_synonyms[match]
                    break
        
        # Extract year with regex and validation
        year_match = re.search(r'\b(20\d{2})\b', user_query)
        if year_match:
            potential_year = int(year_match.group(1))
            if potential_year in self.available_years:
                year = potential_year
            else:
                # If mentioned year isn't in our data, default to most recent
                year = max(self.available_years)
        else:
            # Default to the most recent year
            year = max(self.available_years)
        
        # Extract metric with improved matching
        metric = None
        # First check for exact phrase matches
        for key in self.metric_mappings:
            if key in user_query:
                metric = key
                break
        
        # If no exact match, try fuzzy matching on individual words
        if not metric:
            query_tokens = set(user_query.split())
            # Try to match each metric term
            for key in self.metric_mappings:
                key_tokens = set(key.split())
                if any(token in query_tokens for token in key_tokens):
                    metric = key
                    break
        
        # If still no match, try semantic similarity on the whole query
        if not metric:
            # Extract non-company, non-year terms to focus on metric
            filtered_query = ' '.join([
                token for token in user_query.split() 
                if token not in self.company_synonyms and not re.match(r'20\d{2}', token)
            ])
            
            # Check if filtered query has similarity to any metrics
            for key in self.metric_mappings:
                # Simple word overlap as a basic similarity measure
                key_tokens = set(key.split())
                query_tokens = set(filtered_query.split())
                overlap = len(key_tokens.intersection(query_tokens))
                if overlap > 0:
                    metric = key
                    break
        
        # Special handling for financial statement metrics
        if 'balance sheet' in user_query or 'assets' in user_query or 'liabilities' in user_query:
            if 'assets' in user_query or ('balance' in user_query and 'liabilities' not in user_query):
                metric = 'total assets'
            elif 'liabilities' in user_query or 'debt' in user_query:
                metric = 'total liabilities'
        
        elif 'income' in user_query or 'profit' in user_query or 'earnings' in user_query:
            if 'margin' in user_query:
                metric = 'profit margin'
            else:
                metric = 'net income'
        
        elif 'cash' in user_query or 'flow' in user_query:
            metric = 'cash flow'
        
        # Add smart fallbacks for metrics
        if not metric and ('how much' in user_query or 'performance' in user_query):
            # Default to revenue as the most common financial metric
            metric = 'revenue'
        
        return company, year, metric
    
    def interpret_intent(self, user_query):
        """Identify the main intent of the user query."""
        user_query = user_query.lower()
        
        # Check for comparison intent
        if 'compare' in user_query or 'comparison' in user_query or 'vs' in user_query or 'versus' in user_query:
            return 'compare'
        
        # Check for trend analysis intent
        if 'trend' in user_query or 'over time' in user_query or 'historical' in user_query:
            return 'trend'
        
        # Check for specific metric query
        company, year, metric = self.parse_query(user_query)
        if company and metric:
            return 'metric_query'
        
        # Check for general company performance
        if company and ('how' in user_query + 'performance' in user_query or 'overview' in user_query):
            return 'company_overview'
        
        # Default to basic query
        return 'basic_query'
    
    def generate_response(self, user_query):
        """Generate a detailed response for the query using NLP understanding."""
        # Parse the query
        company, year, metric = self.parse_query(user_query)
        
        # Validate parsed information and provide helpful errors
        if not company:
            return self._create_help_response("company", 
                f"I couldn't identify which company you're asking about. Try specifying Apple, Microsoft, or Tesla.")
        
        if company not in self.companies_data:
            return f"Sorry, I don't have data for {company}. Available companies: {', '.join(self.available_companies)}."
        
        if year not in self.companies_data[company]:
            available_years = sorted(list(self.companies_data[company].keys()))
            return self._create_help_response("year",
                f"I don't have data for {company} in {year}. Available years: {', '.join(map(str, available_years))}.")
        
        if not metric:
            # Try to infer what the user might be asking about
            if "how" in user_query.lower() and "do" in user_query.lower():
                return self._create_help_response("metric",
                    f"I understand you're asking about {company}'s performance in {year}, but I'm not sure which specific metric you want. "
                    f"Try asking about revenue, profit, assets, liabilities, or cash flow.")
            else:
                sample_metrics = ['revenue', 'net income', 'profit margin', 'total assets']
                return self._create_help_response("metric",
                    f"I'm not sure which financial metric you're interested in for {company} in {year}. "
                    f"You can ask about metrics like: {', '.join(sample_metrics)}.")
        
        if metric not in self.metric_mappings:
            close_matches = get_close_matches(metric, self.metric_mappings.keys(), n=3, cutoff=0.6)
            suggestion = f" Did you mean: {', '.join(close_matches)}?" if close_matches else ""
            return self._create_help_response("metric_specific",
                f"I don't recognize '{metric}' as a financial metric.{suggestion} "
                f"Try common metrics like revenue, profit, assets, or liabilities.")
        
        # Get data and format response
        try:
            row = self.companies_data[company][year]
            column = self.metric_mappings[metric]['column']
            value = row[column]
            unit = self.metric_mappings[metric]['unit']
            description = self.metric_mappings[metric]['description']
            
            # Format value based on unit
            if unit == '$M':
                formatted_value = f"${value:,.0f} million"
            elif unit == '%':
                formatted_value = f"{value:.2f}%"
            else:
                formatted_value = f"{value:.2f}"
            
            # Add economic context
            context = self.economic_context[company][year]
            
            # Add benchmarks for specific metrics
            benchmark_info = ""
            if column in self.benchmarks:
                bench_value = self.benchmarks[column]['value']
                bench_desc = self.benchmarks[column]['description']
                comparison = 'above' if value > bench_value else 'below' if value < bench_value else 'at'
                benchmark_info = f", {comparison} the {bench_desc} ({bench_value}{'%' if unit == '%' else ''})"
            
            # Add year-over-year comparison if data is available
            yoy_insight = ""
            if year > min(self.available_years) and f"{column}_YoY_Growth (%)" in row:
                growth = row[f"{column}_YoY_Growth (%)"]
                if not pd.isna(growth):
                    direction = "increased" if growth > 0 else "decreased"
                    yoy_insight = f" This represents a {abs(growth):.2f}% {direction} from {year-1}."
            
            # Generate core response
            response = (f"{company}'s {description} in {year} was {formatted_value}{benchmark_info}, "
                       f"reflecting its {context}.{yoy_insight}")
            
            # Add industry comparison where relevant
            if metric in ['profit margin', 'revenue', 'net income']:
                # Get data for all companies in the same year for comparison
                company_values = {}
                for comp in self.available_companies:
                    if year in self.companies_data[comp]:
                        comp_row = self.companies_data[comp][year]
                        company_values[comp] = comp_row[column]
                
                # Find rank
                values_list = sorted(company_values.items(), key=lambda x: x[1], reverse=True)
                rank = next(i+1 for i, (c, _) in enumerate(values_list) if c == company)
                
                # Add comparison insight
                if len(values_list) > 1:
                    if rank == 1:
                        response += f" This was the highest {description} among the tech companies analyzed for {year}."
                    else:
                        response += f" This ranked #{rank} among the tech companies analyzed for {year}."
            
            return response
            
        except (KeyError, ValueError) as e:
            logger.error(f"Error generating response: {e}")
            return f"I'm sorry, but I encountered an error retrieving data for {company}'s {metric} in {year}. Please try a different query."
    
    def _create_help_response(self, error_type, message):
        """Create a helpful response when the query can't be fully understood."""
        if error_type == "company":
            return (f"{message} \n\nExample queries:\n"
                   f"• 'What was Apple's revenue in 2023?'\n"
                   f"• 'Tell me Microsoft's profit margin in 2024'\n"
                   f"• 'How much did Tesla make in 2022?'")
        
        elif error_type == "year":
            return message
        
        elif error_type == "metric":
            return (f"{message} \n\nExample queries:\n"
                   f"• 'What was the revenue?'\n"
                   f"• 'Tell me about profit margin'\n"
                   f"• 'Show me the assets and liabilities'")
        
        elif error_type == "metric_specific":
            return message
        
        else:
            return "I'm not sure I understand your question. Try asking about a company's specific financial metric for a particular year."

    def process_query(self, user_query):
        """Process user query and generate appropriate response."""
        intent = self.interpret_intent(user_query)
        
        if intent == 'compare':
            # Basic comparison handling (could be expanded)
            return "I understand you want to compare companies or metrics. Currently, I can only answer about specific metrics. Please ask about a specific company and metric."
        
        elif intent == 'trend':
            # Basic trend handling (could be expanded)
            company, _, metric = self.parse_query(user_query)
            if company and metric and metric in self.metric_mappings:
                return f"I understand you want to see trends for {company}'s {metric}. Currently, I can only answer about specific years. Please specify a year in your query."
            else:
                return self.generate_response(user_query)
        
        else:
            # Standard query handling
            return self.generate_response(user_query)
    
    def interactive_loop(self):
        """Run the command-line interaction loop."""
        print("\n=== Financial Data Assistant ===")
        print("Ask me about Apple, Microsoft, or Tesla's financial metrics for 2022-2024.")
        print("Examples: 'What was Apple's revenue in 2023?' or 'Tell me Tesla's profit margin in 2024'")
        print("Type 'exit' to quit.\n")
        
        while True:
            user_input = input("\nYour question: ")
            if user_input.lower().strip() in ['exit', 'quit', 'bye']:
                print("\nThank you for using the Financial Data Assistant. Goodbye!")
                break
                
            if user_input.strip():
                response = self.process_query(user_input)
                print(f"\n{response}")
            else:
                print("\nPlease enter a question or type 'exit' to quit.")


# Run the chatbot
if __name__ == "__main__":
    try:
        chatbot = FinancialChatbot("Preprocessed_Financial_Data_MSFT_TSLA_AAPL_2022_2024.csv")
        chatbot.interactive_loop()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")