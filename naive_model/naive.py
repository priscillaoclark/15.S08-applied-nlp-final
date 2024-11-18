import os
import re
from collections import Counter
from scipy.stats import chi2_contingency
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Tuple

class KeywordAnalyzer:
    def __init__(self, keywords: List[str], pre_folder: str, post_folder: str):
        """
        Initialize the KeywordAnalyzer with keywords and folder paths.
        
        Args:
            keywords: List of keywords to analyze
            pre_folder: Path to pre-event documents
            post_folder: Path to post-event documents
        """
        self.keywords = [k.lower() for k in keywords]
        self.pre_folder = Path(pre_folder)
        self.post_folder = Path(post_folder)
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the analyzer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def read_file(self, file_path: Path) -> str:
        """
        Read text from a file with error handling.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            The contents of the file as a string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().lower()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return ""

    def count_keywords_in_text(self, text: str) -> Counter:
        """
        Count keyword occurrences in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Counter object with keyword counts
        """
        counts = Counter()
        for keyword in self.keywords:
            counts[keyword] = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
        return counts

    def analyze_folder(self, folder_path: Path) -> Tuple[Counter, Counter]:
        """
        Analyze all documents in a folder for keyword occurrences and document counts.
        
        Args:
            folder_path: Path to the folder to analyze
            
        Returns:
            Tuple of (keyword_counts, document_counts)
        """
        keyword_counts = Counter()
        doc_counts = Counter()
        
        try:
            files = [f for f in folder_path.glob("*.htm")]
            self.logger.info(f"Processing {len(files)} files in {folder_path}")
            
            for file_path in files:
                text = self.read_file(file_path)
                if not text:
                    continue
                
                # Count total occurrences
                counts = self.count_keywords_in_text(text)
                keyword_counts.update(counts)
                
                # Count documents containing each keyword
                for keyword in self.keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                        doc_counts[keyword] += 1
                        
            return keyword_counts, doc_counts
            
        except Exception as e:
            self.logger.error(f"Error processing folder {folder_path}: {str(e)}")
            return Counter(), Counter()

    def perform_chi_squared(self, pre_counts: Counter, post_counts: Counter) -> Tuple[float, float]:
        """
        Perform chi-squared test on keyword counts.
        
        Args:
            pre_counts: Counter object with pre-event counts
            post_counts: Counter object with post-event counts
            
        Returns:
            Tuple of (chi2 statistic, p-value)
        """
        contingency_table = [[pre_counts[k], post_counts[k]] for k in self.keywords]
        chi2, p, _, _ = chi2_contingency(contingency_table)
        return chi2, p

    def create_summary_dataframe(self, pre_counts: Counter, post_counts: Counter, 
                               pre_doc_counts: Counter, post_doc_counts: Counter) -> pd.DataFrame:
        """
        Create a summary DataFrame of all counts and changes.
        
        Returns:
            DataFrame with keyword statistics
        """
        data = []
        for keyword in self.keywords:
            data.append({
                'Keyword': keyword,
                'Pre_Count': pre_counts[keyword],
                'Post_Count': post_counts[keyword],
                'Count_Change': post_counts[keyword] - pre_counts[keyword],
                'Pre_Doc_Count': pre_doc_counts[keyword],
                'Post_Doc_Count': post_doc_counts[keyword],
                'Doc_Count_Change': post_doc_counts[keyword] - pre_doc_counts[keyword]
            })
        return pd.DataFrame(data)

    def analyze(self) -> Dict:
        """
        Perform complete analysis of keywords in both folders.
        
        Returns:
            Dictionary containing all analysis results
        """
        # Analyze both folders
        pre_counts, pre_doc_counts = self.analyze_folder(self.pre_folder)
        post_counts, post_doc_counts = self.analyze_folder(self.post_folder)
        
        # Perform statistical tests
        count_chi2, count_p = self.perform_chi_squared(pre_counts, post_counts)
        doc_chi2, doc_p = self.perform_chi_squared(pre_doc_counts, post_doc_counts)
        
        # Create summary DataFrame
        summary_df = self.create_summary_dataframe(
            pre_counts, post_counts, pre_doc_counts, post_doc_counts
        )
        
        return {
            'summary': summary_df,
            'count_statistics': {
                'chi2': count_chi2,
                'p_value': count_p,
                'significant': count_p < 0.05
            },
            'document_statistics': {
                'chi2': doc_chi2,
                'p_value': doc_p,
                'significant': doc_p < 0.05
            }
        }

# Example usage
if __name__ == "__main__":
    keywords = ["liquidity", "capital", "small bank"]
    pre_folder = "/Users/bluebird/develop/15.S08-applied-nlp-final/documents/pre_svb"
    post_folder = "/Users/bluebird/develop/15.S08-applied-nlp-final/documents/post_svb"
    
    analyzer = KeywordAnalyzer(keywords, pre_folder, post_folder)
    results = analyzer.analyze()
    
    # Print results
    print("\nKeyword Analysis Summary:")
    print(results['summary'].to_string(index=False))
    
    print("\nKeyword Count Statistics:")
    print(f"Chi-squared: {results['count_statistics']['chi2']:.2f}")
    print(f"P-value: {results['count_statistics']['p_value']:.4f}")
    print(f"Significant: {results['count_statistics']['significant']}")
    
    print("\nDocument Count Statistics:")
    print(f"Chi-squared: {results['document_statistics']['chi2']:.2f}")
    print(f"P-value: {results['document_statistics']['p_value']:.4f}")
    print(f"Significant: {results['document_statistics']['significant']}")