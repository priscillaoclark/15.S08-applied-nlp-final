import os
import re
from collections import Counter
from scipy.stats import chi2_contingency, fisher_exact, binomtest
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import json

# Configuration
PRE_FOLDER = "/Users/bluebird/develop/15.S08-applied-nlp-final/documents/pre_svb"
POST_FOLDER = "/Users/bluebird/develop/15.S08-applied-nlp-final/documents/post_svb"
KEYWORDS = ["liquidity", "capital", "small bank"]
OUTPUT_DIR = "./naive_model/results"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        return super().default(obj)

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

    def analyze_folder(self, folder_path: Path) -> Tuple[Counter, Counter, int]:
        """
        Analyze all documents in a folder for keyword occurrences and document counts.
        
        Args:
            folder_path: Path to the folder to analyze
            
        Returns:
            Tuple of (keyword_counts, document_counts, total_documents)
        """
        keyword_counts = Counter()
        doc_counts = Counter()
        
        try:
            files = [f for f in folder_path.glob("*.htm")]
            total_documents = len(files)
            self.logger.info(f"Processing {total_documents} files in {folder_path}")
            
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
                        
            return keyword_counts, doc_counts, total_documents
            
        except Exception as e:
            self.logger.error(f"Error processing folder {folder_path}: {str(e)}")
            return Counter(), Counter(), 0

    def perform_statistical_analysis(self,
                                  pre_counts: Counter,
                                  post_counts: Counter,
                                  pre_doc_counts: Counter,
                                  post_doc_counts: Counter,
                                  total_pre_docs: int,
                                  total_post_docs: int) -> Dict:
        """
        Perform comprehensive statistical analysis of keyword frequencies and document proportions.
        
        Args:
            pre_counts: Counter of keyword occurrences in pre-event documents
            post_counts: Counter of keyword occurrences in post-event documents
            pre_doc_counts: Counter of documents containing each keyword pre-event
            post_doc_counts: Counter of documents containing each keyword post-event
            total_pre_docs: Total number of pre-event documents
            total_post_docs: Total number of post-event documents
            
        Returns:
            Dictionary containing frequency and document analysis results for each keyword
        """
        results = {}
        
        # Calculate totals for frequency analysis
        pre_total = sum(pre_counts.values())
        post_total = sum(post_counts.values())
        
        for keyword in self.keywords:
            # Word frequency chi-square test
            freq_table = np.array([
                [pre_counts[keyword], pre_total - pre_counts[keyword]],
                [post_counts[keyword], post_total - post_counts[keyword]]
            ])
            
            # Handle zero counts
            if np.any(freq_table == 0):
                # Add small constant to avoid zero cells
                freq_table = freq_table + 0.5
                
            freq_chi2, freq_p, _, _ = chi2_contingency(freq_table, correction=True)
            
            # Document proportion tests
            doc_table = np.array([
                [pre_doc_counts[keyword], total_pre_docs - pre_doc_counts[keyword]],
                [post_doc_counts[keyword], total_post_docs - post_doc_counts[keyword]]
            ])
            
            # Calculate Fisher's exact test
            fisher_odds_ratio, fisher_p = fisher_exact(doc_table)
            
            # Calculate proportions for binomial test
            pre_prop = pre_doc_counts[keyword] / total_pre_docs if total_pre_docs > 0 else 0
            
            # Perform binomial test
            if total_post_docs > 0:
                binom_result = binomtest(
                    k=post_doc_counts[keyword],
                    n=total_post_docs,
                    p=pre_prop if pre_prop > 0 else 0.5  # Use 0.5 if no pre-event data
                )
                binom_p = binom_result.pvalue
            else:
                binom_p = 1.0
            
            # Store results
            results[keyword] = {
                'frequency_analysis': {
                    'pre_frequency': pre_counts[keyword] / pre_total if pre_total > 0 else 0,
                    'post_frequency': post_counts[keyword] / post_total if post_total > 0 else 0,
                    'frequency_change': (post_counts[keyword] / post_total if post_total > 0 else 0) - 
                                     (pre_counts[keyword] / pre_total if pre_total > 0 else 0),
                    'chi2': freq_chi2,
                    'p_value': freq_p,
                    'significant': freq_p < 0.05
                },
                'document_analysis': {
                    'pre_proportion': pre_doc_counts[keyword] / total_pre_docs if total_pre_docs > 0 else 0,
                    'post_proportion': post_doc_counts[keyword] / total_post_docs if total_post_docs > 0 else 0,
                    'proportion_change': (post_doc_counts[keyword] / total_post_docs if total_post_docs > 0 else 0) -
                                      (pre_doc_counts[keyword] / total_pre_docs if total_pre_docs > 0 else 0),
                    'fisher_p': fisher_p,
                    'fisher_odds_ratio': fisher_odds_ratio,
                    'binomial_p': binom_p,
                    'significant': fisher_p < 0.05 or binom_p < 0.05
                }
            }
            
        return results

    def create_summary_dataframe(self, 
                               pre_counts: Counter, 
                               post_counts: Counter,
                               pre_doc_counts: Counter,
                               post_doc_counts: Counter,
                               total_pre_docs: int,
                               total_post_docs: int,
                               statistical_results: Dict) -> pd.DataFrame:
        """
        Create a summary DataFrame of all counts and statistical results.
        """
        data = []
        for keyword in self.keywords:
            freq_stats = statistical_results[keyword]['frequency_analysis']
            doc_stats = statistical_results[keyword]['document_analysis']
            
            data.append({
                'Keyword': keyword,
                'Pre_Count': pre_counts[keyword],
                'Post_Count': post_counts[keyword],
                'Pre_Doc_Count': pre_doc_counts[keyword],
                'Post_Doc_Count': post_doc_counts[keyword],
                'Count_Change': post_counts[keyword] - pre_counts[keyword],
                'Count_Change_Pct': ((post_counts[keyword] / pre_counts[keyword]) - 1) * 100 if pre_counts[keyword] > 0 else np.inf,
                'Pre_Frequency': freq_stats['pre_frequency'],
                'Post_Frequency': freq_stats['post_frequency'],
                'Frequency_Change': freq_stats['frequency_change'] * 100,
                'Word_Freq_P_Value': freq_stats['p_value'],
                'Word_Freq_Significant': freq_stats['significant'],
                'Pre_Doc_Proportion': doc_stats['pre_proportion'],
                'Post_Doc_Proportion': doc_stats['post_proportion'],
                'Doc_Proportion_Change': doc_stats['proportion_change'] * 100,
                'Fisher_P_Value': doc_stats['fisher_p'],
                'Fisher_Odds_Ratio': doc_stats['fisher_odds_ratio'],
                'Binomial_P_Value': doc_stats['binomial_p'],
                'Doc_Analysis_Significant': doc_stats['significant']
            })
        
        return pd.DataFrame(data)

    def create_visualizations(self, df: pd.DataFrame, metadata: Dict):
        """
        Create a comprehensive visualization dashboard for the keyword analysis.
        """
        # Use a default matplotlib style instead of seaborn
        plt.style.use('default')  # or 'classic' if you prefer
        
        # Set the figure background to white
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = GridSpec(2, 3, figure=fig)
        #fig.suptitle('Keyword Analysis Dashboard', fontsize=16, y=0.95)

        # 1. Raw Count Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        df_melt = pd.melt(df, 
                         id_vars=['Keyword'], 
                         value_vars=['Pre_Count', 'Post_Count'],
                         var_name='Period', 
                         value_name='Count')
        
        # Use standard bar plot with custom colors
        colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
        bars = ax1.bar(x=np.arange(len(df)) - 0.2, 
                      height=df['Pre_Count'],
                      width=0.4,
                      label='Pre',
                      color=colors[0])
        bars = ax1.bar(x=np.arange(len(df)) + 0.2, 
                      height=df['Post_Count'],
                      width=0.4,
                      label='Post',
                      color=colors[1])
        
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['Keyword'], rotation=45)
        ax1.set_title('Raw Keyword Counts')
        ax1.set_ylabel('Number of Occurrences')
        ax1.legend()

        # 2. Frequency Change
        ax2 = fig.add_subplot(gs[0, 1])
        freq_change = df[['Keyword', 'Frequency_Change']].copy()
        bars = ax2.bar(freq_change['Keyword'], freq_change['Frequency_Change'],
                      color=['darkred' if sig else 'gray' 
                            for sig in df['Word_Freq_Significant']])
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Change in Keyword Frequency')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylabel('Change in Frequency (%)')

        # 3. Document Proportion Change
        ax3 = fig.add_subplot(gs[0, 2])
        doc_change = df[['Keyword', 'Doc_Proportion_Change']].copy()
        bars = ax3.bar(doc_change['Keyword'], doc_change['Doc_Proportion_Change'],
                      color=['darkred' if sig else 'gray' 
                            for sig in df['Doc_Analysis_Significant']])
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Change in Document Presence')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylabel('Change in Document Proportion (%)')

        # 4. Statistical Significance Heatmap
        ax4 = fig.add_subplot(gs[1, 0])
        sig_data = df[['Keyword', 'Word_Freq_P_Value', 'Fisher_P_Value', 'Binomial_P_Value']].copy()
        sig_data = sig_data.set_index('Keyword')
        sig_data = -np.log10(sig_data)  # Transform p-values for better visualization
        
        im = ax4.imshow(sig_data.T, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im, ax=ax4, label='-log10(p-value)')
        
        # Set labels
        ax4.set_xticks(range(len(sig_data.index)))
        ax4.set_xticklabels(sig_data.index, rotation=45)
        ax4.set_yticks(range(len(sig_data.columns)))
        ax4.set_yticklabels(sig_data.columns)
        
        # Add value annotations
        for i in range(len(sig_data.index)):
            for j in range(len(sig_data.columns)):
                ax4.text(i, j, f'{sig_data.iloc[i, j]:.2f}',
                        ha='center', va='center')
        
        ax4.set_title('Statistical Significance\n(-log10 p-value)')

        # 5. Odds Ratios
        ax5 = fig.add_subplot(gs[1, 1])
        odds_data = df[['Keyword', 'Fisher_Odds_Ratio']].copy()
        bars = ax5.bar(odds_data['Keyword'], odds_data['Fisher_Odds_Ratio'])
        ax5.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
        ax5.set_title('Fisher\'s Exact Test Odds Ratios')
        ax5.tick_params(axis='x', rotation=45)
        ax5.set_ylabel('Odds Ratio')
        ax5.set_yscale('log')
        
        # 6. Summary Statistics Table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        summary_text = [
            'Analysis Summary:',
            f"Total Pre-Event Documents: {metadata['total_pre_documents']}",
            f"Total Post-Event Documents: {metadata['total_post_documents']}",
            f"Total Pre-Event Keywords: {metadata['total_pre_keywords']}",
            f"Total Post-Event Keywords: {metadata['total_post_keywords']}",
            '\nSignificant Changes:',
            f"Word Frequency: {df['Word_Freq_Significant'].sum()}/{len(df)} keywords",
            f"Document Presence: {df['Doc_Analysis_Significant'].sum()}/{len(df)} keywords"
        ]
        ax6.text(0, 0.8, '\n'.join(summary_text), fontsize=10, verticalalignment='top')

        plt.tight_layout()
        return fig

    def analyze(self) -> Dict:
            """
            Perform complete analysis of keywords in both folders.
            """
            # Analyze both folders
            pre_counts, pre_doc_counts, total_pre_docs = self.analyze_folder(self.pre_folder)
            post_counts, post_doc_counts, total_post_docs = self.analyze_folder(self.post_folder)
            
            # Perform statistical analysis
            statistical_results = self.perform_statistical_analysis(
                pre_counts, post_counts,
                pre_doc_counts, post_doc_counts,
                total_pre_docs, total_post_docs
            )
            
            # Create summary DataFrame
            summary_df = self.create_summary_dataframe(
                pre_counts, post_counts,
                pre_doc_counts, post_doc_counts,
                total_pre_docs, total_post_docs,
                statistical_results
            )
            
            # Create metadata
            metadata = {
                'total_pre_documents': total_pre_docs,
                'total_post_documents': total_post_docs,
                'total_pre_keywords': sum(pre_counts.values()),
                'total_post_keywords': sum(post_counts.values())
            }
            
            # Create visualizations
            visualization = self.create_visualizations(summary_df, metadata)
            
            return {
                'summary_df': summary_df,
                'detailed_statistics': statistical_results,
                'metadata': metadata,
                'visualization': visualization
            }

    def save_results(self, results: Dict, output_dir: str = '.'):
        """
        Save analysis results to files.
        
        Args:
            results: Dictionary containing analysis results
            output_dir: Directory to save results (default: current directory)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame to CSV
        results['summary_df'].to_csv(output_path / 'keyword_analysis_summary.csv', index=False)
        
        # Save visualization
        results['visualization'].savefig(
            output_path / 'keyword_analysis_dashboard.png', 
            dpi=300, 
            bbox_inches='tight'
        )
        
        # Save detailed statistics to JSON
        stats_to_save = {
            'metadata': results['metadata'],
            'statistics': {}
        }
        
        # Simplify the statistics structure
        for keyword, stats in results['detailed_statistics'].items():
            stats_to_save['statistics'][keyword] = {
                'frequency': {
                    'pre': float(stats['frequency_analysis']['pre_frequency']),
                    'post': float(stats['frequency_analysis']['post_frequency']),
                    'change': float(stats['frequency_analysis']['frequency_change']),
                    'chi2': float(stats['frequency_analysis']['chi2']),
                    'p_value': float(stats['frequency_analysis']['p_value']),
                    'significant': bool(stats['frequency_analysis']['significant'])
                },
                'document': {
                    'pre_proportion': float(stats['document_analysis']['pre_proportion']),
                    'post_proportion': float(stats['document_analysis']['post_proportion']),
                    'proportion_change': float(stats['document_analysis']['proportion_change']),
                    'fisher_p': float(stats['document_analysis']['fisher_p']),
                    'fisher_odds_ratio': float(stats['document_analysis']['fisher_odds_ratio']),
                    'binomial_p': float(stats['document_analysis']['binomial_p']),
                    'significant': bool(stats['document_analysis']['significant'])
                }
            }
        
        # Save with custom encoder
        with open(output_path / 'detailed_statistics.json', 'w') as f:
            json.dump(stats_to_save, f, indent=2, cls=NumpyEncoder)

def main():
    """
    Main function to run the keyword analysis with predefined parameters.
    """    
    # Create analyzer
    analyzer = KeywordAnalyzer(KEYWORDS, PRE_FOLDER, POST_FOLDER)
    
    # Run analysis
    print("Running keyword analysis...")
    results = analyzer.analyze()
    
    # Save results
    print(f"\nSaving results to {OUTPUT_DIR}")
    analyzer.save_results(results, OUTPUT_DIR)
    
    # Display summary
    print("\nKeyword Analysis Summary:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results['summary_df'].to_string(index=False))
    
    print("\nAnalysis complete. Results have been saved to the output directory.")

if __name__ == "__main__":
    main()