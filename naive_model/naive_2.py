import os
import re
from collections import Counter, defaultdict
from scipy.stats import poisson, chi2
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
KEYWORDS = ["liquidity", "capital", "capital requirement", "small bank", "mid-sized bank", "community bank", "interest rate risk", "concentration"]
OUTPUT_DIR = "./naive_model/results"
MIN_KEYWORD_THRESHOLD = 3  # Minimum number of times a keyword must appear in a document to be counted

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
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

    def count_keywords_in_text(self, text: str) -> Dict[str, int]:
        """
        Count keyword occurrences in text, including common variations.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with keyword counts for this document
        """
        counts = {}
        text = text.lower()
        
        for keyword in self.keywords:
            # Split multi-word keywords
            words = keyword.split()
            
            if len(words) == 1:
                # Single word - match word + common endings
                pattern = r'\b' + re.escape(words[0]) + r'(?:s|ing|er|ers|ed)?\b'
            else:
                # Multi-word - match last word + common endings
                pattern = r'\b' + r'\s+'.join(
                    [re.escape(w) if i < len(words)-1 
                    else re.escape(w) + r'(?:s|ing|er|ers|ed)?\b'
                    for i, w in enumerate(words)]
                )
            
            counts[keyword] = len(re.findall(pattern, text))
        
        return counts

    def analyze_folder(self, folder_path: Path) -> Tuple[Counter, int]:
        """
        Analyze all documents in a folder for keyword occurrences.
        Count documents where keywords appear at least MIN_KEYWORD_THRESHOLD times.
        
        Args:
            folder_path: Path to the folder to analyze
            
        Returns:
            Tuple of (document_counts, total_documents)
        """
        document_counts = Counter()  # Count of documents with >= threshold mentions
        
        try:
            files = [f for f in folder_path.glob("*.htm")]
            total_documents = len(files)
            self.logger.info(f"Processing {total_documents} files in {folder_path}")
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read().lower()
                        doc_counts = self.count_keywords_in_text(text)
                        
                        # Count this document for keywords that appear >= threshold times
                        for keyword, count in doc_counts.items():
                            if count >= MIN_KEYWORD_THRESHOLD:
                                document_counts[keyword] += 1
                                
                except Exception as e:
                    self.logger.error(f"Error reading file {file_path}: {str(e)}")
                    
            return document_counts, total_documents
            
        except Exception as e:
            self.logger.error(f"Error processing folder {folder_path}: {str(e)}")
            return Counter(), 0

    def perform_poisson_analysis(self,
                               pre_counts: Counter,
                               post_counts: Counter,
                               total_pre_docs: int,
                               total_post_docs: int) -> Dict:
        """
        Perform Poisson regression analysis of document frequencies.
        
        Args:
            pre_counts: Counter of documents with >= threshold keyword occurrences pre-event
            post_counts: Counter of documents with >= threshold keyword occurrences post-event
            total_pre_docs: Total number of pre-event documents
            total_post_docs: Total number of post-event documents
            
        Returns:
            Dictionary containing Poisson analysis results for each keyword
        """
        results = {}
        
        # Calculate exposure (document counts)
        pre_exposure = total_pre_docs
        post_exposure = total_post_docs
        
        for keyword in self.keywords:
            # Calculate rates (documents with >= threshold mentions / total documents)
            pre_rate = pre_counts[keyword] / pre_exposure if pre_exposure > 0 else 0
            post_rate = post_counts[keyword] / post_exposure if post_exposure > 0 else 0
            
            # Calculate mean rates for Poisson test
            pooled_rate = (pre_counts[keyword] + post_counts[keyword]) / (pre_exposure + post_exposure)
            
            # Expected counts under null hypothesis
            expected_pre = pooled_rate * pre_exposure
            expected_post = pooled_rate * post_exposure
            
            # Calculate Poisson likelihood ratio test statistic
            if pre_counts[keyword] > 0 and post_counts[keyword] > 0:
                lr_stat = (2 * (pre_counts[keyword] * np.log(pre_counts[keyword] / expected_pre) +
                              post_counts[keyword] * np.log(post_counts[keyword] / expected_post)))
                p_value = 1 - chi2.cdf(lr_stat, df=1)
            else:
                lr_stat = 0
                p_value = 1.0
            
            # Store results
            results[keyword] = {
                'pre_rate': pre_rate,
                'post_rate': post_rate,
                'rate_change': post_rate - pre_rate,
                'rate_ratio': post_rate / pre_rate if pre_rate > 0 else float('inf'),
                'lr_statistic': lr_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
        return results

    def create_summary_dataframe(self, 
                               pre_counts: Counter, 
                               post_counts: Counter,
                               total_pre_docs: int,
                               total_post_docs: int,
                               poisson_results: Dict) -> pd.DataFrame:
        """
        Create a summary DataFrame of document counts and Poisson analysis results.
        """
        data = []
        for keyword in self.keywords:
            stats = poisson_results[keyword]
            
            data.append({
                'Keyword': keyword,
                'Pre_Documents': pre_counts[keyword],  # Number of documents with >= threshold mentions
                'Post_Documents': post_counts[keyword],
                'Total_Pre_Documents': total_pre_docs,
                'Total_Post_Documents': total_post_docs,
                'Pre_Rate': stats['pre_rate'],
                'Post_Rate': stats['post_rate'],
                'Rate_Change': stats['rate_change'],
                'Rate_Ratio': stats['rate_ratio'],
                'LR_Statistic': stats['lr_statistic'],
                'P_Value': stats['p_value'],
                'Significant': stats['significant']
            })
        
        return pd.DataFrame(data)

    def create_visualizations(self, df: pd.DataFrame, metadata: Dict):
        """
        Create visualizations for the Poisson regression analysis.
        """
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = GridSpec(2, 2)
        
        # 1. Document Count Comparison (spanning full width)
        ax1 = fig.add_subplot(gs[0, :])
        df_melt = pd.melt(df, 
                         id_vars=['Keyword'], 
                         value_vars=['Pre_Documents', 'Post_Documents'],
                         var_name='Period', 
                         value_name='Documents')
        
        colors = ['#0158BF', '#76C6FC', '#00143F']
        bar_width = 0.35
        x = np.arange(len(df))
        
        bars1 = ax1.bar(x - bar_width/2, df['Pre_Documents'], 
                       bar_width, label='Pre', color=colors[0])
        bars2 = ax1.bar(x + bar_width/2, df['Post_Documents'], 
                       bar_width, label='Post', color=colors[1])
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Keyword'], rotation=45)
        plt.setp(ax1.get_xticklabels(), horizontalalignment='right')
        ax1.set_title(f'Documents with ≥{MIN_KEYWORD_THRESHOLD} Keyword Mentions (Pre/Post Comparison)')
        ax1.set_ylabel('Number of Documents')
        ax1.legend(title='Period')
        
        # Add value labels on top of bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)}',
                        horizontalalignment='center', verticalalignment='bottom')
        
        # 2. Rate Change
        ax2 = fig.add_subplot(gs[1, 0])
        rate_change = df[['Keyword', 'Rate_Change']].copy()
        bars = ax2.bar(rate_change['Keyword'], rate_change['Rate_Change'],
                      color=[colors[2] if sig else 'gray' 
                            for sig in df['Significant']])
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Change in Document Rate')
        ax2.tick_params(axis='x', rotation=45)
        plt.setp(ax2.get_xticklabels(), horizontalalignment='right')
        ax2.set_ylabel('Change in Rate')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.3f}',
                    horizontalalignment='center', 
                    verticalalignment='bottom' if height > 0 else 'top')

        # 3. Rate Ratio (log scale with special handling for inf/zero cases)
        ax3 = fig.add_subplot(gs[1, 1])
        rate_ratio = df[['Keyword', 'Rate_Ratio', 'Pre_Documents', 'Post_Documents']].copy()
        
        # Create custom ratio labels that show "Inf" for division by zero
        # and "N/A" for 0/0 cases
        ratio_labels = []
        for _, row in rate_ratio.iterrows():
            if row['Pre_Documents'] == 0 and row['Post_Documents'] == 0:
                ratio_labels.append('N/A (0/0)')
            elif row['Pre_Documents'] == 0:
                ratio_labels.append('Inf (→/0)')
            else:
                ratio_labels.append(f'{row["Rate_Ratio"]:.2f}')
        
        # For plotting, replace inf with a large number and 0/0 with 1
        plotting_ratios = []
        for _, row in rate_ratio.iterrows():
            if row['Pre_Documents'] == 0 and row['Post_Documents'] == 0:
                plotting_ratios.append(1)  # Plot at y=1 for N/A cases
            elif row['Pre_Documents'] == 0:
                plotting_ratios.append(100)  # Cap infinite ratios at 100 for visualization
            else:
                plotting_ratios.append(min(row['Rate_Ratio'], 100))  # Cap large finite ratios at 100
                
        bars = ax3.bar(rate_ratio['Keyword'], plotting_ratios,
                      color=[colors[2] if sig else 'gray' 
                            for sig in df['Significant']])
        ax3.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Document Rate Ratio (Post/Pre)')
        ax3.tick_params(axis='x', rotation=45)
        plt.setp(ax3.get_xticklabels(), horizontalalignment='right')
        ax3.set_ylabel('Rate Ratio (capped at 100)')
        ax3.set_yscale('log')
        
        # Set reasonable y-axis limits
        ax3.set_ylim(0.1, 200)  # Show one order of magnitude below and above 1
        
        # Add value labels on top of bars with special handling
        for bar, label in zip(bars, ratio_labels):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, 
                    height * 1.1,  # Place slightly above bar
                    label,
                    horizontalalignment='center', 
                    verticalalignment='bottom',
                    rotation=0)

        # Adjust layout
        plt.tight_layout()
        return fig

    def analyze(self) -> Dict:
        """
        Perform complete Poisson regression analysis of keywords.
        """
        # Analyze both folders
        pre_counts, total_pre_docs = self.analyze_folder(self.pre_folder)
        post_counts, total_post_docs = self.analyze_folder(self.post_folder)
        
        # Perform Poisson analysis
        poisson_results = self.perform_poisson_analysis(
            pre_counts, post_counts,
            total_pre_docs, total_post_docs
        )
        
        # Create summary DataFrame
        summary_df = self.create_summary_dataframe(
            pre_counts, post_counts,
            total_pre_docs, total_post_docs,
            poisson_results
        )
        
        # Create metadata
        metadata = {
            'total_pre_documents': total_pre_docs,
            'total_post_documents': total_post_docs,
            'threshold_pre_documents': {k: v for k, v in pre_counts.items()},
            'threshold_post_documents': {k: v for k, v in post_counts.items()}
        }
        
        # Create visualizations
        visualization = self.create_visualizations(summary_df, metadata)
        
        return {
            'summary_df': summary_df,
            'poisson_results': poisson_results,
            'metadata': metadata,
            'visualization': visualization
        }

    def save_results(self, results: Dict, output_dir: str = '.'):
        """
        Save analysis results to files.
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
            'poisson_results': results['poisson_results']
        }
        
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
    
    print(f"\nAnalysis complete. Results have been saved to the output directory.")
    print(f"Note: Only counting documents with {MIN_KEYWORD_THRESHOLD}+ mentions of each keyword")

if __name__ == "__main__":
    main()