# Evaluation Metrics

## Overview

This document outlines the comprehensive evaluation framework for our LLM-based movie recommendation system, including model evaluation criteria, business metrics, A/B testing framework, and continuous monitoring strategies.

## Evaluation Framework

### Multi-Dimensional Evaluation
Our evaluation approach considers multiple dimensions of recommendation quality:

1. **Accuracy Metrics**: How well predictions match user preferences
2. **Ranking Metrics**: Quality of recommendation ordering
3. **Diversity Metrics**: Variety and novelty of recommendations
4. **Coverage Metrics**: Catalog coverage and fairness
5. **Business Metrics**: User engagement and satisfaction
6. **Explainability Metrics**: Quality of recommendation explanations

## Accuracy Metrics

### Rating Prediction Accuracy
```python
# src/evaluation/accuracy_metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple, Dict

class AccuracyMetrics:
    def __init__(self):
        self.metrics = {}
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Square Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def calculate_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Pearson and Spearman correlation coefficients."""
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr
        }
    
    def calculate_accuracy_at_threshold(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      threshold: float = 0.5) -> float:
        """Accuracy within threshold."""
        return np.mean(np.abs(y_true - y_pred) <= threshold)
    
    def evaluate_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all accuracy metrics."""
        metrics = {
            'rmse': self.calculate_rmse(y_true, y_pred),
            'mae': self.calculate_mae(y_true, y_pred),
            'mape': self.calculate_mape(y_true, y_pred),
            'accuracy_0.5': self.calculate_accuracy_at_threshold(y_true, y_pred, 0.5),
            'accuracy_1.0': self.calculate_accuracy_at_threshold(y_true, y_pred, 1.0),
        }
        
        correlations = self.calculate_correlation(y_true, y_pred)
        metrics.update(correlations)
        
        return metrics
```

## Ranking Metrics

### Information Retrieval Metrics
```python
# src/evaluation/ranking_metrics.py
import numpy as np
from typing import List, Set, Dict

class RankingMetrics:
    def __init__(self):
        pass
    
    def precision_at_k(self, recommended: List[int], relevant: Set[int], k: int) -> float:
        """Precision at K."""
        if k == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_recommended = len([item for item in recommended_k if item in relevant])
        
        return relevant_recommended / k
    
    def recall_at_k(self, recommended: List[int], relevant: Set[int], k: int) -> float:
        """Recall at K."""
        if len(relevant) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_recommended = len([item for item in recommended_k if item in relevant])
        
        return relevant_recommended / len(relevant)
    
    def f1_at_k(self, recommended: List[int], relevant: Set[int], k: int) -> float:
        """F1 Score at K."""
        precision = self.precision_at_k(recommended, relevant, k)
        recall = self.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(self, recommended: List[int], relevant: Set[int]) -> float:
        """Average Precision."""
        if len(relevant) == 0:
            return 0.0
        
        ap = 0.0
        relevant_count = 0
        
        for i, item in enumerate(recommended):
            if item in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap += precision_at_i
        
        return ap / len(relevant)
    
    def mean_average_precision(self, all_recommended: List[List[int]], 
                             all_relevant: List[Set[int]]) -> float:
        """Mean Average Precision across all users."""
        aps = []
        
        for recommended, relevant in zip(all_recommended, all_relevant):
            ap = self.average_precision(recommended, relevant)
            aps.append(ap)
        
        return np.mean(aps)
    
    def ndcg_at_k(self, recommended: List[int], relevance_scores: Dict[int, float], 
                  k: int) -> float:
        """Normalized Discounted Cumulative Gain at K."""
        if k == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            relevance = relevance_scores.get(item, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        sorted_relevance = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevance[:k]):
            idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def hit_rate_at_k(self, recommended: List[int], relevant: Set[int], k: int) -> float:
        """Hit Rate at K (binary relevance)."""
        recommended_k = recommended[:k]
        return 1.0 if any(item in relevant for item in recommended_k) else 0.0
    
    def mean_reciprocal_rank(self, all_recommended: List[List[int]], 
                           all_relevant: List[Set[int]]) -> float:
        """Mean Reciprocal Rank."""
        rrs = []
        
        for recommended, relevant in zip(all_recommended, all_relevant):
            rr = 0.0
            for i, item in enumerate(recommended):
                if item in relevant:
                    rr = 1.0 / (i + 1)
                    break
            rrs.append(rr)
        
        return np.mean(rrs)
```

## Diversity and Novelty Metrics

### Recommendation Diversity
```python
# src/evaluation/diversity_metrics.py
import numpy as np
from typing import List, Dict, Set
from collections import Counter
import pandas as pd

class DiversityMetrics:
    def __init__(self, movie_features: pd.DataFrame):
        self.movie_features = movie_features
    
    def intra_list_diversity(self, recommended: List[int], 
                           similarity_matrix: np.ndarray) -> float:
        """Average pairwise dissimilarity within recommendation list."""
        if len(recommended) < 2:
            return 0.0
        
        total_dissimilarity = 0.0
        pairs = 0
        
        for i in range(len(recommended)):
            for j in range(i + 1, len(recommended)):
                item_i, item_j = recommended[i], recommended[j]
                dissimilarity = 1 - similarity_matrix[item_i, item_j]
                total_dissimilarity += dissimilarity
                pairs += 1
        
        return total_dissimilarity / pairs if pairs > 0 else 0.0
    
    def genre_diversity(self, recommended: List[int]) -> float:
        """Genre diversity using Shannon entropy."""
        # Get genres for recommended movies
        movie_genres = []
        for movie_id in recommended:
            movie_info = self.movie_features[self.movie_features['movieId'] == movie_id]
            if not movie_info.empty:
                genres = movie_info.iloc[0]['genres'].split('|')
                movie_genres.extend(genres)
        
        if not movie_genres:
            return 0.0
        
        # Calculate genre distribution
        genre_counts = Counter(movie_genres)
        total_genres = len(movie_genres)
        
        # Shannon entropy
        entropy = 0.0
        for count in genre_counts.values():
            p = count / total_genres
            entropy -= p * np.log2(p)
        
        return entropy
    
    def temporal_diversity(self, recommended: List[int]) -> float:
        """Diversity across movie release years."""
        years = []
        for movie_id in recommended:
            movie_info = self.movie_features[self.movie_features['movieId'] == movie_id]
            if not movie_info.empty and 'year' in movie_info.columns:
                year = movie_info.iloc[0]['year']
                if pd.notna(year):
                    years.append(year)
        
        if len(years) < 2:
            return 0.0
        
        return np.std(years) / np.mean(years) if np.mean(years) > 0 else 0.0
    
    def novelty_score(self, recommended: List[int], popularity_scores: Dict[int, float]) -> float:
        """Average novelty (inverse popularity) of recommendations."""
        novelty_scores = []
        
        for movie_id in recommended:
            popularity = popularity_scores.get(movie_id, 0.0)
            # Novelty is inverse of popularity (log scale)
            novelty = -np.log(popularity + 1e-10)  # Add small epsilon to avoid log(0)
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def serendipity_score(self, recommended: List[int], user_profile: Dict[str, any], 
                         similarity_threshold: float = 0.7) -> float:
        """Serendipity: unexpected but relevant recommendations."""
        serendipitous_items = 0
        
        for movie_id in recommended:
            # Check if item is dissimilar to user's historical preferences
            # but still relevant (this is a simplified version)
            movie_info = self.movie_features[self.movie_features['movieId'] == movie_id]
            if not movie_info.empty:
                # Calculate similarity to user profile (simplified)
                similarity = self.calculate_profile_similarity(movie_info.iloc[0], user_profile)
                
                if similarity < similarity_threshold:  # Unexpected
                    # Check if it's still relevant (you'd need additional logic here)
                    serendipitous_items += 1
        
        return serendipitous_items / len(recommended) if recommended else 0.0
    
    def calculate_profile_similarity(self, movie: pd.Series, user_profile: Dict) -> float:
        """Calculate similarity between movie and user profile (simplified)."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated similarity measures
        movie_genres = set(movie['genres'].split('|'))
        user_genres = set(user_profile.get('favorite_genres', []))
        
        if not movie_genres or not user_genres:
            return 0.0
        
        intersection = len(movie_genres.intersection(user_genres))
        union = len(movie_genres.union(user_genres))
        
        return intersection / union if union > 0 else 0.0
```

## Coverage and Fairness Metrics

### Catalog Coverage
```python
# src/evaluation/coverage_metrics.py
import numpy as np
from typing import List, Set, Dict
from collections import Counter

class CoverageMetrics:
    def __init__(self, total_items: int):
        self.total_items = total_items
    
    def catalog_coverage(self, all_recommendations: List[List[int]]) -> float:
        """Percentage of catalog items that appear in recommendations."""
        recommended_items = set()
        
        for recommendations in all_recommendations:
            recommended_items.update(recommendations)
        
        return len(recommended_items) / self.total_items
    
    def gini_coefficient(self, all_recommendations: List[List[int]]) -> float:
        """Gini coefficient measuring recommendation concentration."""
        # Count frequency of each item in recommendations
        item_counts = Counter()
        for recommendations in all_recommendations:
            item_counts.update(recommendations)
        
        if not item_counts:
            return 0.0
        
        # Calculate Gini coefficient
        counts = sorted(item_counts.values())
        n = len(counts)
        
        if n == 0:
            return 0.0
        
        cumsum = np.cumsum(counts)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def long_tail_coverage(self, all_recommendations: List[List[int]], 
                          popularity_scores: Dict[int, float], 
                          tail_threshold: float = 0.2) -> float:
        """Coverage of long-tail (less popular) items."""
        # Identify long-tail items
        sorted_items = sorted(popularity_scores.items(), key=lambda x: x[1])
        tail_cutoff = int(len(sorted_items) * tail_threshold)
        tail_items = set([item for item, _ in sorted_items[:tail_cutoff]])
        
        # Count long-tail items in recommendations
        recommended_tail_items = set()
        for recommendations in all_recommendations:
            for item in recommendations:
                if item in tail_items:
                    recommended_tail_items.add(item)
        
        return len(recommended_tail_items) / len(tail_items) if tail_items else 0.0
    
    def genre_coverage(self, all_recommendations: List[List[int]], 
                      movie_genres: Dict[int, List[str]]) -> Dict[str, float]:
        """Coverage per genre."""
        genre_items = {}
        genre_recommended = {}
        
        # Group items by genre
        for movie_id, genres in movie_genres.items():
            for genre in genres:
                if genre not in genre_items:
                    genre_items[genre] = set()
                    genre_recommended[genre] = set()
                genre_items[genre].add(movie_id)
        
        # Count recommended items per genre
        for recommendations in all_recommendations:
            for movie_id in recommendations:
                if movie_id in movie_genres:
                    for genre in movie_genres[movie_id]:
                        genre_recommended[genre].add(movie_id)
        
        # Calculate coverage per genre
        coverage_by_genre = {}
        for genre in genre_items:
            total_items = len(genre_items[genre])
            recommended_items = len(genre_recommended[genre])
            coverage_by_genre[genre] = recommended_items / total_items if total_items > 0 else 0.0
        
        return coverage_by_genre
```

## Business Metrics

### User Engagement Metrics
```python
# src/evaluation/business_metrics.py
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class BusinessMetrics:
    def __init__(self, interaction_data: pd.DataFrame):
        self.interaction_data = interaction_data
    
    def click_through_rate(self, recommendation_logs: pd.DataFrame) -> float:
        """CTR: Percentage of recommendations that were clicked."""
        total_recommendations = len(recommendation_logs)
        clicked_recommendations = len(recommendation_logs[recommendation_logs['clicked'] == True])
        
        return clicked_recommendations / total_recommendations if total_recommendations > 0 else 0.0
    
    def conversion_rate(self, recommendation_logs: pd.DataFrame) -> float:
        """Percentage of recommendations that led to ratings/purchases."""
        total_recommendations = len(recommendation_logs)
        converted_recommendations = len(recommendation_logs[recommendation_logs['converted'] == True])
        
        return converted_recommendations / total_recommendations if total_recommendations > 0 else 0.0
    
    def user_retention_rate(self, user_activity: pd.DataFrame, 
                          period_days: int = 30) -> float:
        """Percentage of users who return within specified period."""
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Users active before cutoff
        early_users = user_activity[user_activity['last_activity'] < cutoff_date]['user_id'].unique()
        
        # Users active after cutoff
        recent_users = user_activity[user_activity['last_activity'] >= cutoff_date]['user_id'].unique()
        
        # Retained users (active in both periods)
        retained_users = set(early_users).intersection(set(recent_users))
        
        return len(retained_users) / len(early_users) if len(early_users) > 0 else 0.0
    
    def average_session_duration(self, session_data: pd.DataFrame) -> float:
        """Average time users spend per session."""
        session_durations = session_data['session_duration'].dropna()
        return session_durations.mean() if len(session_durations) > 0 else 0.0
    
    def recommendation_acceptance_rate(self, feedback_data: pd.DataFrame) -> float:
        """Percentage of recommendations rated positively."""
        total_rated = len(feedback_data[feedback_data['rating'].notna()])
        positive_ratings = len(feedback_data[feedback_data['rating'] >= 4.0])
        
        return positive_ratings / total_rated if total_rated > 0 else 0.0
    
    def user_satisfaction_score(self, survey_data: pd.DataFrame) -> float:
        """Average user satisfaction score from surveys."""
        satisfaction_scores = survey_data['satisfaction_score'].dropna()
        return satisfaction_scores.mean() if len(satisfaction_scores) > 0 else 0.0
    
    def revenue_impact(self, purchase_data: pd.DataFrame, 
                      recommendation_data: pd.DataFrame) -> Dict[str, float]:
        """Revenue metrics related to recommendations."""
        # Revenue from recommended items
        recommended_purchases = purchase_data[
            purchase_data['movie_id'].isin(recommendation_data['movie_id'])
        ]
        
        total_revenue = purchase_data['revenue'].sum()
        recommendation_revenue = recommended_purchases['revenue'].sum()
        
        return {
            'total_revenue': total_revenue,
            'recommendation_revenue': recommendation_revenue,
            'recommendation_revenue_percentage': (recommendation_revenue / total_revenue * 100) if total_revenue > 0 else 0.0
        }
```

## A/B Testing Framework

### Experiment Design
```python
# src/evaluation/ab_testing.py
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd

class ABTestFramework:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def design_experiment(self, effect_size: float, power: float = 0.8, 
                         baseline_rate: float = 0.1) -> int:
        """Calculate required sample size for A/B test."""
        # Using Cohen's h for proportions
        h = 2 * (np.arcsin(np.sqrt(baseline_rate + effect_size)) - 
                 np.arcsin(np.sqrt(baseline_rate)))
        
        # Sample size calculation
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        z_beta = stats.norm.ppf(power)
        
        n = ((z_alpha + z_beta) / h) ** 2
        return int(np.ceil(n))
    
    def run_proportion_test(self, control_successes: int, control_total: int,
                          treatment_successes: int, treatment_total: int) -> Dict:
        """Run two-proportion z-test."""
        # Calculate proportions
        p1 = control_successes / control_total
        p2 = treatment_successes / treatment_total
        
        # Pooled proportion
        p_pool = (control_successes + treatment_successes) / (control_total + treatment_total)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))
        
        # Z-statistic
        z_stat = (p2 - p1) / se
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Confidence interval for difference
        se_diff = np.sqrt(p1 * (1 - p1) / control_total + p2 * (1 - p2) / treatment_total)
        margin_error = stats.norm.ppf(1 - self.significance_level / 2) * se_diff
        ci_lower = (p2 - p1) - margin_error
        ci_upper = (p2 - p1) + margin_error
        
        return {
            'control_rate': p1,
            'treatment_rate': p2,
            'difference': p2 - p1,
            'relative_improvement': ((p2 - p1) / p1 * 100) if p1 > 0 else 0,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'confidence_interval': (ci_lower, ci_upper)
        }
    
    def run_ttest(self, control_values: List[float], 
                  treatment_values: List[float]) -> Dict:
        """Run independent t-test for continuous metrics."""
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
        
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values, ddof=1) + 
                             (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) / 
                            (len(control_values) + len(treatment_values) - 2))
        
        cohens_d = (treatment_mean - control_mean) / pooled_std
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'difference': treatment_mean - control_mean,
            'relative_improvement': ((treatment_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'effect_size': cohens_d
        }
    
    def sequential_testing(self, control_data: List[float], 
                          treatment_data: List[float], 
                          min_sample_size: int = 100) -> Dict:
        """Sequential testing with early stopping."""
        results = []
        
        for i in range(min_sample_size, min(len(control_data), len(treatment_data)) + 1):
            control_subset = control_data[:i]
            treatment_subset = treatment_data[:i]
            
            result = self.run_ttest(control_subset, treatment_subset)
            result['sample_size'] = i
            results.append(result)
            
            # Early stopping if significant
            if result['significant'] and result['p_value'] < self.significance_level / 2:
                break
        
        return {
            'final_result': results[-1],
            'all_results': results,
            'early_stop': len(results) < (min(len(control_data), len(treatment_data)) - min_sample_size + 1)
        }
```

## Comprehensive Evaluation Pipeline

### Evaluation Orchestrator
```python
# src/evaluation/evaluator.py
from typing import Dict, List, Any
import pandas as pd
import numpy as np

class RecommendationEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accuracy_metrics = AccuracyMetrics()
        self.ranking_metrics = RankingMetrics()
        self.diversity_metrics = DiversityMetrics(config['movie_features'])
        self.coverage_metrics = CoverageMetrics(config['total_items'])
        self.business_metrics = BusinessMetrics(config['interaction_data'])
    
    def evaluate_model(self, predictions: pd.DataFrame, 
                      ground_truth: pd.DataFrame,
                      recommendations: Dict[int, List[int]]) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        
        results = {}
        
        # Accuracy metrics
        y_true = ground_truth['rating'].values
        y_pred = predictions['predicted_rating'].values
        results['accuracy'] = self.accuracy_metrics.evaluate_all(y_true, y_pred)
        
        # Ranking metrics
        all_recommended = list(recommendations.values())
        all_relevant = [set(ground_truth[ground_truth['user_id'] == uid]['movie_id']) 
                       for uid in recommendations.keys()]
        
        results['ranking'] = {
            'map': self.ranking_metrics.mean_average_precision(all_recommended, all_relevant),
            'mrr': self.ranking_metrics.mean_reciprocal_rank(all_recommended, all_relevant),
            'ndcg@10': np.mean([
                self.ranking_metrics.ndcg_at_k(rec, {}, 10) 
                for rec in all_recommended
            ])
        }
        
        # Diversity metrics
        results['diversity'] = {
            'genre_diversity': np.mean([
                self.diversity_metrics.genre_diversity(rec) 
                for rec in all_recommended
            ]),
            'novelty': self.diversity_metrics.novelty_score(
                [item for sublist in all_recommended for item in sublist],
                self.config['popularity_scores']
            )
        }
        
        # Coverage metrics
        results['coverage'] = {
            'catalog_coverage': self.coverage_metrics.catalog_coverage(all_recommended),
            'gini_coefficient': self.coverage_metrics.gini_coefficient(all_recommended)
        }
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models across all metrics."""
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'model': model_name}
            
            # Flatten nested results
            for category, metrics in results.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        row[f"{category}_{metric_name}"] = value
                else:
                    row[category] = metrics
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report."""
        
        report = "# Model Evaluation Report\n\n"
        
        # Accuracy section
        report += "## Accuracy Metrics\n"
        for metric, value in results['accuracy'].items():
            report += f"- {metric.upper()}: {value:.4f}\n"
        
        # Ranking section
        report += "\n## Ranking Metrics\n"
        for metric, value in results['ranking'].items():
            report += f"- {metric.upper()}: {value:.4f}\n"
        
        # Diversity section
        report += "\n## Diversity Metrics\n"
        for metric, value in results['diversity'].items():
            report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        # Coverage section
        report += "\n## Coverage Metrics\n"
        for metric, value in results['coverage'].items():
            report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
        
        return report
```

## Next Steps

1. **Real-time Evaluation**: Implement online evaluation metrics
2. **Multi-objective Optimization**: Balance competing metrics
3. **Fairness Metrics**: Add bias and fairness evaluation
4. **Causal Inference**: Implement causal impact measurement
5. **User Studies**: Conduct qualitative user experience studies

This comprehensive evaluation framework ensures our recommendation system is assessed across all critical dimensions of performance, user satisfaction, and business impact.
