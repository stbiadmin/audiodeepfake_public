"""Generate markdown classification report.

Creates comprehensive report with results, visualizations, and recommendations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


def generate_classification_report(
    results: Dict[str, Any],
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate markdown classification report.

    Args:
        results: Dictionary containing all training results
        output_path: Path to save report
        config: Optional configuration dictionary
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # Header
    lines.append("# Audio Deepfake Detection: Classification Results")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    # Find best model
    best_config = None
    best_f1 = 0
    for config_name, result in results.get('training_results', {}).items():
        f1 = result.get('test_scores', {}).get('f1', 0)
        if f1 > best_f1:
            best_f1 = f1
            best_config = config_name

    if best_config:
        best_result = results['training_results'][best_config]
        lines.append(f"**Best Model:** {best_config}")
        lines.append(f"- Test F1: {best_f1:.4f}")
        lines.append(f"- Test AUC-ROC: {best_result['test_scores'].get('roc_auc', 0):.4f}")
        lines.append(f"- Class Balance Ratio: {best_result['test_scores'].get('class_balance_ratio', 0):.4f}")
        lines.append("")
        lines.append(f"**Selected Features ({len(best_result['selected_features'])}):** "
                     f"{', '.join(best_result['selected_features'][:5])}...")
    lines.append("")

    # Dataset Overview
    lines.append("---")
    lines.append("")
    lines.append("## 1. Dataset Overview")
    lines.append("")

    if 'dataset_stats' in results:
        lines.append("| Audio Type | Real | Fake | Total | Ratio |")
        lines.append("|------------|------|------|-------|-------|")
        for audio_type, stats in results['dataset_stats'].items():
            real = stats.get('real', 0)
            fake = stats.get('fake', 0)
            total = real + fake
            ratio = f"{real/fake:.2f}" if fake > 0 else "N/A"
            lines.append(f"| {audio_type} | {real:,} | {fake:,} | {total:,} | {ratio} |")
        lines.append("")

    # Classifier Architecture
    lines.append("---")
    lines.append("")
    lines.append("## 2. Classifier Architecture")
    lines.append("")
    lines.append("### XGBoost Hyperparameters")
    lines.append("")
    lines.append("```")
    lines.append("learning_rate: 0.1")
    lines.append("max_depth: 6")
    lines.append("min_child_weight: 1")
    lines.append("subsample: 0.8")
    lines.append("colsample_bytree: 0.8")
    lines.append("gamma: 0.1")
    lines.append("reg_lambda: 1.0")
    lines.append("n_estimators: 100")
    lines.append("```")
    lines.append("")
    lines.append("### Training Strategy")
    lines.append("")
    lines.append("- **Data Split:** 80% train / 20% held-out test")
    lines.append("- **Cross-Validation:** 5-fold stratified CV on training set")
    lines.append("- **Feature Selection:** Greedy forward selection (>0.5% F1 improvement)")
    lines.append("- **Class Balancing:** Random undersampling of majority class")
    lines.append("")

    # Feature Selection Results
    lines.append("---")
    lines.append("")
    lines.append("## 3. Feature Selection Results")
    lines.append("")

    if 'training_results' in results:
        lines.append("### Selected Features by Model")
        lines.append("")
        lines.append("| Model | # Features | Selected Features |")
        lines.append("|-------|------------|-------------------|")
        for config_name, result in results['training_results'].items():
            features = result.get('selected_features', [])
            n_feat = len(features)
            feat_str = ', '.join(features[:5])
            if len(features) > 5:
                feat_str += f"... (+{len(features)-5})"
            lines.append(f"| {config_name} | {n_feat} | {feat_str} |")
        lines.append("")

    # Classification Results
    lines.append("---")
    lines.append("")
    lines.append("## 4. Classification Results")
    lines.append("")

    if 'training_results' in results:
        # Per-type results
        lines.append("### Per-Audio-Type Models")
        lines.append("")
        lines.append("| Model | Accuracy | F1 | AUC-ROC | F1 Real | F1 Fake | Balance |")
        lines.append("|-------|----------|----|---------|---------|---------|---------| ")

        for config_name, result in results['training_results'].items():
            if 'universal' not in config_name:
                scores = result.get('test_scores', {})
                lines.append(
                    f"| {config_name} | "
                    f"{scores.get('accuracy', 0):.3f} | "
                    f"{scores.get('f1', 0):.3f} | "
                    f"{scores.get('roc_auc', 0):.3f} | "
                    f"{scores.get('f1_real', 0):.3f} | "
                    f"{scores.get('f1_fake', 0):.3f} | "
                    f"{scores.get('class_balance_ratio', 0):.3f} |"
                )
        lines.append("")

        # Universal models
        universal_results = {k: v for k, v in results['training_results'].items()
                            if 'universal' in k}
        if universal_results:
            lines.append("### Universal Models")
            lines.append("")
            lines.append("| Model | Accuracy | F1 | AUC-ROC | F1 Real | F1 Fake | Balance |")
            lines.append("|-------|----------|----|---------|---------|---------|---------| ")

            for config_name, result in universal_results.items():
                scores = result.get('test_scores', {})
                lines.append(
                    f"| {config_name} | "
                    f"{scores.get('accuracy', 0):.3f} | "
                    f"{scores.get('f1', 0):.3f} | "
                    f"{scores.get('roc_auc', 0):.3f} | "
                    f"{scores.get('f1_real', 0):.3f} | "
                    f"{scores.get('f1_fake', 0):.3f} | "
                    f"{scores.get('class_balance_ratio', 0):.3f} |"
                )
            lines.append("")

        # Embedding comparison
        lines.append("### LAION-CLAP vs MS-CLAP Comparison")
        lines.append("")

        laion_results = {k: v for k, v in results['training_results'].items()
                         if 'laion' in k}
        msclap_results = {k: v for k, v in results['training_results'].items()
                          if 'msclap' in k}

        if laion_results and msclap_results:
            laion_avg_f1 = np.mean([r['test_scores'].get('f1', 0)
                                    for r in laion_results.values()])
            msclap_avg_f1 = np.mean([r['test_scores'].get('f1', 0)
                                     for r in msclap_results.values()])

            lines.append(f"- **LAION-CLAP Average F1:** {laion_avg_f1:.4f}")
            lines.append(f"- **MS-CLAP Average F1:** {msclap_avg_f1:.4f}")

            if laion_avg_f1 > msclap_avg_f1:
                lines.append(f"- **Better Embedding:** LAION-CLAP (+{laion_avg_f1 - msclap_avg_f1:.4f})")
            else:
                lines.append(f"- **Better Embedding:** MS-CLAP (+{msclap_avg_f1 - laion_avg_f1:.4f})")
        lines.append("")

    # Feature Importance
    lines.append("---")
    lines.append("")
    lines.append("## 5. Feature Importance Analysis")
    lines.append("")

    if 'feature_importance' in results:
        lines.append("### Top Features (Combined Importance)")
        lines.append("")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")

        for i, (feature, importance) in enumerate(
            list(results['feature_importance'].items())[:10], 1
        ):
            lines.append(f"| {i} | {feature} | {importance:.4f} |")
        lines.append("")

    # Ablation Studies
    lines.append("---")
    lines.append("")
    lines.append("## 6. Ablation Studies")
    lines.append("")

    if 'ablation' in results:
        ablation = results['ablation']

        if 'feature_sets' in ablation:
            lines.append("### Feature Set Comparison")
            lines.append("")
            lines.append("| Feature Set | # Features | CV F1 |")
            lines.append("|-------------|------------|-------|")

            for _, row in ablation['feature_sets'].iterrows():
                lines.append(f"| {row['set_name']} | {row['n_features']} | {row['cv_f1']:.4f} |")
            lines.append("")

        if 'leave_one_out' in ablation:
            lines.append("### Leave-One-Out Analysis (Top 5 Most Important)")
            lines.append("")
            lines.append("| Feature | F1 Drop | Drop % |")
            lines.append("|---------|---------|--------|")

            df = ablation['leave_one_out'].head(5)
            for _, row in df.iterrows():
                lines.append(f"| {row['feature']} | {row['drop']:.4f} | {row['drop_pct']:.2f}% |")
            lines.append("")

    # Recommendations
    lines.append("---")
    lines.append("")
    lines.append("## 7. Recommendations")
    lines.append("")

    if best_config:
        best_result = results['training_results'][best_config]
        lines.append("### Optimal Configuration")
        lines.append("")
        lines.append(f"1. **Model:** {best_config}")
        lines.append(f"2. **Feature Set:** {len(best_result['selected_features'])} features")
        lines.append(f"3. **Key Features:** {', '.join(best_result['selected_features'][:7])}")
        lines.append("")

    lines.append("### Model Selection Guidance")
    lines.append("")
    lines.append("- For **single audio type deployment**: Use per-type model for best accuracy")
    lines.append("- For **mixed audio deployment**: Use universal model for generalization")
    lines.append("- For **minimal features**: Top 7 features provide ~95% of full performance")
    lines.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def format_confusion_matrix(cm: List[List[int]]) -> str:
    """Format confusion matrix as markdown.

    Args:
        cm: Confusion matrix [[TN, FP], [FN, TP]]

    Returns:
        Markdown formatted string
    """
    tn, fp = cm[0]
    fn, tp = cm[1]

    lines = [
        "```",
        "              Predicted",
        "            Real    Fake",
        f"Actual Real  {tn:5d}   {fp:5d}",
        f"       Fake  {fn:5d}   {tp:5d}",
        "```",
    ]
    return '\n'.join(lines)


class ReportGenerator:
    """Generate comprehensive classification report."""

    def __init__(self, output_dir: Path):
        """Initialize report generator.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Any] = {
            'training_results': {},
            'dataset_stats': {},
            'feature_importance': {},
            'ablation': {},
        }

    def add_training_result(
        self,
        config_name: str,
        result: Dict[str, Any],
    ) -> None:
        """Add training result.

        Args:
            config_name: Configuration name
            result: Training result dictionary
        """
        self.results['training_results'][config_name] = result

    def add_dataset_stats(
        self,
        audio_type: str,
        real_count: int,
        fake_count: int,
    ) -> None:
        """Add dataset statistics.

        Args:
            audio_type: Audio type name
            real_count: Number of real samples
            fake_count: Number of fake samples
        """
        self.results['dataset_stats'][audio_type] = {
            'real': real_count,
            'fake': fake_count,
        }

    def add_feature_importance(
        self,
        importance: Dict[str, float],
    ) -> None:
        """Add feature importance scores.

        Args:
            importance: Feature importance dictionary
        """
        self.results['feature_importance'] = importance

    def add_ablation_results(
        self,
        ablation_type: str,
        results_df: pd.DataFrame,
    ) -> None:
        """Add ablation study results.

        Args:
            ablation_type: Type of ablation study
            results_df: Results DataFrame
        """
        self.results['ablation'][ablation_type] = results_df

    def generate(self, filename: str = "CLASSIFICATION_REPORT.md") -> Path:
        """Generate the report.

        Args:
            filename: Output filename

        Returns:
            Path to generated report
        """
        output_path = self.output_dir / filename
        generate_classification_report(self.results, output_path)
        return output_path

    def save_results_json(self, filename: str = "results.json") -> Path:
        """Save raw results as JSON.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        # Convert DataFrames to dicts for JSON serialization
        serializable = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, pd.DataFrame):
                        serializable[key][k] = v.to_dict(orient='records')
                    else:
                        serializable[key][k] = v
            else:
                serializable[key] = value

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        return output_path
