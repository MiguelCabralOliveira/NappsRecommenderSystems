# training/similarity/knn/knn_pipeline.py

import pandas as pd
import numpy as np
import os
import time
import traceback
import json
from typing import Optional, Dict, Tuple, Any

# Imports
try:
    from .weighted_knn_core import _WeightedKNN, WeightOptimizer
    from .evaluation import run_evaluation
    from ..utils import ensure_dir, save_text_report
except ImportError:
     print("Warning: KNN Pipeline using fallback imports.")
     from weighted_knn_core import _WeightedKNN, WeightOptimizer
     from evaluation import run_evaluation
     from utils import ensure_dir, save_text_report

def run_knn_training(
    features_df: pd.DataFrame,
    product_ids: pd.Series,
    output_dir: str,
    n_neighbors: int,
    base_weights: Optional[Dict[str, float]] = None,
    optimize_weights: bool = False,
    optimization_method: str = 'evolutionary',
    optimization_params: Optional[Dict[str, Any]] = None,
    filter_variants: bool = False,
    similar_products_df: Optional[pd.DataFrame] = None,
    products_df_for_titles: Optional[pd.DataFrame] = None,
    catalog_csv_path: Optional[str] = None, # Argumento para path do catálogo
    skip_evaluation: bool = False,
    save_model: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[_WeightedKNN]]:
    """
    Main pipeline function for training/optimizing the Weighted KNN model.
    Includes optional evaluation step that requires catalog_csv_path for coverage.
    """
    print("\n" + "="*30 + " Running Weighted KNN Training Pipeline " + "="*30)
    pipeline_start_time = time.time()
    ensure_dir(output_dir)

    # Basic validation
    if features_df is None or features_df.empty or product_ids is None or product_ids.empty:
        print("Error: Input features_df or product_ids is empty or None.")
        return pd.DataFrame(), None
    if len(features_df) != len(product_ids):
         print(f"Error: Mismatch in length between features ({len(features_df)}) and product IDs ({len(product_ids)}).")
         return pd.DataFrame(), None
    if filter_variants and (similar_products_df is None or similar_products_df.empty):
        print("Warning: Variant filtering requested but similar_products_df is missing or empty. Disabling filter.")
        filter_variants = False

    # Initialize
    final_model: Optional[_WeightedKNN] = None
    optimizer: Optional[WeightOptimizer] = None
    best_weights_used = base_weights

    # 1. Weight Optimization (Optional)
    if optimize_weights:
        print(f"\n=== Starting Feature Weight Optimization (Method: {optimization_method}) ===")
        opt_start_time = time.time()
        opt_dir = os.path.join(output_dir, "weight_optimization")
        try:
            optimizer = WeightOptimizer(features_df=features_df.copy(), product_ids=product_ids.copy(), n_neighbors=n_neighbors, output_dir=opt_dir, random_state=42)
            if optimization_params is None: optimization_params = {}
            print(f"Running {optimization_method} optimization...")
            if optimization_method == 'grid': optimizer.grid_search(**optimization_params, save_results=True)
            elif optimization_method == 'random': optimizer.random_search(**optimization_params, save_results=True)
            elif optimization_method == 'evolutionary': optimizer.evolutionary_search(**optimization_params, save_results=True)
            else: optimizer.evolutionary_search(**optimization_params, save_results=True); print("Defaulted to evolutionary.")
            optimizer.save_optimization_results(filename="optimization_results_details.csv")
            print("\nBuilding final KNN model using optimized weights...")
            final_model = optimizer.build_optimized_model(save_model_files=save_model)
            if final_model: best_weights_used = optimizer.best_weights
            else: print("Error: Failed to build model after optimization.")
            opt_end_time = time.time()
            print(f"=== Weight Optimization Finished (Duration: {opt_end_time - opt_start_time:.2f}s) ===")
        except Exception as e:
            print(f"\n--- ERROR during Weight Optimization: {type(e).__name__} - {str(e)} ---"); traceback.print_exc()
            final_model = None; best_weights_used = base_weights

    # 2. Fit Model with Base/Default or Optimized Weights
    if final_model is None:
        fit_mode = "Base/Default" if not optimize_weights else "Fallback (Base/Default)"
        print(f"\n=== Building KNN Model with {fit_mode} Weights ===")
        model_build_start_time = time.time()
        try:
            final_model = _WeightedKNN(n_neighbors=n_neighbors, random_state=42)
            final_model._identify_feature_groups(features_df.columns.tolist())
            final_model.set_feature_weights(base_weights=best_weights_used, normalize_by_count=True)
            final_model.fit(features_df.copy(), product_ids.copy())
            model_build_end_time = time.time()
            if final_model.recommendation_matrix is not None:
                 print(f"Model fitting with {fit_mode} weights completed in {model_build_end_time - model_build_start_time:.2f}s.")
                 if save_model:
                      model_save_dir_name = "model_optimized_weights" if optimize_weights and final_model.feature_weights != final_model._default_base_weights else "model_default_weights"
                      model_save_dir = os.path.join(output_dir, model_save_dir_name)
                      print(f"\nSaving the model fitted with {fit_mode} weights to: {model_save_dir}")
                      final_model.save_model(model_save_dir)
                 best_weights_used = final_model.feature_weights # Store actual weights
            else: print(f"Error: Failed to fit model with {fit_mode} weights."); final_model = None
        except Exception as e: print(f"ERROR during Model Fitting: {e}"); traceback.print_exc(); final_model = None

    # 3. Generate Visualizations, Export Recs, Evaluate
    all_recommendations_df: pd.DataFrame = pd.DataFrame()
    final_recommendations_path: Optional[str] = None
    unfiltered_recs_path = os.path.join(output_dir, "knn_recommendations_unfiltered.csv")
    final_recs_path_filtered = os.path.join(output_dir, "knn_recommendations_final.csv")
    vis_success = {'importance': False, 'tsne': False, 'distribution': False}

    if final_model is None or final_model.recommendation_matrix is None:
         print("\nError: No valid KNN model was fitted. Cannot generate recommendations or evaluate.")
    else:
        print("\n=== Generating Visualizations and Exporting Results ===")
        vis_dir = os.path.join(output_dir, "visualizations"); ensure_dir(vis_dir)
        try: final_model.visualize_feature_importance(output_file=os.path.join(vis_dir, "feature_importance.png")); vis_success['importance'] = True
        except Exception as e: print(f"Error generating feature importance plot: {e}")
        try: final_model.visualize_tsne(output_file=os.path.join(vis_dir, "tsne_visualization.png")); vis_success['tsne'] = True
        except Exception as e: print(f"Error generating t-SNE plot: {e}")
        try: final_model.visualize_similarity_distribution(output_file=os.path.join(vis_dir, "similarity_distribution.png")); vis_success['distribution'] = True
        except Exception as e: print(f"Error generating similarity distribution plot: {e}")

        # Export Unfiltered
        print(f"\nExporting UNFILTERED KNN recommendations to: {unfiltered_recs_path}")
        try:
             unfiltered_recs_df = final_model.export_recommendations(unfiltered_recs_path, n_neighbors, False, None, products_df_for_titles)
             if unfiltered_recs_df.empty: print("Warning: Unfiltered export resulted in empty DataFrame.")
        except Exception as e: print(f"Error Exporting Unfiltered Recs: {e}"); traceback.print_exc(); unfiltered_recs_df = pd.DataFrame()

        # Export Final (Potentially Filtered)
        print(f"\nExporting FINAL recommendations to: {final_recs_path_filtered}")
        print(f"(Variant filtering {'ENABLED' if filter_variants else 'DISABLED'})")
        try:
             all_recommendations_df = final_model.export_recommendations(final_recs_path_filtered, n_neighbors, filter_variants, similar_products_df, products_df_for_titles)
             final_recommendations_path = final_recs_path_filtered
             if all_recommendations_df.empty: print("Warning: Final export resulted in empty DataFrame.")
        except Exception as e: print(f"Error Exporting Final Recs: {e}"); traceback.print_exc(); all_recommendations_df = pd.DataFrame(); final_recommendations_path = None

        # Evaluation
        if not skip_evaluation:
            print("\n" + "="*30 + " Starting Evaluation " + "="*30)
            eval_output_dir = os.path.join(output_dir, "evaluation_metrics"); ensure_dir(eval_output_dir)
            print(f"Evaluation Catalog Path: {catalog_csv_path if catalog_csv_path else 'Not Provided (Coverage Skipped)'}")

            # Evaluate UNFILTERED
            if os.path.exists(unfiltered_recs_path):
                 print(f"\n--- Evaluating UNFILTERED Recs ({os.path.basename(unfiltered_recs_path)}) ---")
                 try: run_evaluation(unfiltered_recs_path, n_neighbors, eval_output_dir, catalog_csv_path, None, unfiltered_recs_path, f"evaluation_k{n_neighbors}_knn_unfiltered.json")
                 except Exception as e: print(f"Error during UNFILTERED Eval: {e}"); traceback.print_exc()
            else: print("Skipping UNFILTERED Evaluation: File not found.")

            # Evaluate FINAL
            if final_recommendations_path and os.path.exists(final_recommendations_path):
                if final_recommendations_path != unfiltered_recs_path:
                    print(f"\n--- Evaluating FINAL Recs ({os.path.basename(final_recommendations_path)}) ---")
                    try: run_evaluation(final_recommendations_path, n_neighbors, eval_output_dir, catalog_csv_path, None, unfiltered_recs_path, f"evaluation_k{n_neighbors}_knn_final.json")
                    except Exception as e: print(f"Error during FINAL Eval: {e}"); traceback.print_exc()
                else: print("\nSkipping FINAL evaluation (same as unfiltered).")
            else: print("\nSkipping FINAL Evaluation: File not found or not generated.")
        else: print("\nSkipping Evaluation step.")

        # Summary Report
        print("\nGenerating Summary Report...")
        summary_path = os.path.join(output_dir, "knn_summary_report.txt")
        try:
            if final_model:
                 avg_similarity = final_model.calculate_average_similarity()
                 num_products = len(final_model.product_ids) if final_model.product_ids is not None else 'N/A'
                 num_features = len(final_model.feature_names_in_order) if final_model.feature_names_in_order is not None else 'N/A'
                 model_algorithm = final_model.algorithm; model_metric = final_model.metric
            else: avg_similarity = 0.0; num_products = 'N/A'; num_features = 'N/A'; model_algorithm = 'N/A'; model_metric = 'N/A'

            report_content = ["=== Weighted KNN Training Summary ===", ""]
            report_content.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"Total Pipeline Duration: {time.time() - pipeline_start_time:.2f} seconds\n")
            report_content.append(f"Input Products: {num_products}")
            report_content.append(f"Input Features: {num_features}\n")
            report_content.append(f"KNN Parameters:")
            report_content.append(f"- n_neighbors (k): {n_neighbors}")
            report_content.append(f"- algorithm: {model_algorithm}")
            report_content.append(f"- metric: {model_metric}\n")
            report_content.append(f"Weight Optimization: {'Enabled (' + optimization_method + ')' if optimize_weights else 'Disabled'}")
            if best_weights_used:
                 report_content.append("Weights Used (Base or Optimized):")
                 weights_str = json.dumps(best_weights_used, indent=2); report_content.append(weights_str + "\n")
            report_content.append(f"Variant Filtering: {'Enabled' if filter_variants else 'Disabled'}")
            if filter_variants and similar_products_df is not None: report_content.append(f"- Using {len(similar_products_df)} pairs from similar products file.\n")
            report_content.append("Model Performance:")
            report_content.append(f"- Average Recommendation Similarity: {avg_similarity:.4f}\n")
            report_content.append("Outputs Generated:")
            report_content.append(f"- Unfiltered Recommendations: {'Yes' if os.path.exists(unfiltered_recs_path) else 'No'} ({os.path.basename(unfiltered_recs_path)})")
            final_recs_status = 'Yes' if final_recommendations_path and os.path.exists(final_recommendations_path) else 'No'
            report_content.append(f"- Final Recommendations: {final_recs_status} ({os.path.basename(final_recs_path_filtered)})")
            model_save_status = 'Yes' if save_model else 'No (Skipped)'
            report_content.append(f"- Saved Model Components: {model_save_status}")
            eval_dir = os.path.join(output_dir, "evaluation_metrics")
            eval_files_exist = os.path.exists(eval_dir) and len(os.listdir(eval_dir)) > 0
            eval_status = 'Skipped' if skip_evaluation else ('Yes' if eval_files_exist else 'No (Or Failed)')
            report_content.append(f"- Evaluation Metrics: {eval_status}")
            vis_list = [name for name, success in vis_success.items() if success]
            report_content.append(f"- Visualizations Generated: {', '.join(vis_list) if vis_list else 'None'}")
            save_text_report("\n".join(report_content), summary_path)
        except Exception as e: print(f"\nError generating summary report: {e}"); traceback.print_exc()

    pipeline_end_time = time.time()
    print("\n" + "="*30 + f" KNN Pipeline Finished (Total Time: {pipeline_end_time - pipeline_start_time:.2f}s) " + "="*30)

    return all_recommendations_df, final_model