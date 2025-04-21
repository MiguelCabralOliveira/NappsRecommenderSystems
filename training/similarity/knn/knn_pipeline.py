# training/similarity/knn_pipeline.py

import pandas as pd
import numpy as np
import os
import time
import traceback
from typing import Optional, Dict, Tuple, Any
import json

# Import core classes and evaluation function from the same module
from .weighted_knn_core import _WeightedKNN, WeightOptimizer
from .evaluation import run_evaluation
from ..utils import ensure_dir, save_text_report

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
    products_df_for_titles: Optional[pd.DataFrame] = None, # For adding titles to export
    skip_evaluation: bool = False,
    save_model: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[_WeightedKNN]]:
    """
    Main pipeline function for training/optimizing the Weighted KNN model.

    Args:
        features_df: DataFrame containing ONLY the numeric feature columns.
        product_ids: Series containing the product IDs, aligned with features_df rows.
        output_dir: Directory to save all training outputs (model, results, plots).
        n_neighbors: Number of neighbors (recommendations) to generate per product.
        base_weights: Optional dictionary of base weights for feature groups. If None or
                      optimize_weights is True, defaults/optimized weights are used.
        optimize_weights: If True, run weight optimization.
        optimization_method: Method for optimization ('grid', 'random', 'evolutionary').
        optimization_params: Dictionary of parameters for the chosen optimization method.
        filter_variants: If True, apply filtering based on similar_products_df.
        similar_products_df: DataFrame for variant filtering (required if filter_variants=True).
        products_df_for_titles: DataFrame with original product info for adding titles to exports.
        skip_evaluation: If True, skip the final evaluation step.
        save_model: If True, save the final fitted model components.

    Returns:
        A tuple containing:
        - DataFrame: The final recommendations DataFrame (filtered or unfiltered).
        - _WeightedKNN: The final fitted _WeightedKNN model instance (or None if failed).
    """
    print("\n" + "="*30 + " Running Weighted KNN Training Pipeline " + "="*30)
    pipeline_start_time = time.time()
    ensure_dir(output_dir) # Ensure output directory exists

    if features_df is None or features_df.empty or product_ids is None or product_ids.empty:
        print("Error: Input features_df or product_ids is empty or None.")
        return pd.DataFrame(), None

    if len(features_df) != len(product_ids):
         print(f"Error: Mismatch in length between features ({len(features_df)}) and product IDs ({len(product_ids)}).")
         return pd.DataFrame(), None

    if filter_variants and (similar_products_df is None or similar_products_df.empty):
        print("Warning: Variant filtering requested but similar_products_df is missing or empty. Disabling filter.")
        filter_variants = False

    final_model: Optional[_WeightedKNN] = None
    optimizer: Optional[WeightOptimizer] = None
    best_weights_used = base_weights # Start assuming base weights unless optimized

    # --- 1. Weight Optimization (Optional) ---
    if optimize_weights:
        print(f"\n=== Starting Feature Weight Optimization (Method: {optimization_method}) ===")
        opt_start_time = time.time()
        opt_dir = os.path.join(output_dir, "weight_optimization")
        try:
            optimizer = WeightOptimizer(
                features_df=features_df,
                product_ids=product_ids,
                n_neighbors=n_neighbors,
                output_dir=opt_dir,
                random_state=42 # Or pass as arg if needed
            )

            if optimization_params is None: optimization_params = {}
            print(f"Running {optimization_method} optimization...")

            if optimization_method == 'grid': optimizer.grid_search(**optimization_params, save_results=True)
            elif optimization_method == 'random': optimizer.random_search(**optimization_params, save_results=True)
            elif optimization_method == 'evolutionary': optimizer.evolutionary_search(**optimization_params, save_results=True)
            else:
                print(f"Warning: Unknown optimization method '{optimization_method}'. Defaulting to evolutionary.")
                optimizer.evolutionary_search(**optimization_params, save_results=True) # Default to evolutionary?

            # Save optimization plots
            optimizer.save_optimization_results(filename="optimization_results_details.csv") # Save detailed results
            # optimizer.visualize_optimization_results(filename=os.path.join(opt_dir, "optimization_performance.png")) # Visualize if needed

            print("\nBuilding final KNN model using optimized weights...")
            final_model = optimizer.build_optimized_model(save_model_files=save_model) # Saves model inside if save_model is True
            if final_model:
                best_weights_used = optimizer.best_weights # Store the weights used by the final model
                print(f"Best weights determined (Score: {optimizer.best_score:.4f})")
            else:
                print("Error: Failed to build model after optimization.")

            opt_end_time = time.time()
            print(f"=== Weight Optimization Finished (Duration: {opt_end_time - opt_start_time:.2f}s) ===")

        except Exception as e:
            print(f"\n--- ERROR during Weight Optimization: {type(e).__name__} - {str(e)} ---")
            traceback.print_exc()
            print("Optimization failed. Attempting to proceed with base/default weights...")
            final_model = None # Ensure model is reset if optimization failed critically
            best_weights_used = base_weights # Revert to base/default for fallback

    # --- 2. Fit Model with Base/Default or Optimized Weights ---
    if final_model is None: # If not optimized or optimization failed
        fit_mode = "Base/Default" if not optimize_weights else "Fallback (Base/Default)"
        print(f"\n=== Building KNN Model with {fit_mode} Weights ===")
        model_build_start_time = time.time()
        try:
            final_model = _WeightedKNN(n_neighbors=n_neighbors, random_state=42)
            # Set weights BEFORE fitting
            final_model._identify_feature_groups(features_df.columns.tolist()) # Identify groups first
            final_model.set_feature_weights(base_weights=best_weights_used, normalize_by_count=True) # Use defaults if best_weights_used is None
            # Now fit
            final_model.fit(features_df, product_ids) # Fit requires features and IDs

            model_build_end_time = time.time()
            if final_model.recommendation_matrix is not None:
                 print(f"Model fitting with {fit_mode} weights completed in {model_build_end_time - model_build_start_time:.2f}s.")
                 if save_model:
                      model_save_dir = os.path.join(output_dir, "model_default_weights" if not best_weights_used else "model_optimized_weights") # Adjust save dir name maybe
                      print(f"\nSaving the model fitted with {fit_mode} weights to: {model_save_dir}")
                      final_model.save_model(model_save_dir)
                 best_weights_used = final_model.feature_weights # Store the actual weights used (might be defaults)
            else:
                 print(f"Error: Failed to fit model with {fit_mode} weights.")
                 final_model = None # Ensure model is None if fit failed

        except Exception as e:
            print(f"\n--- ERROR during Model Fitting with {fit_mode} Weights: {type(e).__name__} - {str(e)} ---")
            traceback.print_exc()
            final_model = None

    # --- 3. Generate Visualizations, Export Recs, Evaluate ---
    all_recommendations_df: Optional[pd.DataFrame] = None
    final_recommendations_path: Optional[str] = None
    unfiltered_recs_path = os.path.join(output_dir, "knn_recommendations_unfiltered.csv")
    final_recs_path_filtered = os.path.join(output_dir, "knn_recommendations_final.csv") # Final, potentially filtered

    if final_model is None or final_model.recommendation_matrix is None:
         print("\nError: No valid KNN model was fitted. Cannot generate recommendations or evaluate.")
         # Ensure we return None model
    else:
        print("\n=== Generating Visualizations and Exporting Results ===")

        # --- Visualizations ---
        vis_success = {'importance': False, 'tsne': False, 'distribution': False}
        vis_dir = os.path.join(output_dir, "visualizations")
        ensure_dir(vis_dir)
        try: final_model.visualize_feature_importance(output_file=os.path.join(vis_dir, "feature_importance.png")); vis_success['importance'] = True
        except Exception as e: print(f"Error generating feature importance plot: {e}")
        try: final_model.visualize_tsne(output_file=os.path.join(vis_dir, "tsne_visualization.png")); vis_success['tsne'] = True
        except Exception as e: print(f"Error generating t-SNE plot: {e}")
        try: final_model.visualize_similarity_distribution(output_file=os.path.join(vis_dir, "similarity_distribution.png")); vis_success['distribution'] = True
        except Exception as e: print(f"Error generating similarity distribution plot: {e}")

        # --- Export Unfiltered Recommendations ---
        print(f"\nExporting UNFILTERED KNN recommendations to: {unfiltered_recs_path}")
        try:
             unfiltered_recs_df = final_model.export_recommendations(
                 output_file=unfiltered_recs_path,
                 n_recommendations=n_neighbors,
                 apply_variant_filter=False, # Explicitly false
                 products_df_for_titles=products_df_for_titles
             )
             if unfiltered_recs_df.empty: print("Warning: Unfiltered export resulted in empty DataFrame.")
        except Exception as e:
             print(f"\n--- Error Exporting Unfiltered Recommendations: {e} ---")
             traceback.print_exc()
             unfiltered_recs_df = pd.DataFrame()


        # --- Export Final Recommendations (Potentially Filtered) ---
        print(f"\nExporting FINAL recommendations to: {final_recs_path_filtered}")
        print(f"(Variant filtering {'ENABLED' if filter_variants else 'DISABLED'})")
        try:
             all_recommendations_df = final_model.export_recommendations(
                 output_file=final_recs_path_filtered,
                 n_recommendations=n_neighbors,
                 apply_variant_filter=filter_variants, # Use the flag
                 similar_products_df=similar_products_df, # Pass the df
                 products_df_for_titles=products_df_for_titles
             )
             final_recommendations_path = final_recs_path_filtered # This is the main output file now
             if all_recommendations_df.empty: print("Warning: Final export resulted in empty DataFrame.")
        except Exception as e:
            print(f"\n--- Error Exporting Final Recommendations: {e} ---")
            traceback.print_exc()
            all_recommendations_df = pd.DataFrame() # Ensure it's an empty DF on error
            final_recommendations_path = None # Indicate failure


        # --- Evaluation ---
        if not skip_evaluation:
            print("\n" + "="*30 + " Starting Evaluation " + "="*30)
            eval_output_dir = os.path.join(output_dir, "evaluation_metrics")
            ensure_dir(eval_output_dir)

            # We need a catalog file path - assuming it's related to products_df_for_titles if provided
            catalog_file_path = None
            if products_df_for_titles is not None:
                 # Attempt to find where the original raw data might be saved or use a temp save
                 # This part is tricky without knowing the upstream saving conventions
                 # Let's assume for now `products_df_for_titles` can be saved temporarily if needed
                 # Or ideally, the path to the *raw* product file is passed in.
                 # For now, let's skip catalog-dependent metrics if path isn't clear
                 # TODO: Need a clear way to get the catalog path (e.g., pass as argument)
                 print("Warning: Catalog file path needed for Coverage evaluation is not explicitly provided. Skipping Coverage.")

            # Evaluate UNFILTERED recommendations
            if os.path.exists(unfiltered_recs_path):
                 print(f"\n--- Evaluating UNFILTERED Recommendations ({os.path.basename(unfiltered_recs_path)}) ---")
                 try:
                      run_evaluation(
                          recs_path=unfiltered_recs_path,
                          catalog_path=None, # Pass None if path unknown
                          k=n_neighbors,
                          output_dir=eval_output_dir,
                          pop_path=None, # Popularity not used here
                          knn_sims_path=unfiltered_recs_path, # Use unfiltered recs for ILS base
                          results_filename=f"evaluation_k{n_neighbors}_knn_unfiltered.json"
                      )
                 except Exception as e:
                      print(f"Error during UNFILTERED Evaluation: {e}")
                      traceback.print_exc()
            else: print("Skipping UNFILTERED Evaluation: File not found.")


            # Evaluate FINAL recommendations
            if final_recommendations_path and os.path.exists(final_recommendations_path):
                # Only evaluate if different from unfiltered
                if final_recommendations_path != unfiltered_recs_path:
                    print(f"\n--- Evaluating FINAL Recommendations ({os.path.basename(final_recommendations_path)}) ---")
                    try:
                        run_evaluation(
                            recs_path=final_recommendations_path,
                            catalog_path=None, # Pass None if path unknown
                            k=n_neighbors,
                            output_dir=eval_output_dir,
                            pop_path=None, # Popularity not used here
                            knn_sims_path=unfiltered_recs_path, # STILL use unfiltered sims for ILS base
                            results_filename=f"evaluation_k{n_neighbors}_knn_final.json"
                        )
                    except Exception as e:
                         print(f"Error during FINAL Evaluation: {e}")
                         traceback.print_exc()
                else:
                    print("\nSkipping FINAL evaluation as it's the same as the unfiltered recommendations.")
            else:
                print("\nSkipping FINAL Evaluation: Final recommendations file not found or not generated.")
        else:
            print("\nSkipping Evaluation step.")


        # --- Summary Report ---
        print("\nGenerating Summary Report...")
        summary_path = os.path.join(output_dir, "knn_summary_report.txt")
        try:
            avg_similarity = final_model.calculate_average_similarity()
            num_products = len(final_model.product_ids) if final_model.product_ids is not None else 'N/A'
            num_features = len(final_model.feature_names_in_order) if final_model.feature_names_in_order is not None else 'N/A'

            with open(summary_path, "w", encoding='utf-8') as f:
                f.write("=== Weighted KNN Training Summary ===\n\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Pipeline Duration: {time.time() - pipeline_start_time:.2f} seconds\n\n")
                f.write(f"Input Products: {num_products}\n")
                f.write(f"Input Features: {num_features}\n\n")
                f.write(f"KNN Parameters:\n")
                f.write(f"- n_neighbors (k): {n_neighbors}\n")
                f.write(f"- algorithm: {final_model.algorithm}\n")
                f.write(f"- metric: {final_model.metric}\n\n")
                f.write(f"Weight Optimization: {'Enabled (' + optimization_method + ')' if optimize_weights else 'Disabled'}\n")
                if best_weights_used:
                     f.write("Weights Used (Base or Optimized - see below for effective):\n")
                     # Format weights nicely
                     weights_str = json.dumps(best_weights_used, indent=4)
                     f.write(weights_str + "\n\n")
                     # Optionally add effective weights too if desired (more verbose)
                     # f.write("Effective Weights Applied (after normalization):\n")
                     # effective_weights_str = json.dumps(final_model.feature_weights, indent=4)
                     # f.write(effective_weights_str + "\n\n")

                f.write(f"Variant Filtering: {'Enabled' if filter_variants else 'Disabled'}\n")
                if filter_variants and similar_products_df is not None:
                     f.write(f"- Using {len(similar_products_df)} pairs from similar products file.\n\n")

                f.write("Model Performance:\n")
                f.write(f"- Average Recommendation Similarity: {avg_similarity:.4f}\n\n")

                f.write("Outputs Generated:\n")
                f.write(f"- Unfiltered Recommendations: {'Yes' if os.path.exists(unfiltered_recs_path) else 'No'} ({unfiltered_recs_path})\n")
                final_recs_status = 'Yes' if final_recommendations_path and os.path.exists(final_recommendations_path) else 'No'
                f.write(f"- Final Recommendations: {final_recs_status} ({final_recs_path_filtered})\n")
                model_save_status = 'Yes' if save_model else 'No (Skipped)' # Check if model was actually saved? More complex state needed.
                f.write(f"- Saved Model Components: {model_save_status}\n")
                eval_status = 'Skipped' if skip_evaluation else ('Yes' if os.path.exists(eval_output_dir) else 'No (Failed?)')
                f.write(f"- Evaluation Metrics: {eval_status}\n")
                vis_list = [name for name, success in vis_success.items() if success]
                f.write(f"- Visualizations Generated: {', '.join(vis_list) if vis_list else 'None'}\n")

            save_text_report(f.getvalue(), summary_path) # Use getvalue if writing to string buffer first
            # Or just rely on the file being closed properly by 'with open'
            print(f"Summary report saved to {summary_path}")

        except Exception as e:
             print(f"\nError generating summary report: {e}")


    pipeline_end_time = time.time()
    print("\n" + "="*30 + f" KNN Pipeline Finished (Total Time: {pipeline_end_time - pipeline_start_time:.2f}s) " + "="*30)

    # Return the final recommendations DataFrame and the fitted model object
    return all_recommendations_df, final_model