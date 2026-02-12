import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


# Local tasks - token/lexical level features should suffice
LOCAL_TASKS = {
    # Geographic - place names are lexical
    "114_nyc_borough_Manhattan",
    "115_nyc_borough_Brooklyn",
    "116_nyc_borough_Bronx",
    "117_us_state_FL",
    "118_us_state_CA",
    "119_us_state_TX",
    "120_us_timezone_Chicago",
    "121_us_timezone_New_York",
    "122_us_timezone_Los_Angeles",
    "123_world_country_United_Kingdom",
    "124_world_country_United_States",
    "125_world_country_Italy",
    # Code detection - syntax tokens
    "158_code_C",
    "159_code_Python",
    "160_code_HTML",
    # News topic - keyword driven
    "139_news_class_Politics",
    "140_news_class_Technology",
    "141_news_class_Entertainment",
    "161_agnews_0",
    "162_agnews_1",
    "163_agnews_2",
    # Wikidata entity properties - entity names
    "56_wikidatasex_or_gender",
    "57_wikidatais_alive",
    "58_wikidatapolitical_party",
    "59_wikidata_occupation_isjournalist",
    "60_wikidata_occupation_isathlete",
    "61_wikidata_occupation_isactor",
    "62_wikidata_occupation_ispolitician",
    "63_wikidata_occupation_issinger",
    "64_wikidata_occupation_isresearcher",
    # Historical figures - name based
    "5_hist_fig_ismale",
    "6_hist_fig_isamerican",
    "7_hist_fig_ispolitician",
    # Headlines with specific entities
    "21_headline_istrump",
    "22_headline_isobama",
    "23_headline_ischina",
    "24_headline_isiran",
    # Art type - format keywords
    "126_art_type_book",
    "127_art_type_song",
    "128_art_type_movie",
    # Disease/cancer - medical terminology
    "142_cancer_cat_Thyroid_Cancer",
    "143_cancer_cat_Lung_Cancer",
    "144_cancer_cat_Colon_Cancer",
    "145_disease_class_digestive system diseases",
    "146_disease_class_cardiovascular diseases",
    "147_disease_class_nervous system diseases",
    # Athlete sport - sport terminology
    "154_athlete_sport_football",
    "155_athlete_sport_basketball",
    "156_athlete_sport_baseball",
    # IT ticket category - technical terms
    "151_it_tick_HR Support",
    "152_it_tick_Hardware",
    "153_it_tick_Administrative rights",
    # Spam - lexical patterns
    "96_spam_is",
    # Compound words - lexical
    "65_high-school",
    "66_living-room",
    "67_social-security",
    "68_credit-card",
    "69_blood-pressure",
    "70_prime-factors",
    "71_social-media",
    "72_gene-expression",
    "73_control-group",
    "74_magnetic-field",
    "75_cell-lines",
    "76_trial-court",
    "77_second-derivative",
    "78_north-america",
    "79_human-rights",
    "80_side-effects",
    "81_public-health",
    "82_federal-government",
    "83_third-party",
    "84_clinical-trials",
    "85_mental-health",
}

# Context tasks - require cross-token reasoning
CONTEXT_TASKS = {
    # Sentiment - requires full text integration
    "92_glue_sst2",
    "113_movie_sent",
    "157_amazon_5star",
    "148_twt_emotion_worry",
    "149_twt_emotion_happiness",
    "150_twt_emotion_sadness",
    # Entailment/NLI - relational
    "87_glue_cola",
    "89_glue_mrpc",
    "90_glue_qnli",
    "91_glue_qqp",
    "136_glue_mnli_entailment",
    "137_glue_mnli_neutral",
    "138_glue_mnli_contradiction",
    # Reasoning/QA
    "36_sciq_tf",
    "41_truthqa_tf",
    "44_phys_tf",
    "47_reasoning_tf",
    "48_cm_correct",
    "54_cs_tf",
    # Temporal reasoning
    "42_temp_sense",
    "130_temp_cat_Frequency",
    "131_temp_cat_Typical Time",
    "132_temp_cat_Event Ordering",
    # Context type detection
    "133_context_type_Causality",
    "134_context_type_Belief_states",
    "135_context_type_Event_duration",
    # Ethics/deontic - requires understanding scenario
    "50_deon_isvalid",
    "51_just_is",
    "52_virtue_is",
    # Toxicity/hate - context dependent
    "95_toxic_is",
    "106_hate_hate",
    "107_hate_offensive",
    # Fake/clickbait - reasoning over claims
    "100_news_fake",
    "105_click_bait",
    "26_headline_isfrontpage",
    # AI generated - stylistic across text
    "94_ai_gen",
    "110_aimade_humangpt3",
    # Arithmetic - requires computation
    "129_arith_mc_A",
    # Short answer detection
    "49_cm_isshort",
}


def categorize_task(dataset_name):
    if dataset_name in LOCAL_TASKS:
        return "local"
    elif dataset_name in CONTEXT_TASKS:
        return "context"
    else:
        return "unknown"


def plot_by_category(omp_path, ksparse_path, save_path=None):
    omp_results = load_results(omp_path)
    ksparse_results = load_results(ksparse_path)
    
    omp_datasets = {d["dataset_name"]: d for d in omp_results["eval_result_details"]}
    ksparse_datasets = {d["dataset_name"]: d for d in ksparse_results["eval_result_details"]}
    
    all_dataset_names = sorted(set(omp_datasets.keys()) & set(ksparse_datasets.keys()))
    
    # Categorize
    categorized = {"context": [], "local": [], "unknown": []}
    for name in all_dataset_names:
        cat = categorize_task(name)
        categorized[cat].append(name)
    
    # Print categorization for verification
    print("=== LOCAL TASKS ===")
    for name in categorized["local"]:
        print(f"  {name}")
    print(f"\n=== CONTEXT TASKS ===")
    for name in categorized["context"]:
        print(f"  {name}")
    print(f"\n=== UNKNOWN ===")
    for name in categorized["unknown"]:
        print(f"  {name}")
    
    k_values = [1, 2, 5, 10, 20, 50, 100]
    metrics = [
        ("test_accuracy", "Accuracy"),
        ("test_auc", "AUC"),
        ("test_f1", "F1"),
    ]
    
    # Create figure: 2 rows (local, context) x 3 cols (metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    category_order = ["local", "context"]
    category_labels = ["Local (Token-Level)", "Context-Dependent"]
    
    for row_idx, (category, cat_label) in enumerate(zip(category_order, category_labels)):
        task_names = categorized[category]
        
        if not task_names:
            continue
        
        for col_idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            # Aggregate across tasks in category
            omp_by_k = {k: [] for k in k_values}
            ksparse_by_k = {k: [] for k in k_values}
            llm_scores = []
            
            for name in task_names:
                omp_data = omp_datasets.get(name, {})
                ksparse_data = ksparse_datasets.get(name, {})
                
                for k in k_values:
                    key = f"sae_top_{k}_{metric_key}"
                    omp_val = omp_data.get(key)
                    ksparse_val = ksparse_data.get(key)
                    if omp_val is not None:
                        omp_by_k[k].append(omp_val)
                    if ksparse_val is not None:
                        ksparse_by_k[k].append(ksparse_val)
                
                llm_key = f"llm_{metric_key}"
                llm_val = omp_data.get(llm_key) or ksparse_data.get(llm_key)
                if llm_val is not None:
                    llm_scores.append(llm_val)
            
            # Compute averages
            omp_avg = []
            omp_k = []
            for k in k_values:
                if omp_by_k[k]:
                    omp_avg.append(np.mean(omp_by_k[k]))
                    omp_k.append(k)
            
            ksparse_avg = []
            ksparse_k = []
            for k in k_values:
                if ksparse_by_k[k]:
                    ksparse_avg.append(np.mean(ksparse_by_k[k]))
                    ksparse_k.append(k)
            
            llm_avg = np.mean(llm_scores) if llm_scores else None
            
            # Plot
            if omp_avg:
                ax.plot(omp_k, omp_avg, "o-", color="tab:blue", linewidth=2, markersize=6, label="OMP")
            if ksparse_avg:
                ax.plot(ksparse_k, ksparse_avg, "s--", color="tab:red", linewidth=2, markersize=6, label="Mean-Diff")
            
            if llm_avg is not None:
                ax.axhline(y=llm_avg, color="tab:green", linestyle="--", alpha=0.7,
                          label=f"LLM Dense ({llm_avg:.3f})")
            
            all_k = sorted(set(omp_k + ksparse_k))
            if all_k:
                ax.set_xscale("log")
                ax.set_xticks(all_k)
                ax.set_xticklabels([str(k) for k in all_k])
            ax.set_xlabel("k")
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis range
            all_vals = omp_avg + ksparse_avg + ([llm_avg] if llm_avg else [])
            if all_vals:
                y_min = min(all_vals) - 0.02
                y_max = max(all_vals) + 0.02
                ax.set_ylim(max(0.5, y_min), min(1, y_max))
            
            # Title and legend
            if col_idx == 1:
                n_tasks = len(task_names)
                ax.set_title(f"{cat_label} (n={n_tasks})", fontsize=12, fontweight="bold")
            if col_idx == 2:
                ax.legend(fontsize=9, loc="lower right")
    
    plt.suptitle("OMP vs Mean-Diff: Local vs Context-Dependent Tasks", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_by_category.py <omp_json> <ksparse_json> [save_path]")
        sys.exit(1)
    
    omp_path = sys.argv[1]
    ksparse_path = sys.argv[2]
    save_path = sys.argv[3] if len(sys.argv) > 3 else None
    plot_by_category(omp_path, ksparse_path, save_path)