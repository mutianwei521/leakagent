# Water distribution network partitioning tools module
from .hydraulic import run_hydraulic_simulation, calculate_average_pressure
from .similarity import build_similarity_matrix, create_network_graph
from .partitioning import run_louvain_partitioning, extract_unique_partitions, generate_merged_partitions, extract_partitions_with_merge
from .visualization import plot_partition, save_all_partitions_plots
from .visual_perception import (
    generate_pressure_heatmap,
    generate_flow_visualization,
    generate_combined_heatmap,
    extract_visual_features,
    analyze_network_visually,
    get_vlm_prompt_template
)

