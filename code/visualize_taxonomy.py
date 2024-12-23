#!/usr/bin/env python3

import os
import argparse
from collections import defaultdict
import graphviz

def read_term_clusters(file_path):
    """Read and parse term clusters file."""
    clusters = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                cluster_id, term, confidence = parts
                clusters[cluster_id].append((term, float(confidence)))
    
    # Sort terms by confidence score within each cluster
    for cluster_id in clusters:
        clusters[cluster_id].sort(key=lambda x: x[1], reverse=True)
    
    return clusters

def create_taxonomy_visualization(clusters, output_path, max_terms_per_cluster=10):
    """Create a visual taxonomy using graphviz."""
    dot = graphviz.Digraph(comment='Taxonomy Visualization')
    dot.attr(rankdir='TB')  # Top to bottom layout
    
    # Add root node
    dot.node('root', 'Root Taxonomy', shape='box')
    
    # Color palette for different clusters
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Add cluster nodes and connect to root
    for cluster_id, terms in clusters.items():
        # Get the top term as the cluster label
        cluster_label = terms[0][0] if terms else f"Cluster {cluster_id}"
        cluster_size = len(terms)
        
        # Create cluster node
        cluster_node_id = f'cluster_{cluster_id}'
        cluster_label_with_size = f"{cluster_label}\n(N={cluster_size})"
        color = colors[int(cluster_id) % len(colors)]
        dot.node(cluster_node_id, cluster_label_with_size, 
                shape='box', style='filled', fillcolor=color, fontcolor='white')
        dot.edge('root', cluster_node_id)
        
        # Add top terms as child nodes
        for i, (term, confidence) in enumerate(terms[:max_terms_per_cluster]):
            if i == 0:  # Skip the first term as it's used as cluster label
                continue
            term_node_id = f'term_{cluster_id}_{i}'
            term_label = f"{term}\n(conf={confidence:.3f})"
            dot.node(term_node_id, term_label, shape='ellipse', 
                    style='filled', fillcolor=f"{color}22")  # Light version of cluster color
            dot.edge(cluster_node_id, term_node_id)
    
    # Save the visualization
    dot.render(output_path, format='pdf', cleanup=True)
    print(f"Taxonomy visualization saved to {output_path}.pdf")

def main():
    parser = argparse.ArgumentParser(description='Create a visual taxonomy from clustering results')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to term_clusters.txt file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save the visualization (without extension)')
    parser.add_argument('--max-terms', type=int, default=10,
                      help='Maximum number of terms to show per cluster')
    
    args = parser.parse_args()
    
    # Read clusters
    clusters = read_term_clusters(args.input)
    
    # Create visualization
    create_taxonomy_visualization(clusters, args.output, args.max_terms)

if __name__ == "__main__":
    main()
