import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.patches as mpatches
import random
import io

def get_neighbors(n_graph, artist, max_depth=1):
    """
    Get neighbors of an artist up to a certain depth.
    Returns:
        - nodes_set: Set of all neighbor nodes
        - node_sizes: Dictionary of node sizes based on depth
        - node_colors: Dictionary of node colors based on depth
        - node_depths: Dictionary mapping nodes to their depths
    """
    nodes_set = set()
    node_depths = {artist: 0}
    
    def explore_neighbors(node, current_depth):
        if current_depth > max_depth:
            return
        for neighbor in n_graph.neighbors(node):
            # Add neighbor to the set
            nodes_set.add(neighbor)
            # Record depth (use minimum if node is found through multiple paths)
            if neighbor not in node_depths or current_depth < node_depths[neighbor]:
                node_depths[neighbor] = current_depth
            # Explore next level
            explore_neighbors(neighbor, current_depth + 1)
    
    # Start exploration from the center artist
    explore_neighbors(artist, 1)
    
    # Add the center artist to the set
    nodes_set.add(artist)
    
    # Generate node sizes - bigger for center and closer nodes
    node_sizes = {}
    max_size = 300  # Max size for center node
    min_size = 100  # Min size for furthest nodes
    for node, depth_1 in node_depths.items():
        # Size decreases with depth: 300 for center, 200 for depth 1, etc.
        node_sizes[node] = max_size - (depth_1 * (max_size - min_size) / max_depth)
    
    # Generate node colors - using a color gradient based on depth
    node_colors = {}
    for node, depth_1 in node_depths.items():
        # Color varies from red (center) to blue (max depth)
        if depth_1 == 0:
            node_colors[node] = 'red'  # Center node
        elif depth_1 == max_depth:
            node_colors[node] = 'blue'  # Furthest nodes
        else:
            # Nodes in between get intermediate colors
            node_colors[node] = f'#{int(255 - 255*(depth_1/max_depth)):02x}{0:02x}{int(255*(depth_1/max_depth)):02x}'
    
    return nodes_set, node_sizes, node_colors, node_depths

def generate_graph(center_artist, max_depth, mention_threshold, show_plot=False, save_plot=False):
    """
    Generate a graph based on the Wikipedia mentions of artists.
    """
    
    # Create a graph object
    graph = nx.DiGraph()

    df = pd.read_csv('wikipedia_music_graph.csv') # Columns source,target,number_of_mentions

    total_lines = len(df)
    print(f'Total lines: {total_lines}')

    # Drop lines below the mention threshold
    df = df[df['number_of_mentions'] >= mention_threshold]
    print(f'Lines after filtering by mention threshold: {len(df)}')

    # Create a streamlit progress bar
    progress_bar = st.progress(0)

    # Add nodes and edges based on your DataFrame
    for index, row in df.iterrows():
        source = row['source']
        target = row['target']
        number_of_mentions = row['number_of_mentions']
        
        # Add nodes and edges to the graph
        graph.add_node(source)
        graph.add_node(target)
        graph.add_edge(source, target, weight=number_of_mentions)

        print(f'Progress: {index / total_lines:.2%}', end='\r')# Get subgraph nodes and styling information
        # Update progress bar
        progress_bar.progress((index + 1) / total_lines)

    # Close the progress bar
    progress_bar.empty()

    subgraph_nodes, node_sizes, node_colors, node_depths = get_neighbors(graph, center_artist, max_depth)
    subgraph_nodes.add(center_artist)  # Include the center artist

    # Create a new graph instead of a subgraph view which is frozen
    subgraph = nx.DiGraph()
    # Copy nodes and edges from original graph
    for node in subgraph_nodes:
        subgraph.add_node(node)
    for u, v, data in graph.edges(data=True):
        if u in subgraph_nodes and v in subgraph_nodes:
            subgraph.add_edge(u, v, **data)

    # Create a custom layout with better node distribution
    # First, get initial positions
    pos = nx.spring_layout(subgraph, k=1.0)  # Higher k means more repulsion

    # Force center artist to be at the center (0,0)
    pos[center_artist] = (0, 0)

    # Run spring layout again with higher repulsion force and more iterations
    pos = nx.spring_layout(subgraph, 
                        pos=pos,               # Start with existing positions
                        fixed=[center_artist], # Keep center fixed
                        k=1,                 # Higher repulsion between nodes
                        iterations=100,        # More iterations for better convergence
                        weight='weight',       # Use edge weights
                        seed=42)               # For reproducible layout

    # Improve node distribution, especially for higher depths
    for node in pos:
        if node != center_artist:
            # Scale based on depth to create more distinct "rings"
            depth_factor = node_depths.get(node, 1)
            # More aggressive scaling - create more distinct separation by depth
            # Higher power (^1.5) creates more separation between depth levels
            scaling_factor = (depth_factor ** 1.2) * 1  # Adjust scaling factor as needed
            
            # Scale by depth - deeper nodes further out
            x, y = pos[node]
            norm = (x**2 + y**2)**0.5  # Distance from center
            if norm > 0:
                # Increase distance for nodes of the same depth
                pos[node] = (x/norm * scaling_factor, y/norm * scaling_factor)
    
    # Add jitter to prevent overlapping nodes at the same depth
    random.seed(42)  # For reproducibility
    jitter_amount = 2 # Amount of jitter to add
    for node in pos:
        if node != center_artist:
            x, y = pos[node]
            # Add small random offset
            pos[node] = (
                x + random.uniform(-jitter_amount, jitter_amount),
                y + random.uniform(-jitter_amount, jitter_amount)
            )

    plt.figure(figsize=(10, 8))

    # Draw nodes with varying sizes and colors
    nx.draw_networkx_nodes(subgraph, pos, 
                        node_size=[node_sizes.get(node, 100) for node in subgraph.nodes()],
                        node_color=[node_colors.get(node, 'lightblue') for node in subgraph.nodes()])

    # Draw labels with bbox (background box)
    nx.draw_networkx_labels(subgraph, pos, font_size=6, font_weight='bold', 
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3))

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(subgraph, 'weight')

    # Calculate edge widths and alpha based on weights
    max_width = 8.0
    min_width = 0.3
    max_alpha = 0.9
    min_alpha = 0.5
    # Normalize edge weights to get widths
    edge_weights = [edge_labels[edge] for edge in subgraph.edges()]
    edge_widths = {}
    alphas = {}
    for edge in subgraph.edges():
        weight = edge_labels[edge]
        
        # Normalize weight to a range [min_width, max_width]
        width = min_width + (weight / max(edge_weights)) * (max_width - min_width)
        edge_widths[edge] = width

        # Normalize alpha to a range [min_alpha, max_alpha]
        alpha = min_alpha + (weight / max(edge_weights)) * (max_alpha - min_alpha)
        alphas[edge] = alpha

    # Draw all edges as curves with calculated widths
    for edge in subgraph.edges():
        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=[edge],
            width=edge_widths[edge],
            alpha=alphas[edge],
            edge_color='gray',
            connectionstyle='arc3,rad=0.15'  # All edges curved with same radius
        )
    
    # Draw edge labels on curves
    for edge, weight in edge_labels.items():
        nx.draw_networkx_edge_labels(
            subgraph, pos,
            edge_labels={edge: weight},
            font_color='red',
            font_size=4,
            alpha=0.8,
            rotate=False,
            connectionstyle='arc3,rad=0.15'  # Match curve of the edges
        )

    # Create a legend showing what each color means in terms of depth
    
    # Get unique depths and sort them
    unique_depths = sorted(set(node_depths.values()))
    
    # Create legend handles
    legend_handles = []
    for d in unique_depths:
        # Find a node with this depth to get its color
        for node, depth_2 in node_depths.items():
            if depth_2 == d:
                color = node_colors[node]
                break
                
        if d == 0:
            label = f"Depth {d}: {center_artist}"
        else:
            label = f"Depth {d}"
            
        legend_handles.append(mpatches.Patch(color=color, label=label))
    
    # Determine best location for legend by finding quadrant with fewest nodes
    top_right = sum(1 for x, y in pos.values() if x > 0 and y > 0)
    top_left = sum(1 for x, y in pos.values() if x < 0 and y > 0)
    bottom_right = sum(1 for x, y in pos.values() if x > 0 and y < 0)
    bottom_left = sum(1 for x, y in pos.values() if x < 0 and y < 0)
    
    # Find quadrant with minimum node count
    counts = {
        'upper right': top_right, 
        'upper left': top_left, 
        'lower right': bottom_right, 
        'lower left': bottom_left
    }
    best_position = min(counts, key=counts.get)
    
    # Add the legend to the plot in best position
    plt.legend(handles=legend_handles, 
              title="Distance from Center Artist",
              loc=best_position, 
              fontsize="small",
              frameon=True,
              facecolor="white",
              edgecolor="lightgray")
    
    title = f"""Subgraph of Musical Artist Wikipedia mentions on the Wikipedia page of {center_artist}\n\
        Depth: {max_depth}, Minimum Mention Threshold: {mention_threshold}"""

    plt.title(title)
    plt.axis('off')  # Turn off axis
    plt.tight_layout()

    # if save_plot:
    #     # Save the figure as a PNG file
    #     file_name = f'{center_artist}_{max_depth}_depth_{mention_threshold}_mention_threshold_subgraph.png'
    #     plt.savefig(f'images/{file_name}', format='png', dpi=300, bbox_inches='tight')
    #     print(f"Graph saved as {file_name}")

    # if show_plot:
    #     plt.show()

    # # plt.close()
    return plt

def get_random_artist():
    """
    Get a random artist from the list of artists.
    """
    df_0 = pd.read_csv('wikipedia_music.csv')
    artists = df_0['ARTIST_NAME'].unique()
    return random.choice(artists)

# Streamlit app
# Set page configuration
st.set_page_config(page_title="Musical Artist Connectivity Analysis", layout="wide")
# Load CSS styles
st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True) 

st.title("Musical Artist Connectivity Analysis")
st.write("This app analyzes the connections between musical artists based on their Wikipedia pages.  It uses a dataset of artists and their connections to visualize the relationships between them.")

# Get all artists from the dataset
df = pd.read_csv('wikipedia_music.csv')

artists = df['ARTIST_NAME'].unique()

# Default artist for the session
default_artist = "The Beatles"
# Check if the session state already has an artist
if 'artist' not in st.session_state:
    st.session_state.artist = default_artist  # Set default artist
if 'artist_index' not in st.session_state:
    st.session_state.artist_index = np.where(artists == default_artist)[0][0]

print(f"Default artist index: {st.session_state.artist_index}")

col1, col2 = st.columns(2, vertical_alignment="bottom")
with col1:
    artist = st.selectbox("Select an artist:", artists, index=int(st.session_state.artist_index), help="Select an artist to visualize their connections.", key="artist_selectbox")
with col2:
    random_artist = st.button("Get random artist", help="Click to select a random artist from the dataset.")

col1, col2 = st.columns(2, vertical_alignment="bottom")
with col1:
    mention_threshold = st.slider("Minimum mention threshold:", min_value=1, max_value=20, value=1, step=1, help="Minimum number of mentions to consider an artist connected.")
with col2:
    max_depth = st.slider("Maximum depth of connections:", min_value=1, max_value=10, value=2, step=1, help="Maximum depth of connections to visualize.")

if random_artist:
    artist = get_random_artist()
    st.session_state.artist = artist  # Store the random artist in session state
    st.session_state.artist_index = np.where(artists == artist)[0][0]  # Store the index of the random artist
    # Update the selectbox to reflect the random artist
    st.rerun()  # Rerun the app to update the selectbox with the random artist

# Display the selected artist
st.markdown(f"## Selected artist: {artist}")

# Generate the graph based on the selected artist and parameters
# if st.button("Generate graph", help="Click to generate the graph based on the selected artist and parameters.", key="generate_graph"):
if True:  # Always generate the graph for demonstration purposes
    with st.spinner("Generating graph... (this may take a few seconds depending on the artist, threshold, and depth)"):
        plot = generate_graph(artist, max_depth, mention_threshold, show_plot=False, save_plot=False)
        success_message = st.success("Graph generated successfully!") # Show success message
        st.pyplot(plot, use_container_width=False)  # Adjusted to use the container width

        st.balloons()  # Show balloons animation when the graph is generated
        # Remove the success message after a few seconds
        success_message.empty()

        # Add download button for the graph, set the dpi to 300 for better quality
        file_name = f'{artist}_{max_depth}_depth_{mention_threshold}_mention_threshold_subgraph.png'
        img = io.BytesIO()
        plot.savefig(img, format='png', dpi=300, bbox_inches='tight')
        st.download_button(
            label="Download graph",
            data=img,  # Get the figure as a PNG image
            file_name=file_name,
            mime="image/png",
            help="Click to download the generated graph as a PNG file."
        )
