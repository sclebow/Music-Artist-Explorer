import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.patches as mpatches
import random
import io
import gravis as gv

def get_mention_threshold_values(artist, df):
    """
    Get the minimum, maximum, and default mention threshold values for the given artist.
    """
    # Get the number of mentions for the selected artist
    mentions = df[df['source'] == artist]['number_of_mentions'].values
    if len(mentions) > 0:
        min_mention_threshold = int(mentions.min())
        max_mention_threshold = int(mentions.max())
        # default_mention_threshold = int(np.median(mentions))
        # default_mention_threshold = int(np.mean(mentions))
        # default_mention_threshold = int(np.percentile(mentions, 75))  # 75th percentile as default
        default_mention_threshold = max(int(max_mention_threshold * 0.5), 1)  # 50% of max mentions
        if min_mention_threshold == max_mention_threshold:
            max_mention_threshold += 1
    else:
        min_mention_threshold = 1
        max_mention_threshold = 20
        default_mention_threshold = 8
    
    print(f"Artist: {artist}, Min: {min_mention_threshold}, Max: {max_mention_threshold}, Default: {default_mention_threshold}")

    return min_mention_threshold, max_mention_threshold, default_mention_threshold

def filter_dataframe_with_neighbors(dataframe, artist, max_depth=1, display_df=False):
    """
    Get neighbors of an artist up to a certain depth.
    Returns:
        - filtered_df: DataFrame containing only the rows with the artist as source or target, with depth 
    """

    # print()
    # print()
    # print(f'max_depth: {type(max_depth)}')

    # Filter the DataFrame to include only rows with the artist as source
    filtered_df = dataframe[(dataframe['source'] == artist)]

    # Recursively find targets up to max_depth using a function
    def find_targets(dataframe, source_artist, depth_of_source, max_depth):
        # Base case: if the maximum depth is reached, return an empty DataFrame
        if depth_of_source >= max_depth:
            return pd.DataFrame()

        # Filter the DataFrame to include only rows with the source artist as target
        new_df = dataframe[dataframe['source'] == source_artist]

        # Recursively find targets for these new sources
        for index, row in new_df.iterrows():
            target_artist = row['target']
            new_df = pd.concat([new_df, find_targets(dataframe, target_artist, depth_of_source + 1, max_depth)], ignore_index=True)

        return new_df
    
    filtered_df = pd.concat([filtered_df, find_targets(dataframe, artist, 0, max_depth)], ignore_index=True)

    # Filter the DataFrame to include only rows with the artist as target
    filtered_df_2 = dataframe[(dataframe['target'] == artist)]

    # Recursively find sources up to max_depth
    for depth in range(1, max_depth):
        # Get the current depth's sources
        current_sources = filtered_df_2['source'].unique()
        
        # Filter the DataFrame to include only rows with the current sources as target
        new_df = dataframe[dataframe['target'].isin(current_sources)]

        # Filter the DataFrame to include only rows with the current sources as source
        new_df = new_df[new_df['source'].isin(current_sources)]

        # Append to the filtered DataFrame
        filtered_df_2 = pd.concat([filtered_df_2, new_df], ignore_index=True)

    # Append to the filtered DataFrame
    filtered_df = pd.concat([filtered_df, filtered_df_2], ignore_index=True)

    # Remove duplicates, keeping the smallest depth
    filtered_df = filtered_df.drop_duplicates()

    # Reset the index
    filtered_df = filtered_df.reset_index(drop=True)

    # print(f"Filtered dataframe shape after filtering: {filtered_df.shape}")
    
    return filtered_df

def generate_graph(center_artist, max_depth, mention_threshold, three_d=False):
    """
    Generate a graph based on the Wikipedia mentions of artists.
    """
    
    # Create a graph object
    graph = nx.DiGraph()

    df = pd.read_csv('wikipedia_music_graph.csv') # Columns source,target,number_of_mentions

    total_lines = len(df)
    # print(f'Total lines: {total_lines}')

    # Drop lines below the mention threshold
    df = df[df['number_of_mentions'] >= mention_threshold]
    # print(f'Lines after filtering by mention threshold: {len(df)}')

    # Filter the DataFrame to include only rows with the center artist as source or target
    df = filter_dataframe_with_neighbors(df, center_artist, max_depth, display_df=True)
    
    # Create a streamlit progress bar
    progress_bar = st.progress(0)

    # Initialize node sizes, colors, and depths
    node_depths = {}

    # Add nodes and edges based on your DataFrame
    for index, row in df.iterrows():
        source = row['source']
        target = row['target']
        number_of_mentions = row['number_of_mentions']
        
        # Add nodes and edges to the graph
        if source not in graph:
            graph.add_node(source)
        if target not in graph:
            graph.add_node(target)
        graph.add_edge(source, target, weight=number_of_mentions)

        # # print(f'Progress: {index / total_lines:.2%}', end='\r')

        # Update progress bar
        progress_bar.progress((index + 1) / total_lines)

    # print(f'node_depths: {node_depths}')

    # Close the progress bar
    progress_bar.empty()

    node_sizes = {}
    node_colors = {}
    color_palette = sns.color_palette("coolwarm", n_colors=max_depth + 1)  # Color palette for depth levels

    def get_node_size(depth, max_depth):
        """
        Get node size based on depth.
        """
        min_size = 100
        max_size = 300
        size_range = max_size - min_size
        size = min_size + ((max_depth - depth) / max_depth) * size_range
        return size
    print()
    # Set node sizes and colors based on depth

    # Initialize a list to keep track of nodes to remove
    nodes_to_remove = []
    for node in graph.nodes():
        # Get the depth of the node from the center artist
        if node == center_artist:
            node_depths[node] = 0
            node_sizes[node] = 300
            node_colors[node] = color_palette[0]
        else:
            # Find the shortest path from the center artist to the node
            try:
                path_length = nx.shortest_path_length(graph, source=center_artist, target=node)
                try:
                    node_colors[node] = color_palette[path_length]
                except IndexError:
                    # print(f"Node {node} exceeds max depth {max_depth}. Removing it.")
                    nodes_to_remove.append(node)                    
                    continue
                node_depths[node] = path_length
                node_sizes[node] = get_node_size(node_depths[node], max_depth)
            except nx.NetworkXNoPath:
                # print(f"Node {node} is not reachable from {center_artist}. Removing it.")
                nodes_to_remove.append(node)
                continue

    for node in nodes_to_remove:
        # Remove nodes that exceed the max depth
        graph.remove_node(node)
        
    # Remove the nodes from the dataframe if the node is in the source or target
    df = df[~df['source'].isin(nodes_to_remove)]
    df = df[~df['target'].isin(nodes_to_remove)]
    with st.expander("Filtered DataFrame", expanded=False):
        st.dataframe(df, use_container_width=True)  # Display the filtered dataframe in Streamlit

    # for node, depth in node_depths.items():
    #     # Set node size based on depth
    #     if depth == 0:
    #         node_sizes[node] = 300
    #     elif depth == 1:
    #         node_sizes[node] = 200
    #     else:
    #         node_sizes[node] = 100
    #     # Set node color based on depth, using the color palette
    #     node_colors[node] = color_palette[depth]        

    # Create a custom layout with better node distribution
    # First, get initial positions
    seed = 42  # Seed for reproducibility 

    pos = nx.spring_layout(graph, k=1.0, seed=seed)  # Use a seed for reproducibility

    # # Force center artist to be at the center (0,0)
    pos[center_artist] = (0, 0)

    # Run spring layout again with higher repulsion force and more iterations
    pos = nx.spring_layout(graph, 
                        pos=pos,               # Start with existing positions
                        fixed=[center_artist], # Keep center fixed
                        k=4,                   # Higher repulsion between nodes
                        iterations=100,        # More iterations for better convergence
                        weight='weight' * -1,       # Use edge weights, higher weights mean closer nodes
                        seed=seed)               # For reproducible layout

    # plt.figure(figsize=(10, 6))

    # Draw nodes with varying sizes and colors
    nx.draw_networkx_nodes(graph, pos, 
                        node_size=[node_sizes.get(node, 100) for node in graph.nodes()],
                        node_color=[node_colors.get(node, 'lightblue') for node in graph.nodes()])

    # Draw labels with bbox (background box)
    nx.draw_networkx_labels(graph, pos, font_size=6, font_weight='bold', 
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3))

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(graph, 'weight')

    # Calculate edge widths and alpha based on weights
    max_width = 8.0
    min_width = 0.3
    max_alpha = 0.9
    min_alpha = 0.5
    # Normalize edge weights to get widths
    edge_weights = [edge_labels[edge] for edge in graph.edges()]
    edge_widths = {}
    alphas = {}
    for edge in graph.edges():
        weight = edge_labels[edge]
        
        # Normalize weight to a range [min_width, max_width]
        width = min_width + (weight / max(edge_weights)) * (max_width - min_width)
        edge_widths[edge] = width

        # Normalize alpha to a range [min_alpha, max_alpha]
        alpha = min_alpha + (weight / max(edge_weights)) * (max_alpha - min_alpha)
        alphas[edge] = alpha

    # Draw all edges as curves with calculated widths
    for edge in graph.edges():
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=[edge],
            width=edge_widths[edge],
            alpha=alphas[edge],
            edge_color='gray',
            connectionstyle='arc3,rad=0.15'  # All edges curved with same radius
        )
        graph[edge[0]][edge[1]]['color'] = 'gray'  # Set edge color to gray
    
    # Draw edge labels on curves
    for edge, weight in edge_labels.items():
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels={edge: weight},
            font_color='red',
            font_size=4,
            alpha=0.8,
            rotate=False,
            connectionstyle='arc3,rad=0.15'  # Match curve of the edges
        )
    
    # # Set node colors as node attributes in the graph
    # for node, color in node_colors.items():
    #     if node in graph.nodes():
    #         color = color_palette[node_depths[node]]
    #         print(f"Node: {node}, Color: {color}")
    #         # Convert the RGB tuple to hex color string for d3
    #         hex_color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
    #         graph.nodes[node]['color'] = hex_color
    #         # Also add depth as a node attribute
    #         graph.nodes[node]['depth'] = max_depth - node_depths.get(node, 0)

    for node in graph.nodes():
        depth = node_depths.get(node, 0)
        # print(f"Node: {node}, Depth: {depth}")

        color = color_palette[depth]
        # Convert the RGB tuple to hex color string for d3
        hex_color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
        graph.nodes[node]['color'] = hex_color
        # Also add depth as a node attribute
        graph.nodes[node]['depth'] = max_depth - depth

    if three_d:
        renderer = gv.three(
                graph,
                use_node_size_normalization=True, 
                node_size_normalization_max=30,
                use_edge_size_normalization=True,
                edge_size_data_source='weight', 
                edge_curvature=0.3,
                node_hover_neighborhood=True,
                show_edge_label=True,
                edge_label_data_source='weight',
                node_label_size_factor=0.5,
                edge_size_factor=0.5,
                edge_label_size_factor=0.5,
                node_size_data_source='depth',
                layout_algorithm_active=True,
                # use_links_force=True,
                # links_force_distance=200,
                use_many_body_force=True,
                many_body_force_strength=-300,
                zoom_factor=1.5,
                graph_height=550,
            )    
    else:
        renderer = gv.d3(
                graph, 
                use_node_size_normalization=True, 
                node_size_normalization_max=30,
                use_edge_size_normalization=True,
                edge_size_data_source='weight', 
                edge_curvature=0.3,
                node_hover_neighborhood=True,
                show_edge_label=True,
                edge_label_data_source='weight',
                node_label_size_factor=0.5,
                edge_size_factor=0.5,
                edge_label_size_factor=0.5,
                node_size_data_source='depth',
                layout_algorithm_active=True,
                # use_links_force=True,
                # links_force_distance=200,
                use_many_body_force=True,
                many_body_force_strength=-300,
                zoom_factor=1.5,
                graph_height=550,
                use_centering_force=True,
            )
    
    return renderer

    # # Create a legend showing what each color means in terms of depth
    
    # # Create legend handles
    # legend_handles = []
    # for i, c in enumerate(color_palette):
    #     if i == 0:
    #         legend_handles.append(mpatches.Patch(color=c, label=f'Center Artist: {center_artist}'))
    #     else:
    #         legend_handles.append(mpatches.Patch(color=c, label=f'Depth {i}'))
    
    # # Determine best location for legend by finding quadrant with fewest nodes
    # top_right = sum(1 for x, y in pos.values() if x > 0 and y > 0)
    # top_left = sum(1 for x, y in pos.values() if x < 0 and y > 0)
    # bottom_right = sum(1 for x, y in pos.values() if x > 0 and y < 0)
    # bottom_left = sum(1 for x, y in pos.values() if x < 0 and y < 0)
    
    # # Find quadrant with minimum node count
    # counts = {
    #     'upper right': top_right, 
    #     'upper left': top_left, 
    #     'lower right': bottom_right, 
    #     'lower left': bottom_left
    # }
    # best_position = min(counts, key=counts.get)
    
    # # Add the legend to the plot in best position
    # plt.legend(handles=legend_handles, 
    #           title="Distance from Center Artist",
    #           title_fontproperties={'size': "small"},
    #           loc=best_position, 
    #           fontsize="xx-small",
    #           frameon=True,
    #           facecolor=(1, 1, 1, 0.2),  # White with some transparency
    #           )
    
    # title = f"""Subgraph of Musical Artist Wikipedia mentions on the Wikipedia page of {center_artist}\n\
    #     Depth: {max_depth}, Minimum Mention Threshold: {mention_threshold}"""

    # plt.title(title)
    # plt.axis('off')  # Turn off axis
    # plt.tight_layout()

    # return plt

def get_random_artist(df):
    """
    Get a random artist from the list of artists.
    """
    artists = df['source'].unique()
    # Remove the current artist from the list of artists
    current_artist = st.session_state.artist
    artists = [artist for artist in artists if artist != current_artist]
    return random.choice(artists)

# Streamlit app
# Set page configuration
st.set_page_config(page_title="Musical Artist Connectivity Analysis", layout="wide")
# Load CSS styles
st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True) 

# col1, col2 = st.columns([3, 1], gap="large", vertical_alignment="bottom")
# with col1:
#     st.title("Musical Artist Connectivity Analysis")
# with col2:
#     st.markdown("### by Scott Lebow")
st.markdown("<h1 style='text-align: center;'>Musical Artist Connectivity Analysis</h1><h3 style='text-align: center;'>by Scott Lebow</h3>", unsafe_allow_html=True)
st.markdown("----")
st.write("This app analyzes the connections between musical artists based on their Wikipedia pages.  It uses a dataset of artists and their connections to visualize the relationships between them.")

# Get all artists from the dataset
df = pd.read_csv('wikipedia_music_graph.csv')

with st.expander("Dataset Overview", expanded=False):
    st.write("This dataset contains information about musical artists and their connections based on Wikipedia mentions.")
    st.write("It is based on the Kaggle dataset [Wikipedia Music Graph](https://www.kaggle.com/datasets/matwario/wikipedia-music-links).")
    st.write("Refactored by Scott Lebow for the purpose of this project.")
    st.dataframe(df, use_container_width=True)  # Display the dataframe in Streamlit

artists = df['source'].unique()

# Default artist for the session
default_artist = "The Beatles"
# Check if the session state already has an artist
if 'artist' not in st.session_state:
    st.session_state.artist = default_artist  # Set default artist
if 'artist_index' not in st.session_state:
    st.session_state.artist_index = np.where(artists == default_artist)[0][0]

# Determine the default values for the mention threshold slider
mention_threshold_min, mention_threshold_max, mention_threshold_default = get_mention_threshold_values(st.session_state.artist, df)

col1, col2 = st.columns(2, vertical_alignment="bottom")
with col1:
    artist = st.selectbox("Select an artist:", artists, index=int(st.session_state.artist_index), help="Select an artist to visualize their connections.", key="artist_selectbox")
with col2:
    random_artist = st.button("Get random artist", help="Click to select a random artist from the dataset.")

col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
with col1:
    mention_threshold = st.slider("Minimum mention threshold:", min_value=mention_threshold_min, max_value=mention_threshold_max, 
                                  value=mention_threshold_default, step=1,  
                                  help="Minimum number of mentions to consider an artist connected.")
with col2:
    max_depth = st.slider("Maximum depth of connections:", min_value=1, max_value=10, value=3, step=1, help="Maximum depth of connections to visualize.")
with col3:
    three_d = st.checkbox("3D view", value=False, help="Check to view the graph in 3D.")

if random_artist:
    artist = get_random_artist(df)  # Get a random artist
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
        plot = generate_graph(artist, max_depth, mention_threshold, three_d=three_d)
        success_message = st.success("Graph generated successfully!") # Show success message

        st.components.v1.html(plot.to_html(), height=550)  # Display the graph using Streamlit's HTML component

        st.balloons()  # Show balloons animation when the graph is generated