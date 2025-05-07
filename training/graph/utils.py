# /training/graph/utils.py

import torch
import random
import numpy as np
import os
import json
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd # Adicionado para carregar info extra

# (set_seed, setup_logging, save_config, load_config permanecem iguais)
def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_logging(log_dir: str, log_filename: str = "training.log"):
    """Configures logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    # Diminuir log de matplotlib e PIL
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] (%(module)s:%(lineno)d) %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete (Level: DEBUG).")
    logging.info(f"Logs will be saved to: {log_path}")

def save_config(args, output_dir: str):
    """Saves the training configuration (argparse args) to a JSON file."""
    config_path = os.path.join(output_dir, 'config.json')
    args_dict = vars(args).copy()
    if 'device' in args_dict:
        args_dict['device'] = str(args_dict['device'])
    try:
        with open(config_path, 'w') as f:
            json.dump(args_dict, f, indent=4)
        logging.info(f"Training configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save configuration: {e}")

def load_config(config_path: str):
    """Loads a training configuration from a JSON file."""
    if not os.path.exists(config_path):
         logging.error(f"Configuration file not found: {config_path}")
         return None
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config_dict
    except Exception as e:
         logging.error(f"Failed to load configuration: {e}")
         return None

# --- ALTERAÇÃO: Nova função para plotar embeddings com t-SNE ---
def plot_tsne_embeddings(embeddings_dict, output_dir, filename_prefix="tsne_embeddings",
                         perplexity=30, n_iter=1000, random_state=42,
                         extra_info_df=None, id_col=None, label_col=None, node_type_to_plot=None):
    """
    Gera e salva plots t-SNE para embeddings de nós.

    Args:
        embeddings_dict (dict): Dicionário {node_type: embeddings_tensor}.
        output_dir (str): Diretório para salvar os plots.
        filename_prefix (str): Prefixo para os nomes dos ficheiros dos plots.
        perplexity (int): Parâmetro perplexity do t-SNE.
        n_iter (int): Número de iterações do t-SNE.
        random_state (int): Seed para reprodutibilidade do t-SNE.
        extra_info_df (pd.DataFrame, optional): DataFrame com informação adicional
                                                (ex: is_buyer) indexado pelo ID original.
        id_col (str, optional): Nome da coluna de ID no extra_info_df.
        label_col (str, optional): Nome da coluna no extra_info_df para usar como label de cor.
        node_type_to_plot (str, optional): Se especificado, plota apenas este tipo de nó.
    """
    logging.info(f"Generating t-SNE plots (perplexity={perplexity}, n_iter={n_iter})...")
    os.makedirs(output_dir, exist_ok=True)

    node_types = list(embeddings_dict.keys())
    all_embeddings = []
    all_labels = [] # Para colorir por tipo de nó
    node_type_indices = {} # Guardar início/fim de cada tipo
    start_idx = 0
    for node_type in node_types:
        if node_type_to_plot and node_type != node_type_to_plot:
            continue
        embeddings = embeddings_dict[node_type].cpu().numpy()
        num_nodes = embeddings.shape[0]
        all_embeddings.append(embeddings)
        all_labels.extend([node_type] * num_nodes)
        node_type_indices[node_type] = (start_idx, start_idx + num_nodes)
        start_idx += num_nodes

    if not all_embeddings:
        logging.warning("No embeddings found to plot.")
        return

    combined_embeddings = np.concatenate(all_embeddings, axis=0)

    # Verificar se há dados suficientes para t-SNE
    if combined_embeddings.shape[0] < perplexity:
        logging.warning(f"Number of samples ({combined_embeddings.shape[0]}) is less than perplexity ({perplexity}). Adjusting perplexity.")
        perplexity = max(5, combined_embeddings.shape[0] // 2 -1) # Ajuste heurístico
        if perplexity < 5:
             logging.error("Not enough samples for t-SNE plot.")
             return

    logging.info(f"Running t-SNE on {combined_embeddings.shape[0]} samples with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    logging.info("t-SNE transformation complete.")

    # --- Plot 1: Colorido por Tipo de Nó ---
    try:
        plt.figure(figsize=(10, 8))
        unique_node_types = sorted(list(set(all_labels)))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_node_types)))
        color_map = {ntype: color for ntype, color in zip(unique_node_types, colors)}

        for i, label in enumerate(all_labels):
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color_map[label], alpha=0.5, s=10) # s=10 para pontos menores

        # Criar legendas
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=ntype,
                              markerfacecolor=color_map[ntype], markersize=10) for ntype in unique_node_types]
        plt.legend(handles=handles, title="Node Types")

        plt.title(f'{filename_prefix} - Colored by Node Type')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        filepath = os.path.join(output_dir, f"{filename_prefix}_by_type.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        logging.info(f"  Plot saved to: {filepath}")
    except Exception as e:
        logging.error(f"  Error generating t-SNE plot by node type: {e}")

    # --- Plot 2: Colorido por Label Extra (se fornecido) ---
    if extra_info_df is not None and id_col is not None and label_col is not None and node_type_to_plot is not None:
        if node_type_to_plot not in node_type_indices:
             logging.warning(f"Node type '{node_type_to_plot}' not found in plotted embeddings for extra label plot.")
             return
        if label_col not in extra_info_df.columns:
             logging.warning(f"Label column '{label_col}' not found in extra_info_df. Skipping plot by label.")
             return
        if id_col not in extra_info_df.columns:
             logging.warning(f"ID column '{id_col}' not found in extra_info_df. Skipping plot by label.")
             return

        try:
            start, end = node_type_indices[node_type_to_plot]
            embeddings_to_label = embeddings_2d[start:end]
            # Precisamos mapear os índices dos embeddings de volta para os IDs originais
            # Assumindo que a ordem dos embeddings corresponde à ordem do id_map original
            # (Isto pode ser frágil se a ordem mudar)
            # Uma abordagem mais segura seria passar o id_map
            # Por agora, vamos assumir que extra_info_df está ordenado da mesma forma que os nós foram carregados
            # ou que o índice do DataFrame corresponde ao índice do nó (0 a N-1)

            # Tentativa de mapear pelo índice se o DF tiver o mesmo número de linhas que embeddings
            if len(extra_info_df) == embeddings_dict[node_type_to_plot].shape[0]:
                 labels_for_plot = extra_info_df[label_col].values
                 logging.info(f"Using label column '{label_col}' based on DataFrame index matching.")
            else:
                 # Tentar mapear por ID (mais seguro se id_col estiver presente)
                 # Precisamos do id_map para isso - esta função não o tem.
                 # Vamos simplificar por agora e assumir que o índice do DF corresponde
                 logging.warning(f"Length mismatch between extra_info_df ({len(extra_info_df)}) and embeddings ({embeddings_dict[node_type_to_plot].shape[0]}). Attempting direct indexing, might be incorrect.")
                 if end - start > len(extra_info_df):
                      logging.error("Cannot map labels, extra_info_df is too small.")
                      return
                 labels_for_plot = extra_info_df[label_col].iloc[:end-start].values


            plt.figure(figsize=(10, 8))
            unique_labels = sorted(pd.unique(labels_for_plot))
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: color for label, color in zip(unique_labels, colors)}

            for i in range(embeddings_to_label.shape[0]):
                label = labels_for_plot[i]
                plt.scatter(embeddings_to_label[i, 0], embeddings_to_label[i, 1],
                            color=color_map.get(label, 'gray'), # Cor cinza para labels não mapeados
                            alpha=0.6, s=10)

            handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color_map.get(label, 'gray'), markersize=10)
                       for label in unique_labels]
            plt.legend(handles=handles, title=label_col)

            plt.title(f'{filename_prefix} - {node_type_to_plot} Colored by {label_col}')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            filepath = os.path.join(output_dir, f"{filename_prefix}_{node_type_to_plot}_by_{label_col}.png")
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            logging.info(f"  Plot saved to: {filepath}")

        except Exception as e:
            logging.error(f"  Error generating t-SNE plot by label '{label_col}': {e}", exc_info=True)

# --- FIM ALTERAÇÃO ---