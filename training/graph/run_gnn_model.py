# /training/graph/run_gnn_model.py

import argparse
import subprocess
import sys
import os
import json

def run_training(args):
    """
    Constructs and executes the command to run train.py.

    Args:
        args: An argparse Namespace containing arguments for train.py.
    """
    command = [
        sys.executable,
        "-m",
        "training.graph.train"
    ]

    # Adiciona os argumentos do train.py à lista de comando
    command.extend(['--data-dir', args.data_dir])
    command.extend(['--output-dir', args.output_dir])
    command.extend(['--hidden-channels', str(args.hidden_channels)])
    command.extend(['--num-gnn-layers', str(args.num_gnn_layers)])
    command.extend(['--dropout', str(args.dropout)])
    # --- ALTERAÇÃO: Adicionar gat-heads ao comando ---
    command.extend(['--gat-heads', str(args.gat_heads)])
    # --- FIM ALTERAÇÃO ---
    command.extend(['--node-emb-dim', str(args.node_emb_dim)])
    command.extend(['--link-pred-edge', args.link_pred_edge])
    command.extend(['--lr', str(args.lr)])
    # --- ALTERAÇÃO: Adicionar weight-decay ao comando ---
    command.extend(['--weight-decay', str(args.weight_decay)])
    # --- FIM ALTERAÇÃO ---
    command.extend(['--epochs', str(args.epochs)])
    command.extend(['--batch-size', str(args.batch_size)])
    num_neighbors_str = ",".join(map(str, args.num_neighbors))
    command.extend(['--num-neighbors', num_neighbors_str])
    command.extend(['--neg-sampling-ratio', str(args.neg_sampling_ratio)])
    command.extend(['--device', args.device])
    command.extend(['--seed', str(args.seed)])

    print("--- Running Training Script ---")
    print(f"Command: {' '.join(command)}")

    try:
        process = subprocess.run(command, check=True, text=True, capture_output=False)
        print("\n--- Training Script Finished Successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"\n--- Training Script Failed ---")
        print(f"Return Code: {e.returncode}")
        # --- ALTERAÇÃO: Imprimir stdout e stderr em caso de erro ---
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        # --- FIM ALTERAÇÃO ---
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Could not find the python executable '{sys.executable}' or the train module.")
        print("Ensure you are running from the correct directory and your environment is set up.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while trying to run train.py: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator for GNN Training")

    # --- Argumentos Principais ---
    parser.add_argument('--shop-id', type=str, required=True,
                        help="Shop ID to determine data directory.")
    parser.add_argument('--results-base-dir', type=str, default='./results',
                        help="Base directory where shop-specific results are stored.")
    parser.add_argument('--config-file', type=str, default=None,
                        help="Optional: Path to a JSON config file with training hyperparameters.")
    parser.add_argument('--tag', type=str, default='default_run',
                        help="A tag for this specific training run (used for output subdirectory).")

    # --- Permite overrides de linha de comando (opcional) ---
    # Adiciona TODOS os argumentos que podem ser sobrescritos
    parser.add_argument('--epochs', type=int, help="Override number of training epochs.")
    parser.add_argument('--lr', type=float, help="Override learning rate.")
    # --- ALTERAÇÃO: Adicionar weight-decay aqui ---
    parser.add_argument('--weight-decay', type=float, help="Override weight decay.")
    # --- FIM ALTERAÇÃO ---
    parser.add_argument('--hidden-channels', type=int, help="Override hidden channels.")
    parser.add_argument('--device', type=str, help="Override device (cpu/cuda/auto).")
    parser.add_argument('--link-pred-edge', type=str, help="Override target edge type for prediction.")
    parser.add_argument('--node-emb-dim', type=int, help="Override target node embedding dimension.")
    parser.add_argument('--num-gnn-layers', type=int, help="Override number of GNN layers.")
    # --- ALTERAÇÃO: Adicionar gat-heads aqui ---
    parser.add_argument('--gat-heads', type=int, help="Override number of GAT heads.")
    # --- FIM ALTERAÇÃO ---
    parser.add_argument('--num-neighbors', type=str, help="Override number of neighbors per layer (comma-separated string).")
    parser.add_argument('--dropout', type=float, help="Override dropout rate.")
    parser.add_argument('--batch-size', type=int, help="Override batch size.")
    parser.add_argument('--neg-sampling-ratio', type=float, help="Override negative sampling ratio.")
    parser.add_argument('--seed', type=int, help="Override random seed.")

    run_args = parser.parse_args()

    # --- Determinar Diretórios ---
    gnn_data_dir = os.path.join(run_args.results_base_dir, run_args.shop_id, 'gnn_data')
    output_dir = os.path.join(run_args.results_base_dir, run_args.shop_id, 'training_output', run_args.tag)

    # --- Carregar Configuração Base ---
    default_train_args = {
        'data_dir': gnn_data_dir,
        'output_dir': output_dir,
        'hidden_channels': 128,
        'num_gnn_layers': 2,
        'dropout': 0.3,
        'gat_heads': 1, # Default GAT heads
        'node_emb_dim': 128,
        'link_pred_edge': 'purchased',
        'lr': 0.001,
        'weight_decay': 0.0, # Default weight decay
        'epochs': 50,
        'batch_size': 512,
        'num_neighbors': [15, 10],
        'neg_sampling_ratio': 1.0,
        'device': 'auto',
        'seed': 42
    }

    # Carrega de ficheiro config se fornecido
    if run_args.config_file:
        print(f"Loading configuration from file: {run_args.config_file}")
        try:
            with open(run_args.config_file, 'r') as f:
                config_from_file = json.load(f)
            if 'num_neighbors' in config_from_file and isinstance(config_from_file['num_neighbors'], str):
                try:
                    config_from_file['num_neighbors'] = [int(n) for n in config_from_file['num_neighbors'].split(',')]
                except ValueError:
                     print(f"Warning: Invalid 'num_neighbors' format in config file: {config_from_file['num_neighbors']}. Ignoring this setting from file.")
                     del config_from_file['num_neighbors']
            default_train_args.update(config_from_file)
            print("Configuration loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: Config file '{run_args.config_file}' not found. Using defaults/overrides.")
        except json.JSONDecodeError:
             print(f"Warning: Error decoding JSON from '{run_args.config_file}'. Using defaults/overrides.")
        except Exception as e:
             print(f"Warning: Error loading config file '{run_args.config_file}': {e}. Using defaults/overrides.")

    # Sobrepõe com argumentos de linha de comando (agora inclui gat_heads e weight_decay)
    override_args = {k: v for k, v in vars(run_args).items() if v is not None and k in default_train_args}
    # Processa num_neighbors do override se existir
    if 'num_neighbors' in override_args and isinstance(override_args['num_neighbors'], str):
         try:
             override_args['num_neighbors'] = [int(n) for n in override_args['num_neighbors'].split(',')]
             num_layers_for_validation = override_args.get('num_gnn_layers', default_train_args['num_gnn_layers'])
             if len(override_args['num_neighbors']) != num_layers_for_validation:
                 print(f"Error: Override --num-neighbors ({override_args['num_neighbors']}) length doesn't match effective --num-gnn-layers ({num_layers_for_validation}).")
                 sys.exit(1)
         except ValueError:
              print(f"Error: Invalid format for override --num-neighbors: {override_args['num_neighbors']}. Must be comma-separated integers.")
              sys.exit(1)
    elif 'num_neighbors' in override_args and not isinstance(override_args['num_neighbors'], list):
          num_layers_for_validation = override_args.get('num_gnn_layers', default_train_args['num_gnn_layers'])
          if len(override_args['num_neighbors']) != num_layers_for_validation:
               print(f"Error: Config/Default --num-neighbors ({override_args['num_neighbors']}) length doesn't match effective --num-gnn-layers ({num_layers_for_validation}).")
               sys.exit(1)

    default_train_args.update(override_args)

    # --- VALIDAÇÃO FINAL: num_neighbors vs num_gnn_layers ---
    if len(default_train_args['num_neighbors']) != default_train_args['num_gnn_layers']:
         print(f"Final Configuration Error: Length of 'num_neighbors' ({default_train_args['num_neighbors']}) "
               f"does not match 'num_gnn_layers' ({default_train_args['num_gnn_layers']}).")
         sys.exit(1)

    # Converte o dicionário final de volta para um Namespace
    final_train_args = argparse.Namespace(**default_train_args)

    # --- Executar o Treino ---
    run_training(final_train_args)

    print(f"\n--- GNN Model Runner Finished ---")
    print(f"Check output directory for results: {final_train_args.output_dir}")