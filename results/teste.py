import pandas as pd
import os

# Definir o caminho do ficheiro diretamente no script
csv_file = r"C:\Users\mike_\Documents\NappsRecommenderSystem\results\all_recommendations.csv"

def count_above_threshold(csv_file):
    # Verifica se o ficheiro existe
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return
    
    try:
        # Lê o CSV para um DataFrame
        df = pd.read_csv(csv_file)

        # Debug: Mostra as primeiras linhas do CSV
        print("Preview do ficheiro:")
        print(df.head())

        # Verifica se a coluna 'similarity_score' existe
        if 'similarity_score' not in df.columns:
            print("Error: CSV file does not contain 'similarity_score' column.")
            return

        # Conta os valores acima de 0.05
        count = (df['similarity_score'] > 0.05).sum()
        print(f"Number of values above 0.05 in similarity_score: {count}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Chamar a função diretamente
count_above_threshold(csv_file)
