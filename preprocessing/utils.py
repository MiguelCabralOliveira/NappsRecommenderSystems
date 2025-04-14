# preprocessing/utils.py
import pandas as pd
import json
import ast
import traceback

def safe_json_parse(json_like_str):
    """
    Tenta fazer o parse de uma string tipo JSON de forma segura.
    Retorna um dicionário Python em caso de sucesso, ou None em caso de falha.
    """
    if pd.isna(json_like_str) or not isinstance(json_like_str, str) or json_like_str.strip() == '{}' or json_like_str.strip() == '':
        return None
    try:
        # Tenta limpar e usar json.loads primeiro (mais standard)
        cleaned_str = json_like_str.replace("'", '"')
        # Lidar com None, True, False que não são standard JSON
        cleaned_str = cleaned_str.replace(': None', ': null').replace(', None', ', null').replace('{None', '{null')
        cleaned_str = cleaned_str.replace(': True', ': true').replace(', True', ', true').replace('{True', '{true')
        cleaned_str = cleaned_str.replace(': False', ': false').replace(', False', ', false').replace('{False', '{false')
        # Lidar com casos específicos problemáticos (adicionar mais se necessário)
        # Exemplo: Se houver floats inválidos como .0,
        # cleaned_str = re.sub(r':\s*\.(\d+)', r': 0.\1', cleaned_str)
        # cleaned_str = re.sub(r',\s*\.(\d+)', r', 0.\1', cleaned_str)
        return json.loads(cleaned_str)
    except (json.JSONDecodeError, TypeError):
        pass  # Ignora o erro e tenta o próximo método

    try:
        # Tenta usar ast.literal_eval como fallback (pode lidar com alguns formatos não-JSON)
        parsed_data = ast.literal_eval(json_like_str)
        # Garante que o resultado é um dicionário
        return parsed_data if isinstance(parsed_data, dict) else None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        # Se ambos falharem, retorna None
        # print(f"Falha ao fazer parse: {json_like_str[:100]}...") # Descomentar para debug
        return None