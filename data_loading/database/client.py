# data_loading/database/client.py
import os
import traceback
from sqlalchemy import create_engine

# Este arquivo não importa outros módulos do pacote,
# então seus imports permanecem inalterados.

def get_db_engine(db_config: dict):
    """Creates and returns a SQLAlchemy engine based on config."""
    required_keys = ['host', 'user', 'password', 'name', 'port']
    # Allow empty password explicitly if needed, otherwise check all keys
    if not all(key in db_config and (db_config[key] or db_config[key]=='') for key in required_keys):
        missing = [key for key in required_keys if key not in db_config or db_config[key] is None] # Check for None or missing
        raise ValueError(f"Database configuration incomplete. Missing or None values for keys: {missing}")

    db_host = db_config['host']
    db_password = db_config['password']
    db_user = db_config['user']
    db_name = db_config['name']
    db_port = db_config['port']

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    try:
        # Adicionar pool_pre_ping para verificar conexões antes de usar
        engine = create_engine(connection_string, pool_pre_ping=True)
        # Test connection briefly
        with engine.connect() as connection:
            print("Database connection successful.")
        return engine
    except Exception as e:
        print(f"Error creating database engine: {e}")
        traceback.print_exc()
        raise # Re-lança a exceção para ser capturada pelo chamador