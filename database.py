
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import os

db_user = os.environ.get("DB_USER")  # Use environment variables!
db_password = os.environ.get("DB_PASSWORD")
db_host = '172.26.219.133' # os.environ.get("DB_HOST")
db_port = '5433'
db_name = 'optimizeai' #postgres in this case

print(db_host)

db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Database connection details (using a connection string format)
# db_url = "postgresql://postgres:Hmniaailc1804@localhost:5433/postgres"  # Replace with your actual password

try:
    engine = create_engine(db_url)  # Create the SQLAlchemy engine

    # Read data into a pandas DataFrame (using the engine)
    df = pd.read_sql("SELECT name, portfolio -> 'positions' AS positions FROM round, LATERAL jsonb_array_elements(portfolios) AS portfolio", engine)

    ata = df["positions"].tolist()

    # Save to a JSON file
    df.to_json("training_data/positions.json", orient="records", indent=4)

    print(df.head())

except Exception as e:  # Catch a broader range of exceptions
    print(f"Error: {e}")

finally:  # Ensure the engine is disposed of, even if there's an error
    if 'engine' in locals(): #check if the engine exists to avoid errors when it doesn't
        engine.dispose()







