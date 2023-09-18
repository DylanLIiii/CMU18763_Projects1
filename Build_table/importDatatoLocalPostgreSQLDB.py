import psycopg2
import pandas as pd


def get_csv_file_paths(directory):
  csv_file_paths = []
  for root, directories, files in os.walk(directory):
    for file in files:
      if os.path.isfile(os.path.join(root, file)):
        file_name, file_extension = os.path.splitext(file)
        if file_extension == '.csv':
          csv_file_paths.append(os.path.join(root, file))
  return csv_file_paths

def add_year(file_path): 
    df = pd.read_csv(file_path)
    # output final year
    
    df['year'] = int(f'20{file_path[-6:-4]}')
    df.to_csv(file_path, index=False)

file_paths = get_csv_file_paths('/home/dylan/dylan_repo/CMU18763_Projects1/fifadata')

# connect to postgreSQL database

conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="Wuhezhizhong2020.",
    port="5432"
    )

cur = conn.cursor()

# table config
table_name = 'fifa'
headers = pd.read_csv('/home/dylan/dylan_repo/CMU18763_Projects1/fifadata/players_15.csv').columns


cur.execute(f"DROP TABLE IF EXISTS {table_name};")
cur.execute(f"CREATE TABLE {table_name} " + "(" + ",".join([f"{header} VARCHAR" for header in headers])+ ", id SERIAL PRIMARY KEY" + ");")

for file_name in file_paths: 
    add_year(file_name)
    with open(file_name, 'r') as f:
        copy_sql = f"""
                COPY {table_name}({', '.join(headers)})
                FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')
            """
        next(f)
        cur.copy_expert(sql=copy_sql, file=f)
        conn.commit()
conn.close()