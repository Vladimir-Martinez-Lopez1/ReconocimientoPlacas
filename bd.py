import sqlite3

# Conexión a la base de datos (se crea si no existe)
def create_database(db_name='baseDatos.db'):
    conn = sqlite3.connect(db_name)

    # Crear un cursor para ejecutar consultas
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS placas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placas TEXT,
            tipo_placa TEXT,
            frame TEXT,
            fecha timestamp DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()  
    return conn

def insert_data(conn, placas, tipo_placa, frame):
    cursor = conn.cursor()

    # Insertar los datos en la tabla
    cursor.execute('''
        INSERT INTO placas (placas, tipo_placa, frame)
        VALUES (?, ?, ?)
    ''', (placas, tipo_placa, frame))

    conn.commit()

def get_data(conn):
    cursor = conn.cursor()

    # Obtener todos los registros de la tabla
    cursor.execute('SELECT * FROM placas')
    rows = cursor.fetchall()

    return rows
