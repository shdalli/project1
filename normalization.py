import pandas as pd
import mysql.connector
import re
import itertools


# Load the dataset
data = input("Enter file name along with Path: ")

df = pd.read_csv(data)
df = df.convert_dtypes()

def check_and_fix_1nf(df):
    non_atomic_columns = []

    # Detect non-atomic columns (e.g., string with commas)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, str) and ',' in x).any():
            non_atomic_columns.append(col)

    if not non_atomic_columns:
        print("Checked for 1NF: Table is in First Normal Form")
        return df

    print("Table is not in 1NF. Fixing...")
    print(f"Non-atomic columns found: {non_atomic_columns}")

    # Normalize table: explode the non-atomic columns
    for col in non_atomic_columns:
        df[col] = df[col].apply(lambda x: x.split(',') if isinstance(x, str) else [x])
        df = df.explode(col).reset_index(drop=True)

    print("Transformed to 1NF")
    return df

df = check_and_fix_1nf(df)
print(df)

for col in df.columns:
    if df[col].dtype == 'Int64':
        df[col] = df[col].astype(str)
        

print(df.to_string())

print('Printing Number of rows and Columns')
print(df.shape)

print('Sample records from Dataframe')
print(df.head(10))

print('Attribute Data types')
print(df.info())


# Get table names and primary keys
table_name1 = input("Enter 1st relation name: ")
table_name2 = input("Enter 2nd relation name: ")

table_name1 = 'Employee'
table_name2 = 'Department'

# Capture multiple primary keys for Table 1
table1_pk = []
print("Enter Primary Keys for Table 1 (Press Enter to stop):")
while True:
    pk = input().strip()
    if pk == "":  # Stop when user presses Enter without input
        break
    table1_pk.append(pk)

# Capture multiple primary keys for Table 2
table2_pk = []
print("Enter Primary Keys for Table 2 (Press Enter to stop):")
while True:
    pk = input().strip()
    if pk == "":
        break
    table2_pk.append(pk)
    
# Convert lists of primary keys to frozensets for immutability and hashability
table1_pk = frozenset(table1_pk)
table2_pk = frozenset(table2_pk)

# Store attributes for each table
table1_attributes = set(table1_pk)
table2_attributes = set(table2_pk)

# Collect functional dependencies from user
fds = []  # List of (LHS, RHS) pairs

for col in df.columns:
    if col in table1_pk or col in table2_pk:
        continue  # Skip primary keys

    print(f"Does {col} depend on {table1_pk}? (Yes/No)")
    response = input().strip().lower()
    
    if response == "yes":
        fds.append((table1_pk, col))  # Store LHS as a frozenset
        table1_attributes.add(col)
    else:
        print(f"Does {col} depend on {table2_pk}? (Yes/No)")
        response2 = input().strip().lower()
        if response2 == "yes":
            fds.append((table2_pk, col))  # Store LHS as a frozenset
            table2_attributes.add(col)

table1_attr_list = list(table1_attributes)
table2_attr_list = list(table2_attributes)


# Display the collected functional dependencies
print("\nFunctional Dependencies (FDs):")
for lhs, rhs in fds:
    print(f"{', '.join(lhs)} → {rhs}")

# Function to compute attribute closure
def compute_closure(attributes, fds):
    closure = set(attributes)
    changed = True

    while changed:
        changed = False
        for lhs, rhs in fds:
            if lhs.issubset(closure) and rhs not in closure:
                closure.add(rhs)
                changed = True

    return closure

# Compute and display closures
print("\nComputing Attribute Closures:")

closures = {}
for lhs, _ in fds:
    closure = compute_closure(lhs, fds)
    closures[frozenset(lhs)] = closure
    print(f"Closure of {', '.join(lhs)}⁺: {closure}")

# Detect Partial Dependencies
print("\nDetecting Partial Dependencies:")

partial_dependencies = []
print('fds',fds)

partial_dependencies_flag = 'No'

for lhs, rhs in fds:
    #print('------------------')
    #print('len(lhs)', len(lhs))
    #print('lhs', lhs)
    #print('rhs', rhs)
    if len(lhs) > 1:  # Check only for composite keys
        for attr in lhs:
            #print('attr',attr)
            reduced_lhs = lhs - {attr}  # Remove one attribute at a time
            #print('reduced_lhs', reduced_lhs)
            closure_reduced = compute_closure(reduced_lhs, fds)
            #print('closure_reduced', closure_reduced)
            if rhs in closure_reduced:
                print(f"Partial Dependency Detected: {', '.join(lhs)} → {rhs}, since {', '.join(reduced_lhs)} → {rhs} also holds.")
                partial_dependencies.append((lhs, rhs))
                partial_dependencies_flag = 'Yes'

if partial_dependencies_flag == 'Yes':
    print('Partial Dependencies Detected')
    print('partial_dependencies', partial_dependencies)
else: 
    print('NO Partial Dependencies Detected')

# Detect Transitive Dependencies
print("\nDetecting Transitive Dependencies:")

transitive_dependencies = []
transitive_dependencies_flag = 'No'

for lhs1, rhs1 in fds:
    #print('------------------')
    #print('lhs1 =', lhs1, 'rhs1 = ', rhs1)
    for lhs2, rhs2 in fds:
        #print('#############')
        #print('lhs2 =', lhs2, 'rhs2 = ', rhs2)
        if rhs1 == lhs2:  # If RHS of one FD is LHS of another, we have a potential transitive dependency
            transitive_dependencies.append((lhs1, rhs2))
            print(f"Transitive Dependency Detected: {', '.join(lhs1)} → {rhs2}, via {rhs1}")
            transitive_dependencies_flag = 'Yes'
            
if transitive_dependencies_flag == 'Yes':
    print('Transitive Dependencies Detected')
    print('transitive_dependencies', transitive_dependencies)
else: 
    print('NO Transitive Dependencies Detected')

# Check if an FD holds
fd_check = input("\nEnter an FD to verify (Format: A -> B): ")
lhs, rhs = map(str.strip, fd_check.split("->"))

closure_lhs = compute_closure({lhs}, fds)  # Convert lhs to a set
if rhs in closure_lhs:
    print(f"The FD {lhs} → {rhs} is VALID.")
else:
    print(f"The FD {lhs} → {rhs} is NOT VALID.")

# Create dataframe (Table1)
table1_df = df[table1_attr_list].drop_duplicates()
print(table1_df.info())

# Create dataframe (Table2)
table2_df = df[table2_attr_list].drop_duplicates()
print(table2_df.info())

# Now, let's display the tables that follow 3NF
print("\nTable 1 (3NF):")
print(table1_df)

print("\nTable 2 (3NF):")
print(table2_df)


def find_candidate_keys(all_attributes, fds):
    """
    Finds all minimal candidate keys for a given set of attributes and functional dependencies.
    """
    candidate_keys = []
    attr_list = list(all_attributes)

    # Try all attribute combinations, from smallest to largest
    for r in range(1, len(attr_list) + 1):
        for subset in itertools.combinations(attr_list, r):
            closure = compute_closure(subset, fds)
            if closure == all_attributes:
                # Check minimality
                if not any(set(ck).issubset(set(subset)) for ck in candidate_keys):
                    candidate_keys.append(subset)

    return candidate_keys

all_attributes = set(df.columns)
candidate_keys = find_candidate_keys(all_attributes, fds)

print("\nCandidate Keys:")
for ck in candidate_keys:
    print(", ".join(ck))

def bcnf_decomposition(df, fds):
    """
    Perform BCNF decomposition on the given dataframe using the functional dependencies.
    Returns a list of decomposed relations \and their corresponding attributes.
    """
    def is_superkey(attributes, fds, all_attributes):
        return compute_closure(attributes, fds) == all_attributes

    relations = [(df.copy(), set(df.columns))]  # Start with the original relation
    decomposed = []

    while relations:
        relation_df, attributes = relations.pop()
        print(f"\nChecking relation with attributes: {attributes}")
        
        # Filter FDs relevant to this relation
        relevant_fds = [(lhs, rhs) for lhs, rhs in fds if rhs in attributes and lhs.issubset(attributes)]
        
        # Find a violating FD (where LHS is not a superkey)
        for lhs, rhs in relevant_fds:
            if not is_superkey(lhs, relevant_fds, attributes):
                print(f"Violation Found: {lhs} → {rhs} is not allowed in BCNF.")
                
                # Create two new relations
                closure = compute_closure(lhs, relevant_fds)
                r1_attrs = closure
                r2_attrs = attributes - (closure - lhs)

                df1 = relation_df[list(r1_attrs)].drop_duplicates()
                df2 = relation_df[list(r2_attrs)].drop_duplicates()

                print(f"Decomposing into:")
                print(f" - R1: {r1_attrs}")
                print(f" - R2: {r2_attrs}")

                relations.append((df1, r1_attrs))
                relations.append((df2, r2_attrs))
                break  # Only decompose one FD at a time
        else:
            # No violation found, keep this relation
            decomposed.append((relation_df, attributes))

    return decomposed
# Decompose to BCNF
print("\nPerforming BCNF Decomposition:")
decomposed_relations = bcnf_decomposition(df, fds)

# Display results
for i, (rel_df, attrs) in enumerate(decomposed_relations, 1):
    print(f"\nBCNF Relation {i}: Attributes = {attrs}")
    print(rel_df.head())

#Preparing DDL Script 
def pandas_dataframe_to_ddl(df, table_name):
    """
    Generates a basic CREATE TABLE DDL statement from a Pandas DataFrame.
    Note: This is a simplified version and might require manual adjustments
          for specific SQL data types and constraints.
    """
    column_definitions = []
    for column, dtype in df.dtypes.items():
        sql_type = None
        if pd.api.types.is_integer_dtype(dtype):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "REAL"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sql_type = "TIMESTAMP"
        elif dtype == 'object':
            # Default to TEXT, might need adjustment based on actual data
            sql_type = "TEXT"
        elif pd.api.types.is_bool_dtype(dtype):
            sql_type = "BOOLEAN"
        else:
            sql_type = "TEXT"  # Default for unknown types

        column_definitions.append(f"{column} {sql_type}")

    ddl_statement = f"CREATE TABLE {table_name} (\n"
    ddl_statement += ",\n".join(column_definitions)
    ddl_statement += "\n);"

    return ddl_statement


ddl1 = pandas_dataframe_to_ddl(table1_df, table_name1)
ddl2 = pandas_dataframe_to_ddl(table2_df, table_name2)

print(ddl1)
print(ddl2)

def execute_ddl_script(host, user, password, database, ddl_script, table_name):
    """
    Connects to a MySQL server and executes the DDL statements from a file.

    Args:
        host (str): The hostname or IP address of the MySQL server.
        user (str): The MySQL username.
        password (str): The MySQL password.
        database (str): The name of the database to use (optional, can be None).
        ddl_file_path (str): The path to the file containing the DDL script.
    """
    try:
        # Establish connection to MySQL server
        mydb = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database  # Specify database if needed
        )

        mycursor = mydb.cursor()

        # Split the script into individual statements (assuming statements end with ';')
        statements = ddl_script.split(';')
        drop_stmt  = 'DROP TABLE IF EXISTS ' + table_name + ';'

        for statement in statements:
            statement = statement.strip()
            if statement:  # Execute non-empty statements
                mycursor.execute(drop_stmt);
                mydb.commit()  # Commit each successful statement
                print(f"DDL script from executed successfully.")
                
                mycursor.execute(statement)
                mydb.commit()  # Commit each successful statement
                print(f"DDL script from executed successfully.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if mydb.is_connected():
            mycursor.close()
            mydb.close()
            print("MySQL connection closed.")
            
def insert_into_table(df, table_name, db_host, db_user, db_password, db_name):
    try:
        mydb = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )

        mycursor = mydb.cursor()

        # Prepare the SQL INSERT statement
        columns = ', '.join(df.columns)
        
        placeholders = ', '.join(['%s'] * len(df.columns))
        
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        print(sql)

        # Execute the INSERT statement for each row in the DataFrame
        for row in df.itertuples(index=False):
            print('starting')
            mycursor.execute(sql, tuple(row))
            print('end')

        # Commit the changes
        mydb.commit()
        print(f"{mycursor.rowcount} rows inserted successfully into table '{table_name}'.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if mydb.is_connected():
            mycursor.close()
            mydb.close()
            print("MySQL connection closed.")
        
if __name__ == "__main__":
    # Replace with your MySQL connection details and DDL file path
    mysql_host = "localhost"  # Or your MySQL server address
    mysql_user = "root"
    mysql_password = "Sai149390!030"
    mysql_database = "test"  # Optional, can be None

    execute_ddl_script(mysql_host, mysql_user, mysql_password, mysql_database, ddl1, table_name1)
    execute_ddl_script(mysql_host, mysql_user, mysql_password, mysql_database, ddl2, table_name2)

    insert_into_table(table2_df, table_name2, mysql_host, mysql_user, mysql_password, mysql_database)
    
    insert_into_table(table1_df, table_name1, mysql_host, mysql_user, mysql_password, mysql_database)

    
