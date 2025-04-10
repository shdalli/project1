# Project 1: Normalization Tool

#### Project Description
This project is a Python tool that takes a csv file from the user and reads it into a Pandas DataFrame, the tool asks users for functional dependencies, primary keys, and computes closure of attribute sets, detects partial and transitive dependencies, and suggests canditate keys. It verifies if the dataset satisfies 1NF, 2NF, 3NF, and BCNF, and creates normalized tables and populates them in MySQL database and provides an interactive query interface where the user can perform basic operations and run custom SQL queries.

#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shdalli/project1.git
   cd project1
#### Usage Instructions
1. Execute normalization.py on the command line.
2. The system will prompt for a file to be uploaded, provide csv file name along with file path.
3. System will prompt user to enter table names, primary key, and functional dependencies.
4. System will compute functional dependencies, closures, partial dependencies, and transitive dependencies. In case of issues, the system will detect partial dependencies or transitive dependencies.
5. System will create new data frames in alignment with 3NF.
6. System will create data definition language(DDL) scripts for new relations. It will apply DDL script programmatically to MySQL server database and insert data into newly created tables programmatically. 
