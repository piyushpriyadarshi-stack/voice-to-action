import sqlite3

DATABASE = "voice2action.db"

connection = sqlite3.connect(DATABASE)
cursor = connection.cursor()

print("Checking database...")

# Check meetings table
cursor.execute("PRAGMA table_info(meetings)")
columns = cursor.fetchall()

column_names = [column[1] for column in columns]

print("Existing columns:")
print(column_names)

# Add user_id if missing
if "user_id" not in column_names:

    print("Adding user_id column...")

    cursor.execute("""
        ALTER TABLE meetings
        ADD COLUMN user_id INTEGER
    """)

    connection.commit()

    print("✅ user_id column added successfully.")

else:

    print("✅ user_id already exists.")


# Check users table
cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    AND name='users'
""")

users_table = cursor.fetchone()

if users_table:

    print("✅ Users table exists.")

    cursor.execute("""
        SELECT id, username
        FROM users
    """)

    users = cursor.fetchall()

    print("Users:")

    for user in users:
        print(user)

else:

    print("❌ Users table does not exist.")


connection.close()

print("\nMigration completed.")