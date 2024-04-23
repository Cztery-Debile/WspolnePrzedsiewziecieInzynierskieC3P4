
import psycopg2

# Connect to the database - replace the SERVICE URI with the actual service URI
conn = psycopg2.connect(
    "postgres://avnadmin:AVNS_bfFc4wTNrJeCe8hQ3Yo@pg-29ab6d03-facesxd.a.aivencloud.com:12523/defaultdb?sslmode=require")


def delete_record(record_id):
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM pictures WHERE id = %s", (record_id,))
        conn.commit()
        print("Record with id {} deleted successfully.".format(record_id))
    except psycopg2.Error as e:
        print("Error deleting record:", e)

# Function to show all records and their ids
def show_records():
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, nazwa,times,direction FROM pictures")
        records = cur.fetchall()
        print("Available records to delete:")
        for record in records:
            print("ID: {}, Name: {}, Czas dodania: {}, Kierunek: {}".format(record[0], record[1], record[2], record[3]))
    except psycopg2.Error as e:
        print("Error fetching records:", e)

# Ask user if they want to see available records
show_available_records = input("Do you want to see available records to delete? (yes/no): ")
if show_available_records.lower() == 'yes':
    show_records()

# Ask user for the record id to delete
record_id_to_delete = input("Enter the ID of the record you want to delete: ")

# Delete the specified record
delete_record(record_id_to_delete)
