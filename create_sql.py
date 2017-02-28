import mysql.connector
db1 = mysql.connector.connect(host="localhost", user="root", passwd="12345")
cursor = db1.cursor()
sql = 'CREATE DATABASE tfidf_data'
cursor.execute(sql)
