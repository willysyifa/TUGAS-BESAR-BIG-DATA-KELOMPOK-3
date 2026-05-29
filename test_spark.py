from pyspark.sql import SparkSession

# Membuat Spark Session
spark = SparkSession.builder \
    .appName("BigDataGempa") \
    .getOrCreate()

# Sample data
data = [
    ("Sumatera", 5.2),
    ("Jawa", 6.1),
    ("Sulawesi", 4.9)
]

# Membuat DataFrame
df = spark.createDataFrame(data, ["Wilayah", "Magnitudo"])

# Menampilkan data
df.show()

# Stop Spark
spark.stop()