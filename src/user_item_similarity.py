import pyspark as ps
import numpy as np

spark = (
        ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("lecture")
        .getOrCreate()
        )

sc = spark.sparkContext

item_train_data = spark.read.json('../data/rec_input/gl_item_train_data1')
user_train_data = spark.read.json('../data/rec_input/gl_user_train_data1')
item_test_data = spark.read.json('../data/rec_input/gl_item_test_data1')
user_test_data = spark.read.json('../data/rec_input/gl_user_test_data1')

nor = ps.mlib.feature.Normalizer(1)
user_test_data_norm = nor.transform(user_test_data)

def rddTranspose(rdd):
    rddT1 = rdd.zipWithIndex()
            .flatMap(lambda (x,i): [(i,j,e) for (j,e) in enumerate(x)])
    rddT2 = rddT1.map(lambda (i,j,e): (j, (i,e)))
            .groupByKey().sortByKey()
    rddT3 = rddT2.map(lambda (i, x): sorted(list(x),
                        cmp=lambda (i1,e1),(i2,e2) : cmp(i1, i2)))
    rddT4 = rddT3.map(lambda x: map(lambda (i, y): y , x))
    return rddT4.map(lambda x: np.asarray(x))


# aws s3 cp s3://tomasbielskis-galvanizebucket/capstone/data/rec_input/ data/rec_input/ --recursive
