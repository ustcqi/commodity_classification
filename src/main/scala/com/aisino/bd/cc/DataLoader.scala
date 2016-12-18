package com.aisino.bd.cc

import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  * Created by kerwin on 16/9/20.
  */
class DataLoader(context: AppContext){
    val spark = context.spark
    val sqlContext = context.sqlContext

    def getMcBmDF(sql: String): DataFrame ={
        //val df = spark.sql("select hwmc, spbm from dzdz_an.dzdz_hwxx_ptfp where spbm is not null")
        val df = spark.sql(sql)
        df
    }

    def getMcBmChars(df: DataFrame) : DataFrame = {
        val sentence2unichars = udf { t: String => t.toArray.map(_.toString) }
        val bmMcUnicharsDF = df.select(col("*"), sentence2unichars(df("hwmc")).as("hwmc_unichars"))
        bmMcUnicharsDF
    }

    def getNgramCharsDF(bmMcUnicharsDF: DataFrame) : DataFrame = {
        val ngram = new NGram().setInputCol("hwmc_unichars").setOutputCol("ngrams").setN(2)
        val ngramCharsDF = ngram.transform(bmMcUnicharsDF)
        ngramCharsDF
    }

    def getData(sql: String) : DataFrame = {
        //val sql = "select hwmc, spbm from dzdz_an.dzdz_hwxx_ptfp where spbm is not null"
        val bmMcUnicharsDF = getMcBmDF(sql)
        val df = getMcBmChars(bmMcUnicharsDF)
        val dfc_ngram = getNgramCharsDF(df)
        val listcomb = udf { (t1: Seq[String], t2: Seq[String]) => t1 ++ t2 }
        val dfc_ngram_f = dfc_ngram.select(col("spbm"), listcomb(col("hwmc_unichars"), col("ngrams")).as("hwmc_unichars_and_ngrams"))
        dfc_ngram_f
    }
}