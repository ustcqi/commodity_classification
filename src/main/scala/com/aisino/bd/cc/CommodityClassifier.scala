package com.aisino.bd.cc

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

class CommodityClassifier(context: AppContext){



    def train(dataDF: DataFrame): Unit = {
        val cvModel: CountVectorizerModel = new CountVectorizer()
                .setInputCol("hwmc_unichars_and_ngrams")
                .setOutputCol("hwmc_f")
                .setVocabSize(5000)
                .setMinDF(2)
                .fit(dataDF)
        val dfi = cvModel.transform(dataDF)

        val labelIndexer = new StringIndexer()
                .setInputCol("spbm")
                .setOutputCol("label")
                .fit(dfi)

        val featureIndexer = new VectorIndexer()
                .setInputCol("hwmc_f")
                .setOutputCol("hwmc_i")
                .setMaxCategories(4)
                .fit(dfi)

        val Array(trainingData, testData) = dfi.randomSplit(Array(0.7, 0.3))
        //val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("hwmc_i").setNumTrees(100)
        //val gbt = new GBTClassifier().setLabelCol("label").setFeaturesCol("hwmc_i").setMaxIter(10)

        val nb = new NaiveBayes().setLabelCol("label").setFeaturesCol("hwmc_i")
        val labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels)
        val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, nb, labelConverter))


        val model = pipeline.fit(trainingData)

        val evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy")

        val trainingPredictions = model.transform(trainingData)
        //trainingPredictions.select("predictedLabel", "spbm", "hwmc_i").show
        val trainingAccuracy = evaluator.evaluate(trainingPredictions)
        println("Training Error = " + (1.0 - trainingAccuracy))

        val testPredictions = model.transform(testData)
        val accuracy = evaluator.evaluate(testPredictions)
        println("Test Error = " + (1.0 - accuracy))
        //val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
        //println("Learned classification tree model:\n" + treeModel.toDebugString)
    }


    def splitData(dataDF: DataFrame, prop: Double) : Array[DataFrame] = {
        val countVecModel: CountVectorizerModel = new CountVectorizer()
                .setInputCol("hwmc_unichars_and_ngrams")
                .setOutputCol("hwmc_feature")
                .setVocabSize(5000)
                .setMinDF(2)
                .fit(dataDF)
        val df = countVecModel.transform(dataDF)
        val Array(trainingData, testData) = df.randomSplit(Array(1.0-prop, prop))
        Array(trainingData, testData)
    }

    def train2(dataDF: DataFrame): Unit = {
        val Array(trainingData, testData) = splitData(dataDF, 0.3)

        val cvModel: CountVectorizerModel = new CountVectorizer()
                .setInputCol("hwmc_unichars_and_ngrams")
                .setOutputCol("hwmc_f")
                .setVocabSize(5000)
                .setMinDF(2)
                .fit(dataDF)
        val df = cvModel.transform(dataDF)
        val labelIndexer = new StringIndexer()
                .setInputCol("spbm")
                .setOutputCol("label")
                .fit(df)

        val featureIndexer = new VectorIndexer()
                .setInputCol("hwmc_feature")
                .setOutputCol("feature")
                .setMaxCategories(4)
                .fit(df)

        val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("hwmc_i")

    }
}

object CommodityClassifier{
    def main(args: Array[String]): Unit ={
        val context = new AppContext()
        val commodityCls = new CommodityClassifier(context)
        val dataLoader = new DataLoader(context)
        val sql = "select hwmc, spbm from dzdz_an.dzdz_hwxx_ptfp where spbm is not null"
        val dataDF = dataLoader.getData(sql)
        commodityCls.train(dataDF)
    }
}

