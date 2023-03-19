package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector}
import com.google.common.io.Files

import scala.collection.JavaConverters._
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

class LinRegTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.01
  lazy val data: DataFrame = LinRegTest._data

  "Model" should "can predict" in {
    val model: LinRegModel = new LinRegModel(
      weights = Vectors.dense(1.5, 0.3, -0.7),
      bias = 1.2
    ).setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("prediction")

    val prediction = model.transform(data).collect().map(_.getAs[Double](2))

    prediction.length should be(data.count())

    validateModel(model, model.transform(data))
  }

  "Estimator" should "calculate weights and bias" in {
    val estimator = new LinReg()
      .setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("prediction")

    val model = estimator.fit(data)

    validateModel(model, model.transform(data))
  }

  "Estimator" should "should produce functional model" in {
    val estimator = new LinReg()
      .setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("prediction")

    val model = estimator.fit(data)

    val predictions: Array[Double] = data.collect().map(_.getAs[Double](2))
    val labels: Array[Double] = data.collect().map(_.getAs[Double](0))

    predictions.length should be(data.count())

    (predictions zip labels).foreach {case (pred: Double, label: Double) => pred should be(label +- delta)}
  }

  "Estimator" should "not learn with 0 iterations" in {
    val estimator = new LinReg()
      .setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("prediction")

    estimator.setIters(0)

    val model = estimator.fit(data)

    model.weights(0) should be(0.0 +- delta)
    model.weights(1) should be(0.0 +- delta)
    model.weights(2) should be(0.0 +- delta)
    model.bias should be(0.0 +- delta)
  }

  "Estimator" should "not learn with 0 learning rate" in {
    val estimator = new LinReg()
      .setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("prediction")

    estimator.setLr(0.0)

    val model = estimator.fit(data)

    model.weights(0) should be(0.0 +- delta)
    model.weights(1) should be(0.0 +- delta)
    model.weights(2) should be(0.0 +- delta)
    model.bias should be(0.0 +- delta)
  }

  "Estimator" should "not learn with negative learning rate" in {
    val estimator = new LinReg()
      .setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("prediction")

    estimator.setLr(-1.0)

    val model = estimator.fit(data)

    model.weights(0) should not be(1.5 +- delta)
    model.weights(1) should not be(0.3 +- delta)
    model.weights(2) should not be(-0.7 +- delta)
    model.bias should not be(1.2 +- delta)
  }

  private def validateModel(model: LinRegModel, data: DataFrame) = {
    val predictions: Array[Double] = data.collect().map(_.getAs[Double](2))

    model.weights(0) should be(1.5 +- delta)
    model.weights(1) should be(0.3 +- delta)
    model.weights(2) should be(-0.7 +- delta)
    model.bias should be(1.2 +- delta)
    println("Weights", model.weights)
    predictions.length should be(data.count())
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinReg()
        .setFeatureCol("features")
        .setLabelCol("label")
        .setOutputCol("prediction")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinRegModel]

    validateModel(model, model.transform(data))
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinReg()
        .setFeatureCol("features")
        .setLabelCol("label")
        .setOutputCol("prediction")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(model.stages(0).asInstanceOf[LinRegModel], reRead.transform(data))
  }
}

object LinRegTest extends WithSpark {

  lazy val schema: StructType = StructType(
    Array(
      StructField("features", new VectorUDT()),
      StructField("label", DoubleType)
    ))

  lazy val matrix = DenseMatrix.rand(1000, 3)
  lazy val trueWeights = DenseVector(1.5, 0.3, -0.7)
  lazy val trueBias = 1.2
  lazy val label = matrix * trueWeights + trueBias

  lazy val rowData = (0 until label.length).map(i => Row(Vectors.dense(matrix(i, ::).t.toArray), label(i)))
  //println(rowData)
  lazy val _data: DataFrame = sqlc.createDataFrame(rowData.asJava, schema)
}
