package org.apache.spark.ml.made

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, DoubleParam}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{StructType, DoubleType, StructField}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.functions.lit
import org.apache.spark.ml.feature.VectorAssembler


trait LinRegParams extends HasFeaturesCol with HasLabelCol with HasOutputCol {
  def setFeatureCol(value: String) : this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val numIters = new IntParam(
    this, "numIters","Number of iterations"
  )
  val learningRate = new DoubleParam(
    this, "learningRate","Learning rate for descent")

  def getIters: Int = $(numIters)
  def getLr : Double = $(learningRate)

  def setIters(value: Int): this.type = set(numIters, value)
  def setLr(value: Double) : this.type = set(learningRate, value)

  setDefault(numIters -> 1000)
  setDefault(learningRate -> 0.01)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, DoubleType)
      schema
    } else {
      SchemaUtils.appendColumn(schema, StructField(getOutputCol, DoubleType))
    }
  }
}

class LinReg(override val uid: String) extends Estimator[LinRegModel] with LinRegParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linReg"))

  override def fit(dataset: Dataset[_]): LinRegModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val dataWithBias = dataset.select(dataset($(featuresCol)), lit(1.0).as("bias"), dataset($(labelCol)))

    val vecAssembler = new VectorAssembler().setInputCols(Array($(featuresCol), "bias", $(labelCol))).setOutputCol("concat")

    val assembledData = vecAssembler.transform(dataWithBias)

    val vectors: Dataset[Vector] = assembledData.select(assembledData("concat").as[Vector])

    val dim: Int = AttributeGroup.fromStructField(assembledData.schema("concat")).numAttributes.getOrElse(
      vectors.first().size
    )

    var weights = Vectors.zeros(dim - 1).asBreeze.toDenseVector


    for (i <- 1 to $(numIters)) {
      val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val result = data.foldLeft(new MultivariateOnlineSummarizer())(
          (summarizer, vector) => summarizer.add(
            {
              val x = vector.asBreeze(0 until dim - 1).toDenseVector
              val y = vector.asBreeze(-1)
              //println(x, y)
              mllib.linalg.Vectors.fromBreeze(((x * weights) - y) * x)
            }
          ))
        Iterator(result)
      }).reduce(_ merge _)

      weights = weights - summary.mean.asML.asBreeze.toDenseVector * $(learningRate)
    }

    //println(weights)
    val weightsWithoutBias = Vectors.fromBreeze(weights(0 until weights.size - 1))
    val bias = weights(-1)

    copyValues(new LinRegModel(weightsWithoutBias, bias)).setParent(this)

//    val Row(row: Row) =  dataset
//      .select(Summarizer.metrics("mean", "std").summary(dataset($(inputCol))))
//      .first()
//
//    copyValues(new StandardScalerModel(row.getAs[Vector]("mean").toDense, row.getAs[Vector](1).toDense)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinRegModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinReg extends DefaultParamsReadable[LinReg]

class LinRegModel private[made](
                           override val uid: String,
                           val weights: DenseVector,
                           val bias: Double) extends Model[LinRegModel] with LinRegParams with MLWritable {


  private[made] def this(weights: Vector, bias: Double) =
    this(Identifiable.randomUID("linRegModel"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinRegModel = copyValues(
    new LinRegModel(weights, bias), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bWeightgs = weights.asBreeze
    val transformUdf = dataset.sqlContext.udf.register(uid + "_prediction",
      (x : Vector) => {
        x.asBreeze.dot(bWeightgs) + bias
      })

    dataset.withColumn($(outputCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = weights.toArray :+ bias

      sqlContext.createDataFrame(Seq(Tuple1(Vectors.dense(vectors)))).write.parquet(path + "/vectors")
    }
  }
}

object LinRegModel extends MLReadable[LinRegModel] {
  override def read: MLReader[LinRegModel] = new MLReader[LinRegModel] {
    override def load(path: String): LinRegModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val params = vectors.select(vectors("_1").as[Vector]).first().asBreeze.toDenseVector

      val weights = Vectors.fromBreeze(params(0 until params.size - 1))
      val bias = params(-1) //(params.size - 1)

      val model = new LinRegModel(weights, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}
