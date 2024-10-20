package com.sparrowrecsys.offline.spark.embedding

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}
import redis.clients.jedis.Jedis
import redis.clients.jedis.params.SetParams

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

/**
 * embedding层
 * 与py的Embedding.py相对应
 */
object Embedding {

  val redisEndpoint = "localhost"
  val redisPort = 6379

  /**
   * Item序列处理,利用word2vec
   * 处理评分数据，生成评分电影序列数据
   * @param sparkSession SparkSession实例
   * @param rawSampleDataPath 评分数据路径
   * @return RDD[Seq[String]] 评分电影序列数据
   */
  def processItemSequence(sparkSession: SparkSession, rawSampleDataPath: String): RDD[Seq[String]] ={

    //path of rating data
    //设定rating数据的路径并用spark载入数据
    val ratingsResourcesPath = this.getClass.getResource(rawSampleDataPath)
    val ratingSamples = sparkSession.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)

    //sort by timestamp udf
    //实现一个用户定义的操作函数(UDF)，用于之后的排序
    val sortUdf: UserDefinedFunction = udf((rows: Seq[Row]) => {
      rows.map { case Row(movieId: String, timestamp: String) => (movieId, timestamp) }
        .sortBy { case (_, timestamp) => timestamp }
        .map { case (movieId, _) => movieId }
    })

    ratingSamples.printSchema()

    //process rating data then generate rating movie sequence data
    //把原始的rating数据处理成序列数据
    val userSeq = ratingSamples
      .where(col("rating") >= 3.5)
      .groupBy("userId")
      .agg(sortUdf(collect_list(struct("movieId", "timestamp"))) as "movieIds")
      .withColumn("movieIdStr", array_join(col("movieIds"), " "))

      //把序列数据筛选出来，丢掉其他过程数据
    userSeq.select("userId", "movieIdStr").show(10, truncate = false)
    userSeq.select("movieIdStr").rdd.map(r => r.getAs[String]("movieIdStr").split(" ").toSeq)
  }

  /**
   * 生成用户嵌入向量
   * @param sparkSession SparkSession实例
   * @param rawSampleDataPath 评分数据路径
   * @param word2VecModel item2vec模型
   * @param embLength 嵌入长度
   * @param embOutputFilename 嵌入输出文件名
   * @param saveToRedis 是否保存到Redis
   * @param redisKeyPrefix Redis键前缀
   */
  def generateUserEmb(sparkSession: SparkSession, rawSampleDataPath: String, word2VecModel: Word2VecModel, embLength:Int, embOutputFilename:String, saveToRedis:Boolean, redisKeyPrefix:String): Unit ={
    val ratingsResourcesPath = this.getClass.getResource(rawSampleDataPath)
    val ratingSamples = sparkSession.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    ratingSamples.show(10, false)

    val userEmbeddings = new ArrayBuffer[(String, Array[Float])]()

    ratingSamples.collect().groupBy(_.getAs[String]("userId"))
      .foreach(user => {
        val userId = user._1
        var userEmb = new Array[Float](embLength)

        var movieCount = 0
        userEmb = user._2.foldRight[Array[Float]](userEmb)((row, newEmb) => {
          val movieId = row.getAs[String]("movieId")
          val movieEmb = word2VecModel.getVectors.get(movieId)
          movieCount += 1
          if(movieEmb.isDefined){
            newEmb.zip(movieEmb.get).map { case (x, y) => x + y }
          }else{
            newEmb
          }
        }).map((x: Float) => x / movieCount)
        userEmbeddings.append((userId,userEmb))
      })

    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(embFolderPath.getPath + embOutputFilename)
    val bw = new BufferedWriter(new FileWriter(file))

    for (userEmb <- userEmbeddings) {
      bw.write(userEmb._1 + ":" + userEmb._2.mkString(" ") + "\n")
    }
    bw.close()

    if (saveToRedis) {
      val redisClient = new Jedis(redisEndpoint, redisPort)
      val params = SetParams.setParams()
      //set ttl to 24hs
      params.ex(60 * 60 * 24)

      for (userEmb <- userEmbeddings) {
        redisClient.set(redisKeyPrefix + ":" + userEmb._1, userEmb._2.mkString(" "), params)
      }
      redisClient.close()
    }
  }

  /**
   * 训练item2vec模型
   * @param sparkSession SparkSession实例
   * @param samples 评分电影序列数据
   * @param embLength 嵌入长度
   * @param embOutputFilename 嵌入输出文件名
   * @param saveToRedis 是否保存到Redis
   * @param redisKeyPrefix Redis键前缀
   * @return Word2VecModel item2vec模型
   */
  def trainItem2vec(sparkSession: SparkSession, samples : RDD[Seq[String]], embLength:Int, embOutputFilename:String, saveToRedis:Boolean, redisKeyPrefix:String): Word2VecModel = {
    // 设置模型参数
    val word2vec = new Word2Vec()
      .setVectorSize(embLength)
      .setWindowSize(5)
      .setNumIterations(10)

    // 训练item2vec模型
    val model = word2vec.fit(samples)

    // 打印模型中的前20个同义词
    val synonyms = model.findSynonyms("158", 20)
    for ((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    // 保存模型中的嵌入向量到文件
    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(embFolderPath.getPath + embOutputFilename)
    val bw = new BufferedWriter(new FileWriter(file))
    for (movieId <- model.getVectors.keys) {
      // 用model.getVectors方法获取每个电影id对应的嵌入向量，并将其转换为字符串形式写入文件
      bw.write(movieId + ":" + model.getVectors(movieId).mkString(" ") + "\n")
    }
    bw.close()

    // 如果需要将嵌入向量保存到Redis中，则进行以下操作
    if (saveToRedis) {
      val redisClient = new Jedis(redisEndpoint, redisPort)
      val params = SetParams.setParams()
      //set ttl to 24hs
      params.ex(60 * 60 * 24)
      for (movieId <- model.getVectors.keys) {
        redisClient.set(redisKeyPrefix + ":" + movieId, model.getVectors(movieId).mkString(" "), params)
      }
      redisClient.close()
    }

    // 调用embeddingLSH方法进行LSH嵌入
    embeddingLSH(sparkSession, model.getVectors)
    model
  }

  /**
   * 通过随机游走产生一个样本的过程
   * @param transitionMatrix 转移概率矩阵
   * @param itemDistribution 项目分布
   * @param sampleLength 随机游走长度
   * @return Seq[String] 随机游走序列
   */
  def oneRandomWalk(transitionMatrix : mutable.Map[String, mutable.Map[String, Double]], itemDistribution : mutable.Map[String, Double], sampleLength:Int): Seq[String] ={
    val sample = mutable.ListBuffer[String]()

    //pick the first element
    // 决定起始点
    val randomDouble = Random.nextDouble()
    var firstItem = ""
    var accumulateProb:Double = 0D

    //根据物品出现的概率，随机决定起始点
    breakable { for ((item, prob) <- itemDistribution) {
      accumulateProb += prob
      if (accumulateProb >= randomDouble){
        firstItem = item
        break
      }
    }}

    sample.append(firstItem)
    var curElement = firstItem

    breakable { for(_ <- 1 until sampleLength) {
      if (!itemDistribution.contains(curElement) || !transitionMatrix.contains(curElement)){
        break
      }

      //从curElement到下一个跳的转移概率向量
      val probDistribution = transitionMatrix(curElement)
      val randomDouble = Random.nextDouble()
      var accumulateProb: Double = 0D

      //根据转移概率向量随机决定下一跳的物品
      breakable { for ((item, prob) <- probDistribution) {
        accumulateProb += prob
        if (accumulateProb >= randomDouble){
          curElement = item
          break
        }
      }}
      sample.append(curElement)
    }}
    Seq(sample.toList : _*)
  }

  /**
   * 生成随机游走序列
   * @param transitionMatrix 转移概率矩阵
   * @param itemDistribution 项目分布
   * @param sampleCount 随机游走次数
   * @param sampleLength 随机游走长度
   * @return Seq[Seq[String]] 随机游走序列
   */
  def randomWalk(transitionMatrix : mutable.Map[String, mutable.Map[String, Double]], itemDistribution : mutable.Map[String, Double], sampleCount:Int, sampleLength:Int): Seq[Seq[String]] ={
    // 样本数量
    val samples = mutable.ListBuffer[Seq[String]]()
    for(_ <- 1 to sampleCount) {
      samples.append(oneRandomWalk(transitionMatrix, itemDistribution, sampleLength))
    }
    Seq(samples.toList : _*)
  }

  def generateTransitionMatrix(samples : RDD[Seq[String]]): (mutable.Map[String, mutable.Map[String, Double]], mutable.Map[String, Double]) ={
    //通过flatMap操作把观影序列打碎成一个个影片对
    val pairSamples = samples.flatMap[(String, String)]( sample => {
      var pairSeq = Seq[(String,String)]()
      var previousItem:String = null
      sample.foreach((element:String) => {
        if(previousItem != null){
          pairSeq = pairSeq :+ (previousItem, element)
        }
        previousItem = element
      })
      pairSeq
    })

    //统计影片对的数量
    val pairCountMap = pairSamples.countByValue()
    var pairTotalCount = 0L

    //转移概率矩阵的双层Map数据结构
    val transitionCountMatrix = mutable.Map[String, mutable.Map[String, Long]]()
    val itemCountMap = mutable.Map[String, Long]()

    //求取转移概率矩阵
    pairCountMap.foreach( pair => {
      val pairItems = pair._1
      val count = pair._2

      if(!transitionCountMatrix.contains(pairItems._1)){
        transitionCountMatrix(pairItems._1) = mutable.Map[String, Long]()
      }

      transitionCountMatrix(pairItems._1)(pairItems._2) = count
      itemCountMap(pairItems._1) = itemCountMap.getOrElse[Long](pairItems._1, 0) + count
      pairTotalCount = pairTotalCount + count
    })

    val transitionMatrix = mutable.Map[String, mutable.Map[String, Double]]()
    val itemDistribution = mutable.Map[String, Double]()

    transitionCountMatrix foreach {
      case (itemAId, transitionMap) =>
        transitionMatrix(itemAId) = mutable.Map[String, Double]()
        transitionMap foreach { case (itemBId, transitionCount) => transitionMatrix(itemAId)(itemBId) = transitionCount.toDouble / itemCountMap(itemAId) }
    }

    itemCountMap foreach { case (itemId, itemCount) => itemDistribution(itemId) = itemCount.toDouble / pairTotalCount }
    (transitionMatrix, itemDistribution)
  }

  def embeddingLSH(spark:SparkSession, movieEmbMap:Map[String, Array[Float]]): Unit ={

    val movieEmbSeq = movieEmbMap.toSeq.map(item => (item._1, Vectors.dense(item._2.map(f => f.toDouble))))
    val movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")

    //LSH bucket model
    val bucketProjectionLSH = new BucketedRandomProjectionLSH()
      .setBucketLength(0.1)
      .setNumHashTables(3)
      .setInputCol("emb")
      .setOutputCol("bucketId")

    val bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    val embBucketResult = bucketModel.transform(movieEmbDF)
    println("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    println("movieId, emb, bucketId data result:")
    embBucketResult.show(10, truncate = false)

    println("Approximately searching for 5 nearest neighbors of the sample embedding:")
    val sampleEmb = Vectors.dense(0.795,0.583,1.120,0.850,0.174,-0.839,-0.0633,0.249,0.673,-0.237)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(truncate = false)
  }

  /**
   * 基于Deep Walk方法训练Graph Embedding
   * 图嵌入
   * @param samples 评分电影序列数据
   * @param sparkSession SparkSession实例
   * @param embLength 嵌入长度
   * @param embOutputFilename 嵌入输出文件名
   * @param saveToRedis 是否保存到Redis
   * @param redisKeyPrefix Redis键前缀
   * @return Word2VecModel item2vec模型
   */
  def graphEmb(samples : RDD[Seq[String]], sparkSession: SparkSession, embLength:Int, embOutputFilename:String, saveToRedis:Boolean, redisKeyPrefix:String): Word2VecModel ={
    //通过flatMap操作把观影序列打碎成一个个影片对
    val transitionMatrixAndItemDis = generateTransitionMatrix(samples)

    println(transitionMatrixAndItemDis._1.size)
    println(transitionMatrixAndItemDis._2.size)

    // 随机游走次数
    val sampleCount = 20000
    // 随机游走长度
    val sampleLength = 10
    // 生成随机游走序列
    val newSamples = randomWalk(transitionMatrixAndItemDis._1, transitionMatrixAndItemDis._2, sampleCount, sampleLength)

    // 将随机游走序列转换为RDD  
    val rddSamples = sparkSession.sparkContext.parallelize(newSamples)
    // 训练item2vec模型
    trainItem2vec(sparkSession, rddSamples, embLength, embOutputFilename, saveToRedis, redisKeyPrefix)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    val rawSampleDataPath = "/webroot/sampledata/ratings.csv"
    val embLength = 10

    val samples = processItemSequence(spark, rawSampleDataPath)
    val model = trainItem2vec(spark, samples, embLength, "item2vecEmb.csv", saveToRedis = false, "i2vEmb")
    //graphEmb(samples, spark, embLength, "itemGraphEmb.csv", saveToRedis = true, "graphEmb")
    //generateUserEmb(spark, rawSampleDataPath, model, embLength, "userEmb.csv", saveToRedis = false, "uEmb")
  }
}
