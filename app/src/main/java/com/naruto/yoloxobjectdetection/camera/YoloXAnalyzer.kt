package com.naruto.yoloxobjectdetection.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.core.graphics.get
import androidx.core.graphics.scale
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer


class YoloXAnalyzer(
    context: Context,
    private val onDetections: (List<Detection>) -> Unit
) : ImageAnalysis.Analyzer {
    companion object{
        private const val TAG = "YoloXAnalyzer"
        private const val MODEL = "Yolo-X.tflite"
        private const val TV_CLASS = 62
    }

    private var tfLite: InterpreterApi? = null
    private lateinit var probabilityBuffer: TensorBuffer

    init {
        try {
            val tfLiteModel
                    : MappedByteBuffer = FileUtil.loadMappedFile(
                context,
                MODEL
            )
            tfLite = InterpreterApi.create(tfLiteModel, InterpreterApi.Options())

            val inputTensor = tfLite?.getInputTensor(0)
            Log.d(TAG, "Input: dtype=${inputTensor?.dataType()}, shape=${inputTensor?.shape().contentToString()}")

            val outputTensor = tfLite?.getOutputTensor(0)
            Log.i(TAG, "Type: ${outputTensor?.dataType()} Shape: ${outputTensor?.shape().contentToString()}")

            for (i in 0 until (tfLite?.outputTensorCount ?: 0)) {
                val tensor = tfLite?.getOutputTensor(i)
                Log.i(TAG, "Output $i: dtype=${tensor?.dataType()}, shape=${tensor?.shape().contentToString()}")
            }


            probabilityBuffer =
                TensorBuffer.createFixedSize(intArrayOf(1, 8400, 4), DataType.FLOAT32);
            Log.i(TAG, "init - All Initialized")

        } catch (e: IOException) {
            Log.e(TAG, "Error reading model", e)
        }
    }

    fun Bitmap.resizeAndPadTo640(): Bitmap {
        val targetSize = 640
        val aspectRatio = width.toFloat() / height.toFloat()

        val newWidth: Int
        val newHeight: Int

        if (width >= height) {
            newWidth = targetSize
            newHeight = (targetSize / aspectRatio).toInt()
        } else {
            newHeight = targetSize
            newWidth = (targetSize * aspectRatio).toInt()
        }

        // Resize bitmap keeping aspect ratio
        val scaledBitmap = Bitmap.createScaledBitmap(this, newWidth, newHeight, true)

        // Create a blank 640x640 bitmap and draw the scaled image centered
        val paddedBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(paddedBitmap)
        val left = (targetSize - newWidth) / 2f
        val top = (targetSize - newHeight) / 2f
        canvas.drawColor(android.graphics.Color.BLACK)
        canvas.drawBitmap(scaledBitmap, left, top, null)

        return paddedBitmap
    }


    override fun analyze(imageProxy: ImageProxy) {
        Log.i(TAG,"analyze")
        val bitmap = imageProxy.toBitmap(imageProxy.imageInfo.rotationDegrees) ?: return
        val input = bitmap.preprocessToYoloXInput()

        val outputBoxes = Array(1) { Array(8400) { FloatArray(4) } }  // cx, cy, w, h
        val outputScores = Array(1) { FloatArray(8400) }             // confidence scores
        val outputClasses = Array(1) { ByteArray(8400) }            // class indices


        val outputs = mapOf(
            0 to outputBoxes,
            1 to outputScores,
            2 to outputClasses
        )
        tfLite?.runForMultipleInputsOutputs(arrayOf(input), outputs)
        Log.i(TAG,"outputBoxes - ${outputBoxes.size}")
        Log.i(TAG,"outputScores - ${outputScores.size}")
        Log.i(TAG,"outputClasses - ${outputClasses.size}")
        val detections = mutableListOf<Detection>()
        for (i in 0 until 8400) {
            val score = outputScores[0][i]
            val classId = outputClasses[0][i].toInt() and 0xFF
            if (score > 0.1f && classId==62) {
                val box = outputBoxes[0][i]
                val x = box[0]
                val y = box[1]
                val w = box[2]
                val h = box[3]
                detections.add(Detection(x, y, w, h, score, classId))
            }
        }
        if (detections.isEmpty()){
            onDetections(emptyList())
        }
        else{
            onDetections(listOf(weightedBoxFusion(detections)))
        }

        imageProxy.close()
    }

    private fun weightedBoxFusion(detections: List<Detection>): Detection {
        val totalScore = detections.sumOf { it.score.toDouble() }.toFloat()
        val left = detections.sumOf { it.left.toDouble() * it.score }.toFloat() / totalScore
        val top = detections.sumOf { it.top.toDouble() * it.score }.toFloat() / totalScore
        val right = detections.sumOf { it.right.toDouble() * it.score }.toFloat() / totalScore
        val bottom = detections.sumOf { it.bottom.toDouble() * it.score }.toFloat() / totalScore
        val score = detections.maxOfOrNull { it.score } ?: 0.0F

        return Detection(left, top, right, bottom, score, detections.first().classId).also {
            Log.i(TAG,"detection - $it")
        }
    }

    private fun ImageProxy.toBitmap(rotationDegrees: Int): Bitmap? {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val yuv = out.toByteArray()
        var bitmap = BitmapFactory.decodeByteArray(yuv, 0, yuv.size)

        if (rotationDegrees != 0) {
            val matrix = android.graphics.Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }
        return bitmap
    }

    private fun Bitmap.preprocessToYoloXInput(): ByteBuffer {
        val inputImage = this.scale(640,640)
        val byteBuffer = ByteBuffer.allocateDirect(1 * 640 * 640 * 3 * 4) // 4 bytes per float
        byteBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until 640) {
            for (x in 0 until 640) {
                val pixel = inputImage[x, y]
                byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f) // R
                byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)  // G
                byteBuffer.putFloat((pixel and 0xFF) / 255f)          // B
            }
        }
        return byteBuffer
    }
}
