package com.naruto.yoloxobjectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.core.graphics.get
import androidx.core.graphics.scale
import com.naruto.yoloxobjectdetection.camera.Detection
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

class YoloxLAnalyser(
    private val context: Context,
    private val onDetections: (List<Detection>) -> Unit
) : ImageAnalysis.Analyzer {

    companion object{
        private const val TAG = "YoloXAnalyzer"
        private const val MODEL = "yolox_l_800.tflite"
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
                TensorBuffer.createFixedSize(intArrayOf(1, 13125, 85), DataType.FLOAT32)
            Log.i(TAG, "init - All Initialized")

        } catch (e: IOException) {
            Log.e(TAG, "Error reading model", e)
        }
    }

    private fun sigmoid(x: Float): Float = (1.0f / (1.0f + kotlin.math.exp(-x)))

    override fun analyze(imageProxy: ImageProxy) {
        Log.i(TAG,"analyze")
        val bitmap = imageProxy.toCustomBitmap(imageProxy.imageInfo.rotationDegrees) ?: return
        val input = preprocessBitmap(bitmap, 800)

        val outputArray = Array(1) { Array(13125) { FloatArray(85) } }

        tfLite?.run(input, outputArray)
        Log.i(TAG,"outputBoxes - ${outputArray[0][1].take(20)}")
        Log.i(TAG,"detections - START//")
        val detections = mutableListOf<Detection>()

        for (i in 0 until 13125) {
            val output = outputArray[0][i]

            val x = output[0]      // center x
            val y = output[1]      // center y
            val w = output[2]      // width
            val h = output[3]      // height
            Log.i(TAG,"bounding box - $x, $y, $w, $h")
            val objectness = sigmoid(output[4])
            Log.i(TAG,"objectness - $objectness")
            val classScores = output.copyOfRange(5, 85).map { sigmoid(it) }
            Log.i(TAG,"classScores - $classScores")
            val maxClassScore = classScores.maxOrNull() ?: 0f
            Log.i(TAG,"maxClassScore - $maxClassScore")
            val classId = classScores.indexOfFirst { it == maxClassScore }
            Log.i(TAG,"classId - $classId")
            val confidence = objectness * maxClassScore
            Log.i(TAG,"confidence - $confidence")

            if (confidence > 0.3f) { // You can tune threshold
                // Convert center x, y, w, h → left, top, right, bottom
                val left = x - w / 2
                val top = y - h / 2
                val right = x + w / 2
                val bottom = y + h / 2

                detections.add(
                    Detection(
                        left = left,
                        top = top,
                        right = right,
                        bottom = bottom,
                        score = confidence,
                        classId = classId
                    )
                )
            }
        }
        Log.i(TAG,"detections - END//")
        Log.i(TAG,"detections - ${detections.toList()}")
        onDetections(detections)
        imageProxy.close()
    }

    private fun preprocessBitmap(bitmap: Bitmap, inputSize: Int): ByteBuffer {
        val resized = bitmap.scale(inputSize, inputSize)
        val inputBuffer = ByteBuffer.allocateDirect(1 * 3 * inputSize * inputSize * 4) // FLOAT32
        inputBuffer.order(ByteOrder.nativeOrder())

        val rArray = FloatArray(inputSize * inputSize)
        val gArray = FloatArray(inputSize * inputSize)
        val bArray = FloatArray(inputSize * inputSize)

        var index = 0
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val px = resized[x, y]
                val r = Color.red(px) / 255.0f
                val g = Color.green(px) / 255.0f
                val b = Color.blue(px) / 255.0f

                // Apply normalization
                rArray[index] = (Color.red(px) / 255.0f - 0.485f) / 0.229f
                gArray[index] = (Color.green(px) / 255.0f - 0.456f) / 0.224f
                bArray[index] = (Color.blue(px) / 255.0f - 0.406f) / 0.225f
                index++
            }
        }

        for (i in bArray.indices) inputBuffer.putFloat(bArray[i]) // B first
        for (i in gArray.indices) inputBuffer.putFloat(gArray[i]) // G
        for (i in rArray.indices) inputBuffer.putFloat(rArray[i]) // R

        return inputBuffer
    }



    private fun ImageProxy.toCustomBitmap(rotationDegrees: Int): Bitmap? {
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
        val inputImage = this.scale(640, 640)
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
