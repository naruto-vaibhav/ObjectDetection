package com.naruto.yoloxobjectdetection.screen

import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.naruto.yoloxobjectdetection.camera.Detection
import com.naruto.yoloxobjectdetection.camera.YoloXAnalyzer
import java.util.concurrent.Executors

@Composable
fun CameraViewAndAnalysis(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context) }

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val detections = remember { mutableStateListOf<Detection>() }

    DisposableEffect(Unit) {
        val cameraProvider = ProcessCameraProvider.getInstance(context).get()

        val preview = Preview.Builder().build().apply {
            surfaceProvider = previewView.surfaceProvider
        }

        val imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 640))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .apply {
                setAnalyzer(cameraExecutor, YoloXAnalyzer(context) { newDetections ->
                    detections.clear()
                    Log.i("CameraViewAndAnalysis", "$newDetections")
                    newDetections.forEach{
                        Log.i("CameraViewAndAnalysis", "$it")
                    }
                    detections.addAll(newDetections)
                })
            }

        cameraProvider.unbindAll()
        val camera = cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            imageAnalyzer
        )
        onDispose {
            cameraProvider.unbindAll()
            cameraExecutor.shutdown()
        }
    }

    Box(modifier = modifier.fillMaxSize()) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        Canvas(modifier = Modifier.fillMaxSize()) {
            val scaleX = size.width / 640f
            val scaleY = size.height / 640f
            detections.forEach { d ->
                val left = d.left * scaleX
                val top = d.top * scaleY
                val right = d.right * scaleX
                val bottom = d.bottom * scaleY

                drawRect(
                    color = Color.Red,
                    topLeft = Offset(left, top),
                    size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                    style = Stroke(width = 2f)
                )
                drawContext.canvas.nativeCanvas.drawText(
                    "Class ${d.classId} ${(d.score * 100).toInt()}%",
                    left,
                    top - 10,
                    android.graphics.Paint().apply {
                        color = android.graphics.Color.YELLOW
                        textSize = 30f
                        isAntiAlias = true
                    }
                )
            }
        }
    }
}