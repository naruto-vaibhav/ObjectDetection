package com.naruto.yoloxobjectdetection

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale
import com.naruto.yoloxobjectdetection.screen.CameraViewAndAnalysis
import com.naruto.yoloxobjectdetection.ui.theme.YoloxObjectDetectionTheme

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalPermissionsApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            YoloxObjectDetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    val cameraPermissionState = rememberPermissionState(
                        android.Manifest.permission.CAMERA
                    )
                    if (cameraPermissionState.status.isGranted) {
                        Box(modifier = Modifier.padding(innerPadding)) {
                            Text("Camera permission Granted")
                            CameraViewAndAnalysis()
                        }
                    } else {
                        Column(modifier = Modifier.padding(innerPadding)) {
                            val textToShow = if (cameraPermissionState.status.shouldShowRationale) {
                                // If the user has denied the permission but the rationale can be shown,
                                // then gently explain why the app requires this permission
                                "The camera is important for this app. Please grant the permission."
                            } else {
                                // If it's the first time the user lands on this feature, or the user
                                // doesn't want to be asked again for this permission, explain that the
                                // permission is required
                                "Camera permission required for this feature to be available. " +
                                        "Please grant the permission"
                            }
                            Text(textToShow)
                            Button(onClick = { cameraPermissionState.launchPermissionRequest() }) {
                                Text("Request permission")
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    YoloxObjectDetectionTheme {
        Greeting("Android")
    }
}