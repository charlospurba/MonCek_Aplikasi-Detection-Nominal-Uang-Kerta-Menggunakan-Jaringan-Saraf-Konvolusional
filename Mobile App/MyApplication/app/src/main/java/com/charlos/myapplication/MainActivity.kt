package com.charlos.myapplication

import android.Manifest
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var resultTextView: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var imageCapture: ImageCapture
    private lateinit var cameraExecutor: ExecutorService
    private val client = OkHttpClient()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        Log.d("MainActivity", "Camera permission granted: $isGranted")
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(
                this,
                "Camera permission denied. Please enable it in Settings.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val inputStream = contentResolver.openInputStream(it)
            val imageBytes = inputStream?.readBytes()
            inputStream?.close()
            if (imageBytes != null) {
                uploadPhoto(imageBytes)
            } else {
                Toast.makeText(this, "Failed to read image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { view, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom + 32)
            insets
        }

        viewFinder = findViewById(R.id.viewFinder)
        resultTextView = findViewById(R.id.resultTextView)
        progressBar = findViewById(R.id.progressBar)
        val takePhotoBtn = findViewById<Button>(R.id.takePhotoButton)
        val selectFromGalleryBtn = findViewById<Button>(R.id.selectFromGalleryButton)

        cameraExecutor = Executors.newSingleThreadExecutor()

        takePhotoBtn.setOnClickListener {
            takePhoto()
        }

        selectFromGalleryBtn.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        requestCameraPermission()
    }

    private fun requestCameraPermission() {
        val isCameraGranted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        Log.d("MainActivity", "Checking CAMERA: Granted = $isCameraGranted")
        if (!isCameraGranted) {
            Log.d("MainActivity", "Requesting CAMERA permission")
            permissionLauncher.launch(Manifest.permission.CAMERA)
        } else {
            Log.d("MainActivity", "Camera permission already granted, starting camera")
            startCamera()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )
                Log.d("MainActivity", "Camera started successfully")
            } catch (exc: Exception) {
                Log.e("MainActivity", "Failed to start camera: ${exc.message}")
                runOnUiThread {
                    Toast.makeText(this, "Failed to start camera: ${exc.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        progressBar.visibility = View.VISIBLE
        imageCapture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val buffer = image.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    image.close()

                    runOnUiThread {
                        progressBar.visibility = View.GONE
                        Toast.makeText(this@MainActivity, "Photo captured", Toast.LENGTH_SHORT).show()
                        uploadPhoto(bytes)
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    runOnUiThread {
                        progressBar.visibility = View.GONE
                        Toast.makeText(this@MainActivity, "Photo capture failed: ${exception.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        )
    }

    private fun uploadPhoto(imageBytes: ByteArray) {
        progressBar.visibility = View.VISIBLE
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file",
                "photo.jpg",
                imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
            )
            .build()

        val request = Request.Builder()
            .url("https://d7db-103-177-96-43.ngrok-free.app/predict")
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: java.io.IOException) {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    Toast.makeText(this@MainActivity, "Upload failed: ${e.message}", Toast.LENGTH_SHORT).show()
                    resultTextView.text = "Error: ${e.message}"
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val jsonString = response.body?.string()
                Log.d("MainActivity", "API Response: $jsonString")
                val json = JSONObject(jsonString)
                val audioUrl = json.getString("audio_link")
                val predictedClass = json.getString("predicted_class")
                Log.d("MainActivity", "Predicted class: '$predictedClass'")

                // Normalisasi predicted_class dengan pembersihan lebih agresif
                val normalizedPredictedClass = predictedClass.trim()
                    .lowercase()
                    .replace(Regex("[^a-z0-9]"), "")
                Log.d("MainActivity", "Normalized predicted class: '$normalizedPredictedClass'")

                // Pemetaan nominal dengan format yang lebih fleksibel
                val nominalText = when {
                    normalizedPredictedClass in listOf("1k", "1rb", "1ribu", "1000") -> "Rp 1.000"
                    normalizedPredictedClass in listOf("2k", "2rb", "2ribu", "2000") -> "Rp 2.000"
                    normalizedPredictedClass in listOf("5k", "5rb", "5ribu", "5000") -> "Rp 5.000"
                    normalizedPredictedClass in listOf("10k", "10rb", "10ribu", "10000") -> "Rp 10.000"
                    normalizedPredictedClass in listOf("20k", "20rb", "20ribu", "20000") -> "Rp 20.000"
                    normalizedPredictedClass in listOf("50k", "50rb", "50ribu", "50000") -> "Rp 50.000"
                    normalizedPredictedClass in listOf("100k", "100rb", "100ribu", "100000") -> "Rp 100.000"
                    else -> "Tidak diketahui ($predictedClass)"
                }

                runOnUiThread {
                    progressBar.visibility = View.GONE
                    resultTextView.text = "💵 Uang terdeteksi: $nominalText"
                    playAudio(audioUrl)
                }
            }
        })
    }

    private fun playAudio(url: String) {
        val mediaPlayer = MediaPlayer().apply {
            setDataSource(url)
            setOnPreparedListener { start() }
            setOnCompletionListener { release() }
            prepareAsync()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}