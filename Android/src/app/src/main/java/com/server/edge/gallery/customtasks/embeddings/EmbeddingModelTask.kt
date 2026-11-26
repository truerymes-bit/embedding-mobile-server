package com.server.edge.gallery.customtasks.embeddings

import android.content.Context
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.TextFields
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.EmbeddingEngine
import com.google.ai.edge.litertlm.EmbeddingEngineConfig
import com.server.edge.gallery.customtasks.common.CustomTask
import com.server.edge.gallery.customtasks.common.CustomTaskData
import com.server.edge.gallery.data.Category
import com.server.edge.gallery.data.Model
import com.server.edge.gallery.data.Task
import com.server.edge.gallery.ui.modelmanager.ModelInitializationStatusType
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.IntoSet
import javax.inject.Inject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

private const val TAG = "EmbeddingModelTask"

object EmbeddingModelRuntime {
  const val MODEL_ID = "embeddinggemma-2-740m"

  /** Shared lock prevents inference from racing model initialization or cleanup. */
  val mutex = Mutex()

  @Volatile
  var engine: EmbeddingEngine? = null
}

class EmbeddingModelTask @Inject constructor(
  @ApplicationContext context: Context,
) : CustomTask {
  private val modelFileName = "embeddinggemma-2-740m_Qualcomm_SM8750.litertlm"
  private val modelFilePath =
    File(context.getExternalFilesDir(null) ?: context.filesDir, modelFileName).absolutePath
  private val embeddingModel =
    Model(
      name = EmbeddingModelRuntime.MODEL_ID,
      displayName = "EmbeddingGemma 2 740M",
      info = "On-device text embedding model using the Qualcomm NPU. Place the model file in the app's external files directory: $modelFilePath.",
      localModelFilePathOverride = modelFilePath,
      downloadFileName = modelFileName,
      sizeInBytes = 543522816L,
      showBenchmarkButton = false,
      showRunAgainButton = false,
    )

  override val task: Task =
    Task(
      id = "embedding_models",
      label = "Embeddings",
      category = Category.LLM,
      icon = Icons.Outlined.TextFields,
      description = "Load and manage the on-device EmbeddingGemma model used by the OpenAI-compatible embeddings API.",
      shortDescription = "On-device text embeddings",
      models = mutableListOf(embeddingModel),
    )

  override fun initializeModelFn(
    context: Context,
    coroutineScope: CoroutineScope,
    model: Model,
    onDone: (error: String) -> Unit,
  ) {
    coroutineScope.launch(Dispatchers.IO) {
      try {
        EmbeddingModelRuntime.mutex.withLock {
          val modelFile = java.io.File(model.getPath(context))
          check(modelFile.isFile) { "Embedding model file not found: ${modelFile.absolutePath}" }

          val initializedEngine =
            EmbeddingEngine(
              EmbeddingEngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir),
                cacheDir = context.cacheDir.absolutePath,
              )
            )
          initializedEngine.initialize()

          model.instance = initializedEngine
          EmbeddingModelRuntime.engine = initializedEngine
        }
        Log.i(TAG, "Loaded ${model.name} with EmbeddingEngine")
        onDone("")
      } catch (e: Exception) {
        model.instance = null
        Log.e(TAG, "Failed to initialize ${model.name}", e)
        onDone(e.message ?: "Failed to initialize embedding model")
      }
    }
  }

  override fun cleanUpModelFn(
    context: Context,
    coroutineScope: CoroutineScope,
    model: Model,
    onDone: () -> Unit,
  ) {
    coroutineScope.launch(Dispatchers.IO) {
      try {
        EmbeddingModelRuntime.mutex.withLock {
          val loadedEngine = model.instance as? EmbeddingEngine
          if (loadedEngine?.isInitialized() == true) {
            loadedEngine.close()
          }
          if (EmbeddingModelRuntime.engine === loadedEngine) {
            EmbeddingModelRuntime.engine = null
          }
          model.instance = null
        }
        Log.i(TAG, "Unloaded ${model.name}")
      } catch (e: Exception) {
        Log.e(TAG, "Failed to clean up ${model.name}", e)
      } finally {
        onDone()
      }
    }
  }

  @Composable
  override fun MainScreen(data: Any) {
    val modelManagerViewModel = (data as CustomTaskData).modelManagerViewModel
    val uiState by modelManagerViewModel.uiState.collectAsState()
    val loaded =
      uiState.modelInitializationStatus[EmbeddingModelRuntime.MODEL_ID]?.status ==
        ModelInitializationStatusType.INITIALIZED

    Column(
      modifier = Modifier.padding(24.dp),
      verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
      Text("EmbeddingGemma 2 740M", style = MaterialTheme.typography.titleLarge)
      Text(
        if (loaded) {
          "Loaded and available at POST /v1/embeddings."
        } else {
          "Load this model from its model card to enable POST /v1/embeddings.",
        )
      )
      Text("OpenAI model ID: ${EmbeddingModelRuntime.MODEL_ID}")
    }
  }
}

@Module
@InstallIn(SingletonComponent::class)
internal object EmbeddingModelTaskModule {
  @Provides
  @IntoSet
  fun provideTask(): CustomTask = EmbeddingModelTask()
}
