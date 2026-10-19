plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.stemdeckport.android"
    compileSdk = 36
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "org.stemdeckport.android"
        minSdk = 29
        targetSdk = 35
        versionCode = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
        // VERSION, or VERSION-dev.N for development builds (see .github/workflows/build.yml).
        versionName = System.getenv("APP_VERSION")?.ifEmpty { null } ?: "1.0.0-dev"

        ndk {
            // arm64 for phones; x86_64 so it also runs in the emulator.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_shared", "-DCMAKE_BUILD_TYPE=Release")
            }
        }
    }

    signingConfigs {
        // Every build must be signed with the same key, or Android refuses to
        // install an update over the previous one. CI decodes the key from
        // the SIGNING_KEYSTORE_BASE64 repository secret (see signing/README.md).
        // Without it, the build falls back to the debug key.
        create("release") {
            val ks = System.getenv("SIGNING_KEYSTORE_PATH")
            if (ks != null && file(ks).exists()) {
                storeFile = file(ks)
                storePassword = System.getenv("SIGNING_KEYSTORE_PASSWORD")?.ifEmpty { null } ?: "stemdeck-sideload"
                keyAlias = System.getenv("SIGNING_KEY_ALIAS")?.ifEmpty { null } ?: "sideload"
                keyPassword = System.getenv("SIGNING_KEY_PASSWORD")?.ifEmpty { null } ?: "stemdeck-sideload"
            } else {
                initWith(getByName("debug"))
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
        }
        debug {
            // The engine is unusably slow without optimisation.
            externalNativeBuild { cmake { arguments += "-DCMAKE_BUILD_TYPE=Release" } }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    lint {
        // Lint must not block a sideload build; run ./gradlew lint to see its findings.
        checkReleaseBuilds = false
        abortOnError = false
    }
    buildFeatures {
        buildConfig = true
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
    packaging {
        jniLibs { useLegacyPackaging = false }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.webkit:webkit:1.12.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("org.nanohttpd:nanohttpd:2.3.1")
    // The analysis models (beat grid, and from 1.2 sections and the vocal
    // split) run on ONNX Runtime. Only its native library is used, from C++
    // (analysis/src/ort.cpp); the C API header there is from the same version.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
