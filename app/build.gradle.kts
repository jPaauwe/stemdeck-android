plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
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
    // One APK per ABI, native libraries compressed: ONNX Runtime alone is
    // 34 MB (arm64) and 41 MB (x86_64) uncompressed, which made a single
    // APK 87 MB. Phones get the arm64 APK; the x86_64 one is for the emulator.
    // Flavors rather than APK splits, because Chaquopy reads the ABIs from
    // ndk.abiFilters, which Gradle refuses alongside splits.
    flavorDimensions += "abi"
    productFlavors {
        create("arm64") { dimension = "abi"; ndk { abiFilters += "arm64-v8a" } }
        create("x86_64") { dimension = "abi"; ndk { abiFilters += "x86_64" } }
    }
    packaging {
        jniLibs { useLegacyPackaging = true }
    }
}

// Link import (1.4): yt-dlp (Unlicense) and its YouTube challenge solver
// scripts, yt-dlp-ejs (Unlicense), on Chaquopy's Python. Pinned: yt-dlp 2026.08.19
// asks for yt-dlp-ejs 0.8.0. The app can update yt-dlp itself from Settings.
chaquopy {
    defaultConfig {
        version = "3.13"
        pip {
            install("yt-dlp==2026.8.19")
            install("yt-dlp-ejs==0.8.0")
            install("certifi")
        }
    }
}

// QuickJS-NG's qjs, the JavaScript runtime yt-dlp runs for YouTube's
// challenges, built with the NDK as an executable named libqjs.so so that
// Android installs it in the native library directory, where an app may run
// programs (src/main/qjs/CMakeLists.txt).
val qjsAbis = listOf("arm64-v8a", "x86_64")
val qjsOut = layout.buildDirectory.dir("qjs/jniLibs")
val buildQuickJs by tasks.registering {
    inputs.file("src/main/qjs/CMakeLists.txt")
    outputs.dir(qjsOut)
    doLast {
        val sdk = android.sdkDirectory
        // The SDK's CMake if it is installed (Android Studio installs it on
        // first sync; Gradle only fetches it later in a build), else the one
        // on PATH.
        val sdkCmake = File(sdk, "cmake/3.22.1/bin/cmake")
        val cmake = if (sdkCmake.exists()) sdkCmake else File("cmake")
        val ninja = File(sdk, "cmake/3.22.1/bin/ninja")
        val generator = if (ninja.exists()) listOf("-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=${ninja.path}")
            else listOf("-G", "Unix Makefiles")
        val toolchain = File(android.ndkDirectory, "build/cmake/android.toolchain.cmake")
        fun run(vararg cmd: String) {
            val p = ProcessBuilder(*cmd).inheritIO().start()
            if (p.waitFor() != 0) throw GradleException("failed: ${cmd.joinToString(" ")}")
        }
        for (abi in qjsAbis) {
            val build = layout.buildDirectory.dir("qjs/build-$abi").get().asFile
            run(cmake.path, "-S", file("src/main/qjs").path, "-B", build.path, *generator.toTypedArray(),
                "-DCMAKE_TOOLCHAIN_FILE=${toolchain.path}",
                "-DANDROID_ABI=$abi", "-DANDROID_PLATFORM=android-29", "-DCMAKE_BUILD_TYPE=Release")
            run(cmake.path, "--build", build.path, "--target", "qjs_exe")
            File(build, "out/libqjs.so").copyTo(qjsOut.get().dir(abi).file("libqjs.so").asFile, overwrite = true)
        }
    }
}
android.sourceSets.getByName("main").jniLibs.srcDir(qjsOut)
tasks.named("preBuild") { dependsOn(buildQuickJs) }

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
