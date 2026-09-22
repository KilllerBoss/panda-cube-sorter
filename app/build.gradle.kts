plugins {
    id("com.android.application")
}

android {
    namespace = "dev.pandasorter"
    compileSdk = 34
    ndkVersion = "27.1.12297006"

    defaultConfig {
        applicationId = "dev.pandasorter"
        minSdk = 31
        targetSdk = 34
        versionCode = 2
        versionName = "1.0.1"
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake { arguments += listOf("-DANDROID_STL=c++_shared") }
        }
    }

    signingConfigs {
        create("release") {
            storeFile = file("release.keystore")
            storePassword = "pandasorter2026"
            keyAlias = "pandasorter"
            keyPassword = "pandasorter2026"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    packagingOptions {
        jniLibs {
            // extract to disk at install: avoids direct-APK dlopen alignment
            // issues on 4 KB devices; ELF alignment itself is now 16 KB anyway
            useLegacyPackaging = true
        }
    }
}

repositories {
    google()
    mavenCentral()
}

dependencies {
}
