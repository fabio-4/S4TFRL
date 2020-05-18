// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PPO",
    platforms: [
        .macOS(.v10_13),
    ],
    dependencies: [
        .package(url: "../S4TFUnityGym", from: "0.0.1"),
    ],
    targets: [
        .target(
            name: "PPO",
            dependencies: ["S4TFUnityGym"]),
    ]
)
