// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Typo",
    platforms: [.macOS(.v14)],
    targets: [
        .target(
            name: "TypoCore",
            path: "Sources/TypoCore",
            linkerSettings: [.linkedLibrary("sqlite3")]
        ),
        .executableTarget(name: "Typo", dependencies: ["TypoCore"], path: "Sources/Typo"),
        .executableTarget(name: "typo-cli", dependencies: ["TypoCore"], path: "Sources/typo-cli"),
        .testTarget(name: "TypoCoreTests", dependencies: ["TypoCore"], path: "Tests/TypoCoreTests"),
    ]
)
