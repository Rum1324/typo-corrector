#!/bin/sh
# Builds Typo.app (release) into dist/. Signs with your Apple Development identity when one
# exists so macOS keeps the Accessibility / Input Monitoring grants across rebuilds.
# Extra arguments go to `swift build`. Environment: BUNDLE_ID, IDENTITY ("-" = ad hoc),
# BUILD (scratch path), APP (output bundle); scripts/make_release.sh sets them for a public build.
set -e
cd "$(dirname "$0")/.."
# Optional local settings that stay out of git, e.g. BUNDLE_ID=com.example.typo to keep the
# permissions macOS granted to an earlier build under another identifier.
[ -z "$TYPO_RELEASE" ] && [ -f scripts/local.env ] && . scripts/local.env
BUNDLE_ID=${BUNDLE_ID:-io.github.rum1324.typo}
BUILD=${BUILD:-.build}
APP=${APP:-dist/Typo.app}
swift build -c release --product Typo --scratch-path "$BUILD" "$@"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BUILD/release/Typo" "$APP/Contents/MacOS/Typo"
# Debug symbols carry the absolute paths of the source files on the build machine.
strip -S "$APP/Contents/MacOS/Typo" 2>/dev/null
cp Resources/words_en.txt Resources/misspellings_en.txt "$APP/Contents/Resources/"
[ -f Resources/ranker.json ] && cp Resources/ranker.json Resources/ranker.bin "$APP/Contents/Resources/"
# Context path: language model + decider. Without them the app decides by the rules alone.
[ -f Resources/lm.json ] && [ -f Resources/context.json ] && cp Resources/lm.json Resources/lm.bin Resources/context.json \
    Resources/forms_en.tsv Resources/words_more_en.tsv Resources/ranker2.json Resources/ranker2.bin "$APP/Contents/Resources/"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>Typo</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleName</key><string>Typo</string>
  <key>CFBundleDisplayName</key><string>Typo</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHumanReadableCopyright</key><string>Local-only typo correction.</string>
</dict></plist>
PLIST
[ -z "${IDENTITY+set}" ] && IDENTITY=$(security find-identity -v -p codesigning 2>/dev/null | grep -o '"Apple Development:[^"]*"' | head -1 | tr -d '"')
if [ -n "$IDENTITY" ] && [ "$IDENTITY" != "-" ]; then
  codesign --force --deep --sign "$IDENTITY" --options runtime --entitlements scripts/Typo.entitlements "$APP"
  echo "signed with $IDENTITY"
else
  codesign --force --deep --sign - "$APP"
  echo "ad-hoc signed (permissions may need re-granting after each rebuild)"
fi
echo "built $APP"
