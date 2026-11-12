#!/bin/sh
# Builds dist/release/Typo.zip for sharing. Unlike a plain make_app.sh run, the result carries
# nothing from the machine it was built on: the signature is ad hoc (a developer certificate
# would put its owner's name in the app), scripts/local.env is ignored, so the bundle identifier
# is the public default, and the build fails if a local path turns up inside the app.
# macOS will ask whoever downloads it to allow the first launch in Privacy & Security.
# Extra arguments go to `swift build` (for example -j 4).
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd -P)
export TYPO_RELEASE=1 IDENTITY=- BUNDLE_ID=io.github.rum1324.typo APP=dist/release/Typo.app
sh scripts/make_app.sh "$@"
if grep -r -a -q -F -e "$HOME" -e "$ROOT" "$APP"; then
  echo "error: $APP still contains a path from this machine" >&2
  exit 1
fi
ZIP=dist/release/Typo.zip
[ -f "$ZIP" ] && rm "$ZIP"
# UTC, so the entry times do not show the build machine's time zone
TZ=UTC ditto -c -k --keepParent "$APP" "$ZIP"
echo "built $ZIP ($(du -h "$ZIP" | cut -f1 | tr -d ' '))"
