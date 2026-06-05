#!/bin/bash
# Tag a release and push. Usage: ./scripts/release.sh 1.0.0
set -euo pipefail
VERSION=${1:-"1.0.0"}
cd "$(dirname "$0")/.."
echo "Releasing version $VERSION"
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
git add pyproject.toml
git commit -m "chore(release): bump version to $VERSION" || true
git tag -a "v$VERSION" -m "Release v$VERSION"
git push origin HEAD --tags
echo "Tagged v$VERSION"
