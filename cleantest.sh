#!/bin/bash
# This script is for CI Server
echo "clean up"
rm -rf .build
rm -rf Package.pins
rm -rf Package.resolved
echo "build release"
time swift build -c release
echo "perform test"
time swift test > test-results.txt
cat test-results.txt
