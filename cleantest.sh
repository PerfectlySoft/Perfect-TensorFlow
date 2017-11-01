#!/bin/bash
# This script is for CI Server
rm -rf .build
rm -rf Package.pins
rm -rf Package.resolved
time swift build
time swift build -c release
time swift test > test-results.txt
cat test-results.txt
