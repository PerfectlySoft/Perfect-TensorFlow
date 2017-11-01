#!/bin/bash
# This script is for CI Server
./prepare.sh && ./install.sh /tmp/testdata && ./cleantest.sh
