#!/bin/bash
chmod +x run-zeek.sh
echo "Running Zeek tool for collecting the logging information..."
export PATH=/usr/local/zeek/bin:$PATH
export LogDir=/home/gubh001/ENViSEC/collected/logs
sudo chmod +777 /usr/local/zeek/spool
open /usr/local/zeek/etc/node.cfg
open /usr/local/zeek/logs
zeekctl
