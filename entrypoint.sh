#!/bin/bash
# Starts the right Hadoop/Spark daemons depending on ROLE (master | worker)
set -e

if [ "$ROLE" = "master" ]; then
    # Format HDFS only on the very first start (data lives in a Docker volume)
    if [ ! -f /data/namenode/current/VERSION ]; then
        echo ">> First start: formatting NameNode"
        hdfs namenode -format -force -nonInteractive
    fi
    hdfs --daemon start namenode
    hdfs --daemon start secondarynamenode
    yarn --daemon start resourcemanager

    hdfs dfsadmin -safemode wait
    hdfs dfs -mkdir -p /spark-logs /user/root
    start-history-server.sh

elif [ "$ROLE" = "worker" ]; then
    hdfs --daemon start datanode
    yarn --daemon start nodemanager
else
    echo "ROLE must be 'master' or 'worker'"; exit 1
fi

echo ">> $ROLE ($(hostname)) is up"
# Keep the container alive and stream daemon logs to `docker logs`
exec tail -F $HADOOP_HOME/logs/*.log
